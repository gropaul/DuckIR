#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction_data.hpp"
#include "index/vss/vss_index.hpp"

namespace duckdb {

void VSSIndex::UpdateReservedMemory(const int64_t delta) {
	if (delta > 0) {
		buffer_manager.ReserveMemory(static_cast<idx_t>(delta));
		reserved_memory_bytes += static_cast<idx_t>(delta);
	} else if (delta < 0) {
		buffer_manager.FreeReservedMemory(static_cast<idx_t>(-delta));
		reserved_memory_bytes -= static_cast<idx_t>(-delta);
	}
}

bool VSSIndex::IsInSyncWithTable(DataTable &table) const {
	return !bind_error.HasError() && !needs_reconciliation && !has_unindexed_rows && !has_staged_deletes &&
	       FindStaleRowGroups(table).empty();
}

unique_ptr<StorageLockKey> VSSIndex::SyncAndLockForSearch(ClientContext &context, DataTable &table) {
	bool in_sync;
	{
		auto _lock = rwlock.GetSharedLock();
		in_sync = IsInSyncWithTable(table);
	}
	if (!in_sync) {
		SyncWithTable(context, table);
	}
	return rwlock.GetSharedLock();
}

idx_t VSSIndex::GetRowGroupSize() const {
	return table_io_manager.GetRowGroupSize();
}

bool VSSIndex::TryGetPhysicalRowGroup(DataTable &table, const row_t row_id, VSSRowGroupBounds &result) const {
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	idx_t segment_index;
	if (!row_groups->TryGetSegmentIndex(lock, static_cast<idx_t>(row_id), segment_index)) {
		return false;
	}
	auto node = row_groups->GetSegmentByIndex(lock, static_cast<int64_t>(segment_index));
	result = {static_cast<row_t>(node->GetRowStart()), node->GetCount()};
	return true;
}

bool VSSIndex::IsClosedRowGroup(DataTable &table, const VSSRowGroupBounds &row_group) const {
	VSSRowGroupBounds next_row_group;
	return row_group.count >= GetRowGroupSize() ||
	       TryGetPhysicalRowGroup(table, row_group.row_start + static_cast<row_t>(row_group.count), next_row_group);
}

vector<VSSRowGroupBounds> VSSIndex::GetPhysicalRowGroups(DataTable &table) const {
	vector<VSSRowGroupBounds> result;
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	for (auto node = row_groups->GetRootSegment(lock); node; node = row_groups->GetNextSegment(lock, *node)) {
		result.push_back({static_cast<row_t>(node->GetRowStart()), node->GetCount()});
	}
	return result;
}

idx_t VSSIndex::GetLargestRowGroupCount(DataTable &table) const {
	// At least the attach's row group size, which an open row group can still grow to.
	idx_t largest = GetRowGroupSize();
	for (const auto &row_group : GetPhysicalRowGroups(table)) {
		largest = MaxValue(largest, row_group.count);
	}
	return largest;
}

idx_t VSSIndex::FetchCommittedRowIds(DataTable &table, const row_t start, const row_t end, row_t *const row_ids) {
	const TransactionData committed(MAX_TRANSACTION_ID,
	                                VisibilityBound::Through(DuckTransactionManager::Get(db).GetLastCommit()));
	const vector<StorageIndex> column_ids {StorageIndex()};
	TableScanState scan_state;
	scan_state.Initialize(column_ids);
	table.GetRowGroupCollection()->InitializeScanWithOffset(QueryContext(), scan_state.table_state, column_ids,
	                                                        static_cast<idx_t>(start), static_cast<idx_t>(end));
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(db), {LogicalType::ROW_TYPE});
	idx_t count = 0;
	while (true) {
		chunk.Reset();
		scan_state.table_state.Scan(ScanOptions(committed), chunk);
		if (chunk.size() == 0) {
			break;
		}
		chunk.data[0].Flatten();
		memcpy(row_ids + count, FlatVector::GetData<row_t>(chunk.data[0]), chunk.size() * sizeof(row_t));
		count += chunk.size();
	}
	return count;
}

// One DuckDB row group of row ids at a time
void VSSIndex::ReconcileWithTable(DataTable &table) {
	auto row_ids = make_uniq_array<row_t>(GetLargestRowGroupCount(table));
	const auto stage = [&](const row_t start, const row_t end) {
		StageUnindexedRange(unindexed_row_ranges, {start, end});
	};
	for (const auto &physical : GetPhysicalRowGroups(table)) {
		if (physical.count == 0) {
			continue;
		}
		const row_t end = physical.row_start + static_cast<row_t>(physical.count);
		const idx_t count = FetchCommittedRowIds(table, physical.row_start, end, row_ids.get());
		ReconcileRowGroup(physical.row_start, end, row_ids.get(), count, stage);
	}
}

ErrorData VSSIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	// Only the row ids are needed, so the index expressions are not evaluated.
	return Insert(lock, entries, row_ids);
}

// Called by DuckDB at commit with the final row ids, before the rows enter the table's row groups. Which row group
// they land in is not knowable here, so only the ids are recorded; SyncWithTable indexes them once the table knows.
ErrorData VSSIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	// A commit never waits for a search
	const lock_guard<mutex> staging(commit_staging_lock);

	const idx_t count = data.size();
	if (count == 0) {
		return ErrorData();
	}
	try {
		row_ids.Flatten();
		const auto row_id_data = FlatVector::GetData<row_t>(row_ids);
		const auto stage = [&](const VSSRowRange &staged) {
			StageUnindexedRange(staged_row_ranges, staged);
			// Appends land past the persisted row groups: a row id below persisted_row_end was reused.
			if (staged.start < persisted_row_end) {
				first_reused_row_id = MinValue(first_reused_row_id, staged.start);
			}
		};
		// Determine consecutive ranges of row ids.
		// For example: `1, 2, 3, 5, 6 --> [1, 3] + [5, 6]`
		VSSRowRange range {row_id_data[0], row_id_data[0] + 1};
		for (idx_t i = 1; i < count; i++) {
			if (row_id_data[i] == range.end) {
				range.end++;
				continue;
			}
			stage(range);
			range = {row_id_data[i], row_id_data[i] + 1};
		}
		stage(range);
		has_unindexed_rows = true;
	} catch (...) {
		// DuckDB turns an exception here into a fatal error: the next sync reconciles with the table instead
		RequireReconciliation();
	}
	return ErrorData();
}

// Called by DuckDB at commit. Row ids that are not in the index (never indexed, or reverted before they were) are
// skipped.
void VSSIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	// A commit never waits for a search. Before the first sync after a load or CREATE INDEX, deletes wait for it too:
	// its reconcile reads the table as of the last published commit, where the rows of this commit are still there.
	auto search_lock = rwlock.TryGetExclusiveLock();
	if (search_lock && needs_reconciliation) {
		search_lock.reset();
	}
	const lock_guard<mutex> staging(commit_staging_lock);

	try {
		const idx_t count = entries.size();
		row_ids.Flatten();
		const auto row_id_data = FlatVector::GetData<row_t>(row_ids);
		RemoveUnindexedRows(staged_row_ranges, row_id_data, count);
		staged_deletes.insert(staged_deletes.end(), row_id_data, row_id_data + count);
		if (!search_lock) {
			has_staged_deletes = true;
			return;
		}
		// Holding the lock: this commit's deletes, and those staged meanwhile.
		ApplyStagedDeletes(staged_deletes.data(), staged_deletes.size());
		staged_deletes.clear();
		has_staged_deletes = false;
		has_unindexed_rows = !unindexed_row_ranges.empty() || !staged_row_ranges.empty();
	} catch (...) {
		// DuckDB turns an exception here into a fatal error: the staged deletes wait for the next sync, which also
		// reconciles with the table
		has_staged_deletes = !staged_deletes.empty();
		RequireReconciliation();
	}
}

void VSSIndex::StageUnindexedRange(std::vector<VSSRowRange> &ranges, const VSSRowRange range) {
	// The first staged range that starts after this one.
	auto it = std::upper_bound(ranges.begin(), ranges.end(), range.start,
	                           [](row_t start, const VSSRowRange &staged) { return start < staged.start; });
	if (it != ranges.begin() && std::prev(it)->end > range.start) {
		--it;
		it->end = MaxValue<row_t>(it->end, range.end);
	} else {
		it = ranges.insert(it, range);
	}
	// Absorb the following ranges this one now overlaps.
	auto next = std::next(it);
	while (next != ranges.end() && next->start < it->end) {
		it->end = MaxValue<row_t>(it->end, next->end);
		next = ranges.erase(next);
	}
}

// Deleted rows that were never indexed have nothing left to index: take them out of their staged ranges. Only the
// ranges between the smallest and the largest row id are rebuilt, and spliced back at once.
void VSSIndex::RemoveUnindexedRows(std::vector<VSSRowRange> &ranges, const row_t *const row_ids, const idx_t count) {
	if (ranges.empty() || count == 0) {
		return;
	}
	std::vector<row_t> sorted_ids(row_ids, row_ids + count);
	std::sort(sorted_ids.begin(), sorted_ids.end());
	// The first range that ends after the smallest row id, and the first that starts after the largest one.
	const auto first = std::upper_bound(ranges.begin(), ranges.end(), sorted_ids.front(),
	                                    [](row_t id, const VSSRowRange &range) { return id < range.end; });
	const auto last = std::upper_bound(first, ranges.end(), sorted_ids.back(),
	                                   [](row_t id, const VSSRowRange &range) { return id < range.start; });
	if (first == last) {
		return;
	}
	std::vector<VSSRowRange> rebuilt;
	auto id = sorted_ids.begin();
	for (auto it = first; it != last; ++it) {
		row_t start = it->start;
		for (; id != sorted_ids.end() && *id < it->end; ++id) {
			// Row ids in the gap before the range, or repeated, are skipped
			if (*id < start) {
				continue;
			}
			if (*id > start) {
				rebuilt.push_back({start, *id});
			}
			start = *id + 1;
		}
		if (start < it->end) {
			rebuilt.push_back({start, it->end});
		}
	}
	const auto position = ranges.erase(first, last);
	ranges.insert(position, rebuilt.begin(), rebuilt.end());
}

void VSSIndex::ApplyStagedDeletes(const row_t *const row_ids, const idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		// PDX Delete is idempotent: if the row was never indexed, it is a no-op
		DeleteRow(row_ids[i]);
	}
	RemoveUnindexedRows(unindexed_row_ranges, row_ids, count);
}

bool VSSIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("VSSIndex::MergeIndexes() not implemented");
}

void VSSIndex::Vacuum(IndexLock &state) {
}

void VSSIndex::Verify(IndexLock &state) {
	throw NotImplementedException("VSSIndex::Verify() not implemented");
}

string VSSIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("VSSIndex::ToString() not implemented");
}

void VSSIndex::VerifyAllocations(IndexLock &state) {
	throw NotImplementedException("VSSIndex::VerifyAllocations() not implemented");
}

// Called in debug builds: at most one empty buffer.
void VSSIndex::VerifyBuffers(IndexLock &state) {
	allocator->VerifyBuffers();
}

// IndexStorageInfo cannot be copied. A checkpointed index's info has no WAL buffers.
IndexStorageInfo VSSIndex::CopyStorageInfo(const IndexStorageInfo &info) {
	IndexStorageInfo copy(info.name);
	copy.root = info.root;
	copy.options = info.options;
	copy.allocator_infos = info.allocator_infos;
	copy.root_block_ptr = info.root_block_ptr;
	return copy;
}

// Like ART::WritePartialBlocks. Flushing every time keeps a later New() from landing in a buffer whose block is not on
// disk yet. Empty buffers go first: DuckDB asserts that the buffers it serializes are not empty.
void VSSIndex::WritePartialBlocks(QueryContext context) {
	allocator->RemoveEmptyBuffers();
	PartialBlockManager partial_block_manager(context, table_io_manager.GetIndexBlockManager(),
	                                          PartialBlockType::FULL_CHECKPOINT);
	allocator->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
}

void VSSIndex::WriteDirectory(const VSSDirectory &directory) {
	directory_chain.Free(*allocator);
	VSSBlockChainWriter writer(*allocator);
	VSSChainOutputStream out(writer);
	directory.Write(out);
	directory_chain = writer.Finish();
}

} // namespace duckdb
