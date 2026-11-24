#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction_data.hpp"
#include "index/vss/vss_index.hpp"
#include "index/vss/vss_parallel_executor.hpp"

namespace duckdb {

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

optional_idx VSSIndex::LookupRowGroup(const row_t row_id) const {
	return DispatchWrapper([&](auto &wrapper) { return wrapper.LookupRowGroup(row_id); });
}

bool VSSIndex::IsEmptyRowGroup(const idx_t row_group_idx) const {
	return DispatchWrapper([&](auto &wrapper) { return wrapper.IsEmptyRowGroup(row_group_idx); });
}

VSSRowRange VSSIndex::GetRowGroupRange(const idx_t row_group_idx) const {
	return DispatchWrapper([&](auto &wrapper) {
		const auto &row_group = wrapper.GetRowGroup(row_group_idx);
		return VSSRowRange {row_group.row_start, row_group.row_end};
	});
}

void VSSIndex::AppendRows(const idx_t row_group_idx, const row_t *const row_ids, float *const transformed_embeddings,
                          const idx_t count, PDX::ParallelExecutor *const executor) {
	DispatchWrapper([&](auto &wrapper) {
		wrapper.AppendRows(row_group_idx, row_ids, transformed_embeddings, count, storage_reader.get(), buffer_manager,
		                   executor);
	});
}

void VSSIndex::DeleteRow(const row_t row_id) {
	DispatchWrapper([&](auto &wrapper) { wrapper.DeleteRow(row_id); });
}

vector<VSSRowGroupBounds> VSSIndex::FindStaleRowGroups(DataTable &table) const {
	const auto physical_row_groups = GetPhysicalRowGroups(table);
	return DispatchWrapper([&](auto &wrapper) { return wrapper.FindStaleRowGroups(physical_row_groups); });
}

void VSSIndex::MarkClosedRowGroups(DataTable &table) {
	const auto is_closed = [&](const row_t row_start) {
		VSSRowGroupBounds row_group;
		return TryGetPhysicalRowGroup(table, row_start, row_group) && IsClosedRowGroup(table, row_group);
	};
	DispatchWrapper([&](auto &wrapper) { wrapper.MarkClosedRowGroups(is_closed); });
}

void VSSIndex::RemoveRowGroupsOverlapping(const row_t start, const row_t end) {
	DispatchWrapper([&](auto &wrapper) { wrapper.RemoveRowGroupsOverlapping(start, end); });
}

void VSSIndex::DeleteStaleRowIds(const std::vector<VSSRowRange> &ranges) {
	DispatchWrapper([&](auto &wrapper) {
		for (const auto &range : ranges) {
			wrapper.DeleteStaleRowIds(range.start, range.end);
		}
	});
}

idx_t VSSIndex::FetchRows(DataTable &table, const std::vector<VSSRowRange> &ranges, float *const embeddings,
                          std::vector<idx_t> &rows_returned, row_t *const row_ids, const bool rotate,
                          PDX::ParallelExecutor *const executor, const idx_t max_rows, const bool all_committed) {
	rows_returned.assign(ranges.size(), 0);
	// Scans every committed row, whatever the calling transaction's snapshot. The mirror
	// holds the committed state, and rows a query must not see are dropped when it fetches its results.
	const TransactionData committed(
	    MAX_TRANSACTION_ID, all_committed ? VisibilityBound::AllCommitted()
	                                      : VisibilityBound::Through(DuckTransactionManager::Get(db).GetLastCommit()));
	const vector<StorageIndex> column_ids {StorageIndex(GetColumnIds()[0]), StorageIndex()};
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(db), {logical_types[0], LogicalType::ROW_TYPE});
	const auto num_dimensions = GetNumDimensions();
	auto raw_embeddings = make_uniq_array_uninitialized<float>(STANDARD_VECTOR_SIZE * num_dimensions);

	idx_t count = 0;
	idx_t range_idx = 0;
	while (range_idx < ranges.size() && count < max_rows) {
		if (ranges[range_idx].start >= ranges[range_idx].end) {
			range_idx++;
			continue;
		}
		// Ranges less than a vector apart share one scan: the scan of each would read that vector anyway.
		idx_t last_range_idx = range_idx;
		while (last_range_idx + 1 < ranges.size() && ranges[last_range_idx + 1].start - ranges[last_range_idx].end <
		                                                 static_cast<row_t>(STANDARD_VECTOR_SIZE)) {
			last_range_idx++;
		}
		TableScanState scan_state;
		scan_state.Initialize(column_ids);
		table.GetRowGroupCollection()->InitializeScanWithOffset(QueryContext(), scan_state.table_state, column_ids,
		                                                        static_cast<idx_t>(ranges[range_idx].start),
		                                                        static_cast<idx_t>(ranges[last_range_idx].end));
		while (count < max_rows) {
			chunk.Reset();
			// Deleted rows are gone, NULL embeddings are NULL.
			scan_state.table_state.Scan(ScanOptions(committed), chunk);
			if (chunk.size() == 0) {
				break;
			}
			auto &embedding_column = chunk.data[0];
			embedding_column.Flatten();
			chunk.data[1].Flatten();
			const auto &validity = FlatVector::Validity(embedding_column);
			const auto scanned_embeddings = FlatVector::GetData<float>(ArrayVector::GetChild(embedding_column));
			const auto scanned_row_ids = FlatVector::GetData<row_t>(chunk.data[1]);
			// Without a rotation, the rows go straight to the output.
			float *const batch_embeddings = rotate ? raw_embeddings.get() : embeddings + count * num_dimensions;
			idx_t batch_count = 0;
			const idx_t chunk_size = chunk.size();
			for (idx_t i = 0; i < chunk_size && count + batch_count < max_rows; i++) {
				// The scan starts at the vector that holds the first range's start, and reads the gaps between ranges.
				while (range_idx < last_range_idx && ranges[range_idx].end <= scanned_row_ids[i]) {
					range_idx++;
				}
				if (scanned_row_ids[i] < ranges[range_idx].start || scanned_row_ids[i] >= ranges[range_idx].end) {
					continue;
				}
				rows_returned[range_idx]++;
				// Like NULL embeddings, the ones CREATE INDEX rejects are not indexed.
				if (!validity.RowIsValid(i) ||
				    !IsFiniteEmbedding(scanned_embeddings + i * num_dimensions, num_dimensions)) {
					continue;
				}
				memcpy(batch_embeddings + batch_count * num_dimensions, scanned_embeddings + i * num_dimensions,
				       num_dimensions * sizeof(float));
				if (row_ids) {
					row_ids[count + batch_count] = scanned_row_ids[i];
				}
				batch_count++;
			}
			if (batch_count == 0) {
				continue;
			}
			if (rotate) {
				vss_wrapper->PreprocessEmbeddings(raw_embeddings.get(), embeddings + count * num_dimensions,
				                                  batch_count, executor);
			} else {
				vss_wrapper->NormalizeEmbeddings(batch_embeddings, batch_count);
			}
			count += batch_count;
		}
		range_idx = last_range_idx + 1;
	}
	return count;
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

void VSSIndex::ReconcileRowGroup(const row_t start, const row_t end, const row_t *const row_ids, const idx_t count,
                                 const std::function<void(row_t, row_t)> &stage) {
	DispatchWrapper([&](auto &wrapper) { wrapper.ReconcileRowGroup(start, end, row_ids, count, stage); });
}

void VSSIndex::SyncWithTable(ClientContext &context, DataTable &table) {
	auto _lock = rwlock.GetExclusiveLock();

	// The index failed to load when DuckDB bound it: the queries that use it raise the error. Rethrowing an INTERNAL
	// (or FATAL) error as it is would make DuckDB invalidate the whole database: every later query, on any table, would
	// fail until a restart. It is rethrown as an IO error instead, so that only the queries on this index fail.
	if (bind_error.HasError()) {
		if (bind_error.Type() == ExceptionType::INTERNAL || bind_error.Type() == ExceptionType::FATAL) {
			throw IOException("DuckIR VSS index %s could not be loaded: %s", name, bind_error.RawMessage());
		}
		bind_error.Throw();
	}

	DuckDBParallelExecutor executor(context,
	                                static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads()));

	// One DuckDB row group of transformed embeddings at a time, like the create sink, in a DuckDB buffer of the rows
	// fetched.
	const auto num_dimensions = GetNumDimensions();
	const auto allocate_embeddings = [&](const idx_t num_rows) {
		return buffer_manager.Allocate(MemoryTag::EXTENSION, num_rows * num_dimensions * sizeof(float),
		                               /*can_destroy=*/false);
	};
	auto row_ids = make_uniq_array<row_t>(GetLargestRowGroupCount(table));

	// Row groups a checkpoint merged or dropped: their mirrors go, and the ones still in the table are rebuilt.
	std::vector<idx_t> rows_returned;
	for (const auto &stale : FindStaleRowGroups(table)) {
		RemoveRowGroupsOverlapping(stale.row_start, stale.row_start + static_cast<row_t>(stale.count));
		VSSRowGroupBounds bounds;
		if (!TryGetPhysicalRowGroup(table, stale.row_start, bounds)) {
			continue;
		}
		const row_t bounds_end = bounds.row_start + static_cast<row_t>(bounds.count);
		try {
			auto embeddings_buffer = allocate_embeddings(bounds.count);
			const auto embeddings = reinterpret_cast<float *>(embeddings_buffer.GetDataMutable());
			// As of every commit: a DELETE in flight already removed some of these rows from the index. The rows of an
			// INSERT in flight it brings are in the inbox, whose ids DeleteStaleRowIds evicts later in this sync.
			const idx_t count = FetchRows(table, {{bounds.row_start, bounds_end}}, embeddings, rows_returned,
			                              row_ids.get(), /*rotate=*/true, &executor, NumericLimits<idx_t>::Maximum(),
			                              /*all_committed=*/true);
			if (count > 0) {
				SetUpIndexForRowGroup(row_ids.get(), embeddings, count, bounds.row_start, bounds.count,
				                      IsClosedRowGroup(table, bounds), &executor);
			}
		} catch (...) {
			// A failed rebuild leaves no mirror, and its rows staged: the next sync builds them.
			RemoveRowGroupsOverlapping(bounds.row_start, bounds_end);
			StageUnindexedRange(unindexed_row_ranges, {bounds.row_start, bounds_end});
			has_unindexed_rows = true;
			throw;
		}
	}

	// The first query after the first index bind or CREATE INDEX, or after a change the index could not take (a merge
	// that ran out of memory, a failed Insert or Delete). Taken before the reconcile: a request meanwhile stays set.
	if (needs_reconciliation.exchange(false)) {
		try {
			ReconcileWithTable(table);
		} catch (...) {
			needs_reconciliation = true;
			throw;
		}
	}

	// The commits' inbox. The deletes go first: the rebuilds and the reconcile above read the table as of the last
	// published commit, where a row deleted by a commit in flight is still there, and the new row of a reused row id is
	// in the inbox's ranges.
	{
		std::vector<VSSRowRange> committed_ranges;
		std::vector<row_t> committed_deletes;
		{
			const lock_guard<mutex> staging(commit_staging_lock);
			std::swap(committed_ranges, staged_row_ranges);
			std::swap(committed_deletes, staged_deletes);
			has_staged_deletes = false;
		}
		ApplyStagedDeletes(committed_deletes.data(), committed_deletes.size());
		for (const auto &range : committed_ranges) {
			StageUnindexedRange(unindexed_row_ranges, range);
		}
		// New rows at reused ids replace the stale ones a mirror still holds
		DeleteStaleRowIds(unindexed_row_ranges);
	}

	// Unindexed rows, gathered per DuckDB row group: a row group without a mirror is clustered once from all of its
	// rows. Each range is retired on its own because rows of one commit become visible together.
	std::vector<VSSRowRange> still_unindexed;
	idx_t range_idx = 0;
	while (range_idx < unindexed_row_ranges.size()) {
		VSSRowGroupBounds bounds;
		if (!TryGetPhysicalRowGroup(table, unindexed_row_ranges[range_idx].start, bounds)) {
			// The rows are not in the table yet: their commit is still in flight.
			still_unindexed.push_back(unindexed_row_ranges[range_idx++]);
			continue;
		}
		const row_t bounds_end = bounds.row_start + static_cast<row_t>(bounds.count);
		idx_t num_unindexed_rows = 0;
		for (idx_t i = range_idx; i < unindexed_row_ranges.size() && unindexed_row_ranges[i].start < bounds_end; i++) {
			const auto &range = unindexed_row_ranges[i];
			num_unindexed_rows += static_cast<idx_t>(MinValue<row_t>(range.end, bounds_end) - range.start);
		}
		auto row_group_idx = LookupRowGroup(bounds.row_start);
		std::vector<VSSRowRange> row_group_ranges;
		for (idx_t i = range_idx; i < unindexed_row_ranges.size() && unindexed_row_ranges[i].start < bounds_end; i++) {
			row_group_ranges.push_back(
			    {unindexed_row_ranges[i].start, MinValue<row_t>(unindexed_row_ranges[i].end, bounds_end)});
		}
		try {
			auto embeddings_buffer = allocate_embeddings(num_unindexed_rows);
			const auto embeddings = reinterpret_cast<float *>(embeddings_buffer.GetDataMutable());
			// One scan for the row group's ranges, each of them retired on its own
			const idx_t count = FetchRows(table, row_group_ranges, embeddings, rows_returned, row_ids.get(),
			                              /*rotate=*/true, &executor);
			for (idx_t i = 0; i < row_group_ranges.size(); i++) {
				if (rows_returned[i] == 0) {
					// Not committed yet: a later sync indexes them.
					still_unindexed.push_back(row_group_ranges[i]);
				}
			}
			while (range_idx < unindexed_row_ranges.size() && unindexed_row_ranges[range_idx].start < bounds_end) {
				// A range that continues past this row group goes on in the next one
				if (unindexed_row_ranges[range_idx].end > bounds_end) {
					unindexed_row_ranges[range_idx].start = bounds_end;
					break;
				}
				range_idx++;
			}
			if (count == 0) {
				continue;
			}
			// Every row of this mirror was deleted from our index:
			// we remove the mirror and rebuild it from the new rows
			if (row_group_idx.IsValid() && IsEmptyRowGroup(row_group_idx.GetIndex())) {
				RemoveRowGroupsOverlapping(bounds.row_start, bounds_end);
				row_group_idx = optional_idx();
			}
			if (row_group_idx.IsValid()) {
				AppendRows(row_group_idx.GetIndex(), row_ids.get(), embeddings, count, &executor);
			} else {
				SetUpIndexForRowGroup(row_ids.get(), embeddings, count, bounds.row_start, bounds.count,
				                      IsClosedRowGroup(table, bounds), &executor);
			}
		} catch (...) {
			// The row group's staged rows stay staged: the next sync appends them again (rows already appended are
			// skipped) or builds them. A failed build leaves no mirror. A merge before the append that ran out of
			// memory can have dropped rows of the mirror: the next sync reconciles them back.
			if (!row_group_idx.IsValid()) {
				RemoveRowGroupsOverlapping(bounds.row_start, bounds_end);
			} else {
				needs_reconciliation = true;
			}
			for (const auto &range : row_group_ranges) {
				StageUnindexedRange(unindexed_row_ranges, range);
			}
			has_unindexed_rows = true;
			throw;
		}
	}
	unindexed_row_ranges = std::move(still_unindexed);
	{
		const lock_guard<mutex> staging(commit_staging_lock);
		has_unindexed_rows = !unindexed_row_ranges.empty() || !staged_row_ranges.empty();
	}

	// We need to close rowgroups before the temporary chains are written so that they are written without the appends
	// headroom.
	MarkClosedRowGroups(table);

	// The closed row groups the sync loaded fully go back to temporary chains; the open one stays on the heap.
	if (cluster_paging) {
		WriteTemporaryChains();
	}

	// The sync's appends, deletes and removed row groups changed the self-managed memory without reserving it.
	// Reserving the difference between the size now and what is reserved makes the two equal (it frees when the index
	// shrank).
	const auto self_managed_bytes = static_cast<int64_t>(vss_wrapper->GetSelfManagedMemorySizeInBytes());
	UpdateReservedMemory(self_managed_bytes - static_cast<int64_t>(reserved_memory_bytes.load()));
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

/******************************************************************
 * Index persistence
 ******************************************************************/

unique_ptr<VSSBlockChainReader> VSSIndex::OpenStorage(const IndexStorageInfo &info, const uint32_t num_dimensions,
                                                      VSSDirectory &directory, unique_ptr<float[]> &rotation_matrix,
                                                      std::optional<PDX::SAQModel> &saq_model) {
	const auto version = info.options.find("storage_version");
	if (version == info.options.end() || version->second.GetValue<uint32_t>() != VSS_STORAGE_VERSION) {
		throw NotImplementedException(
		    "The DuckIR VSS index \"%s\" was persisted in a storage format this version of the "
		    "extension cannot read. Update the extension, or DROP INDEX and create it again.",
		    name);
	}
	const auto &allocator_info = info.allocator_infos[0];
	if (allocator_info.segment_size != allocator->GetSegmentSize()) {
		throw SerializationException("The DuckIR VSS index \"%s\" has segments of %llu bytes instead of %llu", name,
		                             allocator_info.segment_size, allocator->GetSegmentSize());
	}
	// Registers the persisted blocks without reading them: the next checkpoints keep or free them.
	allocator->Init(allocator_info);

	auto reader = make_uniq<VSSBlockChainReader>(table_io_manager.GetIndexBlockManager(), allocator_info);
	// Only DuckDB's WAL replay of a READ_ONLY database sets transient_block_handles (ReplayIndexData), and the only
	// index data this index writes to the WAL is the empty directory of SerializeToWAL (CREATE INDEX). Those buffers
	// have no block to read: there is nothing to load, and the first sync builds the index from the table.
	if (allocator_info.transient_block_handles) {
		directory.num_dimensions = num_dimensions;
		return reader;
	}
	VSSChainInputStream in(*reader);
	VSSBlockChain root;
	root.head.Set(info.root);
	root.num_bytes = info.options.at("directory_bytes").GetValue<idx_t>();
	reader->Open(root);
	directory = VSSDirectory::Read(in);
	directory_chain = reader->Finish();
	if (directory.num_dimensions != num_dimensions) {
		throw SerializationException("The DuckIR VSS index \"%s\" was persisted with %u dimensions instead of %u", name,
		                             directory.num_dimensions, num_dimensions);
	}
	// A directory written to the WAL has no rotation.
	if (directory.rotation.num_bytes > 0) {
		const size_t num_values = static_cast<size_t>(num_dimensions) * num_dimensions;
		rotation_matrix = make_uniq_array<float>(num_values);
		reader->Open(directory.rotation);
		PDX::StreamReader rotation_reader {in};
		rotation_reader.Read(rotation_matrix.get(), sizeof(float) * num_values);
		rotation_chain = reader->Finish();
	}
	if (directory.saq_model.num_bytes > 0) {
		reader->Open(directory.saq_model);
		PDX::StreamReader saq_model_reader {in};
		saq_model.emplace();
		saq_model->Load(saq_model_reader);
		saq_model_chain = reader->Finish();
	}
	return reader;
}

// Reserves each row group's self-managed memory as it loads, so DuckDB evicts other data while the index grows. PDX's
// buffers are charged as it allocates them.
void VSSIndex::LoadRowGroups(VSSBlockChainReader &reader, const VSSDirectory &directory, const bool page_clusters) {
	for (const auto &entry : directory.row_groups) {
		const uint64_t self_managed_bytes = DispatchWrapper(
		    [&](auto &wrapper) { return wrapper.LoadRowGroup(reader, entry, buffer_manager, page_clusters); });
		UpdateReservedMemory(static_cast<int64_t>(self_managed_bytes));
	}
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

void VSSIndex::PersistDirtyRowGroups(const std::function<void(idx_t persisted_bytes)> &write_partial_blocks) {
	const bool merge_ran_out_of_memory = DispatchWrapper([&](auto &wrapper) {
		return wrapper.PersistDirtyRowGroups(*allocator, write_partial_blocks, storage_reader.get(), buffer_manager,
		                                     cluster_paging);
	});
	if (merge_ran_out_of_memory) {
		needs_reconciliation = true;
	}
}

void VSSIndex::WriteTemporaryChains() {
	DispatchWrapper([&](auto &wrapper) { wrapper.WriteTemporaryChains(*storage_reader, buffer_manager); });
}

void VSSIndex::AddRowGroupEntries(VSSDirectory &directory) const {
	DispatchWrapper([&](auto &wrapper) { wrapper.AddRowGroupEntries(directory); });
}

void VSSIndex::ResetPersistedChains() {
	DispatchWrapper([&](auto &wrapper) { wrapper.ResetPersistedChains(); });
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

IndexStorageInfo VSSIndex::MakeStorageInfo() const {
	IndexStorageInfo info(name);
	info.root = directory_chain.head.Get();
	info.options.emplace("storage_version", Value::UINTEGER(VSS_STORAGE_VERSION));
	info.options.emplace("directory_bytes", Value::UBIGINT(directory_chain.num_bytes));
	info.options.emplace("metric", Value(GetDistanceMetric()));
	info.options.emplace("quantization", Value(GetQuantization()));
	info.options.emplace("n_probe", Value::INTEGER(static_cast<int32_t>(vss_wrapper->GetNProbe())));
	info.options.emplace("seed", Value::INTEGER(vss_wrapper->GetSeed()));
	info.options.emplace("cluster_sketches", Value::BOOLEAN(vss_wrapper->BuildsClusterSketches()));
	info.allocator_infos.push_back(allocator->GetInfo());
	return info;
}

// Called at every checkpoint, also when nothing changed: only the dirty row groups and the directory are rewritten.
IndexStorageInfo VSSIndex::SerializeToDisk(QueryContext context,
                                           const case_insensitive_map_t<Value> &serialization_options) {
	// Not loaded: the index stays as persisted.
	if (bind_error.HasError()) {
		return CopyStorageInfo(persisted_storage_info);
	}
	// DuckDB holds its checkpoint locks here, which a running search can wait for: a busy index keeps its persisted
	// directory and chains.
	auto _lock = rwlock.TryGetExclusiveLock();
	if (!_lock) {
		// Never checkpointed: it persists the empty directory it wrote to the WAL.
		if (rotation_chain.segments.empty()) {
			VSSDirectory directory;
			directory.num_dimensions = static_cast<uint32_t>(GetNumDimensions());
			WriteDirectory(directory);
		}
		WritePartialBlocks(context);
		auto info = MakeStorageInfo();
		const lock_guard<mutex> staging(commit_staging_lock);
		// The persisted chains still hold rows DuckDB dropped before reusing their ids: the next load drops those row
		// groups.
		if (first_reused_row_id < persisted_row_end) {
			info.options.emplace("first_reused_row_id", Value::BIGINT(first_reused_row_id));
		}
		return info;
	}
	// Reused ids no sync has seen yet: the inbox does not survive a restart, but the stale rows must not be persisted
	{
		const lock_guard<mutex> staging(commit_staging_lock);
		DeleteStaleRowIds(staged_row_ranges);
	}
	idx_t bytes_written_since_sync = 0;
	PersistDirtyRowGroups([&](const idx_t persisted_bytes) {
		WritePartialBlocks(context);
		bytes_written_since_sync += persisted_bytes;
		if (bytes_written_since_sync >= MAX_UNSYNCED_CHECKPOINT_BYTES) {
			table_io_manager.GetIndexBlockManager().FileSync();
			bytes_written_since_sync = 0;
		}
	});
	// A SAQ model that is not persisted yet comes from a fit since the last checkpoint, whose rotation replaced the
	// persisted one: both are written.
	const auto saq_model = vss_wrapper->GetSAQModel();
	const bool write_saq_model = saq_model && saq_model_chain.segments.empty();
	if (write_saq_model) {
		rotation_chain.Free(*allocator);
	}
	// Written once, and again after a SAQ fit.
	if (rotation_chain.segments.empty()) {
		const auto num_dimensions = GetNumDimensions();
		VSSBlockChainWriter writer(*allocator);
		VSSChainOutputStream out(writer);
		out.write(reinterpret_cast<const char *>(GetRotationMatrix()),
		          static_cast<std::streamsize>(sizeof(float) * num_dimensions * num_dimensions));
		rotation_chain = writer.Finish();
	}
	if (write_saq_model) {
		VSSBlockChainWriter writer(*allocator);
		VSSChainOutputStream out(writer);
		saq_model->Save(out);
		saq_model_chain = writer.Finish();
	}

	VSSDirectory directory;
	directory.num_dimensions = static_cast<uint32_t>(GetNumDimensions());
	directory.rotation = VSSBlockChain {rotation_chain.head, rotation_chain.num_bytes, {}};
	directory.saq_model = VSSBlockChain {saq_model_chain.head, saq_model_chain.num_bytes, {}};
	AddRowGroupEntries(directory);
	WriteDirectory(directory);
	WritePartialBlocks(context);
	// The checkpoint can have moved live segments to other blocks.
	if (storage_reader) {
		storage_reader->UpdateBlockPointers(allocator->GetInfo());
	}
	// The rewritten row groups were paged again: free what they no longer use. Reserving more could throw, which a
	// checkpoint must not: the next sync reserves any growth.
	const auto self_managed_bytes = static_cast<int64_t>(vss_wrapper->GetSelfManagedMemorySizeInBytes());
	const auto reserved = static_cast<int64_t>(reserved_memory_bytes.load());
	if (self_managed_bytes < reserved) {
		UpdateReservedMemory(self_managed_bytes - reserved);
	}
	// The chains just persisted hold no row whose id was reused.
	{
		const lock_guard<mutex> staging(commit_staging_lock);
		persisted_row_end = directory.row_groups.empty() ? 0 : directory.row_groups.back().row_end;
		first_reused_row_id = NumericLimits<row_t>::Maximum();
	}
	return MakeStorageInfo();
}

// Called when CREATE INDEX commits. The WAL only gets an empty directory and the options: if the process stops before
// the next checkpoint, replay binds an empty index and its first sync indexes the table. Logging the whole index would
// keep it in memory twice until that checkpoint (2x the index size).
IndexStorageInfo VSSIndex::SerializeToWAL(const case_insensitive_map_t<Value> &serialization_options) {
	auto _lock = rwlock.GetExclusiveLock();

	VSSDirectory directory;
	directory.num_dimensions = static_cast<uint32_t>(GetNumDimensions());
	WriteDirectory(directory);
	// Sets the buffers' allocation sizes, which GetInfo reads.
	auto buffers = allocator->InitSerializationToWAL();
	auto info = MakeStorageInfo();
	info.buffers.push_back(std::move(buffers));
	return info;
}

} // namespace duckdb
