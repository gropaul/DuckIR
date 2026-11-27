#include "index/core/rg_partitioned_index.hpp"

#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "index/core/table_range_scanner.hpp"

namespace duckdb {

RGPartitionedIndex::RGPartitionedIndex(const Identifier &name, const string &index_type,
                                       IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
                                       TableIOManager &table_io_manager,
                                       const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db)
    : BoundIndex(name, index_type, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db),
      buffer_manager(BufferManager::GetBufferManager(db)) {
}

RGPartitionedIndex::~RGPartitionedIndex() {
	UpdateReservedMemory(-static_cast<int64_t>(reserved_memory_bytes.load()));
}

void RGPartitionedIndex::UpdateReservedMemory(const int64_t delta) {
	if (delta > 0) {
		buffer_manager.ReserveMemory(static_cast<idx_t>(delta));
		reserved_memory_bytes += static_cast<idx_t>(delta);
	} else if (delta < 0) {
		buffer_manager.FreeReservedMemory(static_cast<idx_t>(-delta));
		reserved_memory_bytes -= static_cast<idx_t>(-delta);
	}
}

idx_t RGPartitionedIndex::GetRowGroupSize() const {
	return table_io_manager.GetRowGroupSize();
}

RowRange RGPartitionedIndex::GetPhysicalRowGroupRange(DataTable &table, const row_t row_id) const {
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	idx_t segment_index;
	if (!row_groups->TryGetSegmentIndex(lock, static_cast<idx_t>(row_id), segment_index)) {
		return RowRange::Empty();
	}
	auto node = row_groups->GetSegmentByIndex(lock, static_cast<int64_t>(segment_index));
	return RowRange::FromStartAndSize(static_cast<row_t>(node->GetRowStart()), node->GetCount());
}

bool RGPartitionedIndex::IsClosedRowGroup(DataTable &table, const RowRange row_group_range) const {
	return row_group_range.Size() >= GetRowGroupSize() ||
	       !GetPhysicalRowGroupRange(table, row_group_range.end).IsEmpty();
}

vector<RowRange> RGPartitionedIndex::GetPhysicalRowGroupRanges(DataTable &table) const {
	vector<RowRange> result;
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	for (auto node = row_groups->GetRootSegment(lock); node; node = row_groups->GetNextSegment(lock, *node)) {
		result.push_back(RowRange::FromStartAndSize(static_cast<row_t>(node->GetRowStart()), node->GetCount()));
	}
	return result;
}

idx_t RGPartitionedIndex::GetLargestRowGroupCount(DataTable &table) const {
	// At least the attach's row group size, which an open row group can still grow to.
	idx_t largest = GetRowGroupSize();
	for (const auto &row_group_range : GetPhysicalRowGroupRanges(table)) {
		largest = MaxValue(largest, row_group_range.Size());
	}
	return largest;
}

// Scans every committed row, whatever the calling transaction's snapshot. The mirror
// holds the committed state, and rows a query must not see are dropped when it fetches its results.
TransactionData RGPartitionedIndex::GetCommittedSnapshot(const bool all_committed) const {
	return TransactionData(MAX_TRANSACTION_ID,
	                       all_committed ? VisibilityBound::AllCommitted()
	                                     : VisibilityBound::Through(DuckTransactionManager::Get(db).GetLastCommit()));
}

vector<StorageIndex> RGPartitionedIndex::GetScanColumnIds() const {
	return {StorageIndex(GetColumnIds()[0]), StorageIndex()};
}

vector<LogicalType> RGPartitionedIndex::GetScanTypes() const {
	return {logical_types[0], LogicalType::ROW_TYPE};
}

void RGPartitionedIndex::ScanAndSinkRows(DataTable &table, const std::vector<RowRange> &ranges,
                                         IndexSinkState &sink_state, std::vector<idx_t> &rows_returned,
                                         const bool all_committed) {
	TableRangeScanner scanner(table, Allocator::Get(db), GetScanColumnIds(), GetScanTypes(),
	                          GetCommittedSnapshot(all_committed), ranges);
	while (scanner.Next()) {
		// A sink can decide to stop, for example, if it only needs a sample of rows.
		const auto sink_result = Sink(sink_state, scanner.chunk, scanner.sel, scanner.sel_count);
		if (sink_result == SinkResultType::FINISHED) {
			break;
		}
	}
	rows_returned = std::move(scanner.rows_returned);
}

idx_t RGPartitionedIndex::FetchCommittedRowIds(DataTable &table, const RowRange range, row_t *const row_ids) {
	const auto committed = GetCommittedSnapshot();
	const std::vector<RowRange> ranges {range};
	TableRangeScanner scanner(table, Allocator::Get(db), {StorageIndex()}, {LogicalType::ROW_TYPE}, committed, ranges);
	idx_t count = 0;
	while (scanner.Next()) {
		const auto scanned_row_ids = FlatVector::GetData<row_t>(scanner.chunk.data[0]);
		if (!scanner.sel.IsSet()) {
			memcpy(row_ids + count, scanned_row_ids, scanner.sel_count * sizeof(row_t));
			count += scanner.sel_count;
			continue;
		}
		for (idx_t sel_idx = 0; sel_idx < scanner.sel_count; sel_idx++) {
			row_ids[count++] = scanned_row_ids[scanner.sel.get_index(sel_idx)];
		}
	}
	return count;
}

} // namespace duckdb
