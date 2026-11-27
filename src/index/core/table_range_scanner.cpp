#include "index/core/table_range_scanner.hpp"

#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/transaction/transaction_data.hpp"

namespace duckdb {

TableRangeScanner::TableRangeScanner(DataTable &table, Allocator &allocator, vector<StorageIndex> column_ids_p,
                                     const vector<LogicalType> &types, const TransactionData &transaction,
                                     const std::vector<RowRange> &row_id_ranges)
    : rows_returned(row_id_ranges.size(), 0), table(table), scan_options(transaction),
      column_ids(std::move(column_ids_p)), row_id_ranges(row_id_ranges) {
	chunk.Initialize(allocator, types);
}

bool TableRangeScanner::Next() {
	const auto &ranges = row_id_ranges;
	while (true) {
		if (!table_scan_state) {
			while (current_ranges.start < ranges.size() && ranges[current_ranges.start].IsEmpty()) {
				current_ranges.start++;
			}
			if (current_ranges.start >= ranges.size()) {
				return false;
			}
			// Ranges less than a vector apart share one scan: the scan of each would read that vector anyway.
			current_ranges.end = current_ranges.start + 1;
			while (current_ranges.end < ranges.size() &&
			       ranges[current_ranges.end].start - ranges[current_ranges.end - 1].end <
			           static_cast<row_t>(STANDARD_VECTOR_SIZE)) {
				current_ranges.end++;
			}
			table_scan_state = make_uniq<TableScanState>();
			table_scan_state->Initialize(column_ids);
			table.GetRowGroupCollection()->InitializeScanWithOffset(
			    QueryContext(), table_scan_state->table_state, column_ids,
			    static_cast<idx_t>(ranges[current_ranges.start].start),
			    static_cast<idx_t>(ranges[current_ranges.end - 1].end));
		}
		chunk.Reset();
		// Deleted rows are gone, NULL values are NULL.
		table_scan_state->table_state.Scan(scan_options, chunk);
		if (chunk.size() == 0) {
			table_scan_state.reset();
			current_ranges.start = current_ranges.end;
			continue;
		}
		// The row ids are the last column. The index flattens the other columns if it needs to.
		auto &row_id_column = chunk.data[chunk.ColumnCount() - 1];
		row_id_column.Flatten();
		const auto scanned_row_ids = FlatVector::GetData<row_t>(row_id_column);
		const idx_t chunk_size = chunk.size();
		// Row ids ascend: a chunk whose first and last row are in the current range needs no selection.
		const auto &range = ranges[current_ranges.start];
		if (range.Contains(scanned_row_ids[0]) && range.Contains(scanned_row_ids[chunk_size - 1])) {
			rows_returned[current_ranges.start] += chunk_size;
			sel.Initialize(nullptr, 0);
			sel_count = chunk_size;
			return true;
		}
		if (!sel.IsSet()) {
			sel.Initialize(STANDARD_VECTOR_SIZE);
		}
		sel_count = 0;
		for (idx_t i = 0; i < chunk_size; i++) {
			// The scan starts at the vector that holds the first range's start, and reads the gaps between ranges.
			while (current_ranges.start + 1 < current_ranges.end &&
			       ranges[current_ranges.start].end <= scanned_row_ids[i]) {
				current_ranges.start++;
			}
			if (!ranges[current_ranges.start].Contains(scanned_row_ids[i])) {
				continue;
			}
			rows_returned[current_ranges.start]++;
			sel.set_index(sel_count++, i);
		}
		if (sel_count > 0) {
			return true;
		}
	}
}

} // namespace duckdb
