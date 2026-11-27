#pragma once

#include "duckdb/common/enums/scan_options.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "utils/range.hpp"

namespace duckdb {

// Scans `column_ids` (row id last) of `row_id_ranges` (sorted, disjoint) as of its own snapshot, not the calling
// query's.
class TableRangeScanner {
public:
	TableRangeScanner(DataTable &table, Allocator &allocator, vector<StorageIndex> column_ids,
	                  const vector<LogicalType> &types, const TransactionData &transaction,
	                  const std::vector<RowRange> &row_id_ranges);

	// Per range, NULL values included.
	std::vector<idx_t> rows_returned;
	// The rows Next read last. `sel` selects the `sel_count` inside the ranges, unset when all are.
	DataChunk chunk;
	SelectionVector sel;
	idx_t sel_count = 0;

public:
	// Reads the next chunk with rows inside the ranges, row ids flat; false once all ranges are read.
	bool Next();

private:
	DataTable &table;
	const ScanOptions scan_options;
	const vector<StorageIndex> column_ids;
	// The current TableScanState reads row_id_ranges[current_ranges.start, current_ranges.end); start moves to the
	// range the last row matched.
	const std::vector<RowRange> &row_id_ranges;
	Range<idx_t> current_ranges = Range<idx_t>::Empty();
	// One per range group: DuckDB starts a TableScanState only once.
	unique_ptr<TableScanState> table_scan_state;
};

} // namespace duckdb
