#pragma once

#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/transaction/transaction_data.hpp"
#include "utils/range.hpp"

#include <atomic>

namespace duckdb {

// What Sink does with a row the index cannot take, for VSS an embedding with NaN, infinite or too large
// values, for FTS e.g. rows that are no valid UTF-8: CREATE INDEX throws, the sync skips it
enum class UnindexableRowAction : uint8_t { SKIP, THROW };

// The rows of one DuckDB row group a sink state takes (InitializeIndexSinkState).
struct IndexSinkInput {
	IndexSinkInput(const RowRange row_group_range, const idx_t num_rows, const idx_t num_threads,
	               const UnindexableRowAction unindexable_row_action = UnindexableRowAction::SKIP)
	    : row_group_range(row_group_range), num_rows(num_rows), num_threads(num_threads),
	      unindexable_row_action(unindexable_row_action) {
	}

	// The DuckDB row group the rows are in.
	RowRange row_group_range;
	// At most row_group_range.Size().
	idx_t num_rows;
	// The threads the state's work on its row group runs on.
	idx_t num_threads;
	UnindexableRowAction unindexable_row_action;
};

// What an index keeps while it takes rows chunk by chunk (Sink), one per concurrent build.
struct IndexSinkState {
	explicit IndexSinkState(const IndexSinkInput &input) : input(input), indexable_sel(STANDARD_VECTOR_SIZE) {
	}
	virtual ~IndexSinkState() = default;

	const IndexSinkInput input;
	// The rows the sink took, and their ids: set by InitializeIndexSinkState, null for a state that only samples rows
	// (VSS: the SAQ sample).
	idx_t count = 0;
	unique_array<row_t> row_ids;
	// Selection of Rows that are not NULL and not unindexable (e.g., wrong vector dimension)
	SelectionVector indexable_sel;

	// Whether the state's work on its row group runs on several threads. Other states can still run at the same
	// time on other row groups (CREATE INDEX's concurrent builds)
	bool IsParallel() const {
		return input.num_threads > 1;
	}

	template <class TARGET>
	TARGET &Cast() {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<TARGET &>(*this);
	}
};

// An index made of one index per DuckDB row group.
class RGPartitionedIndex : public BoundIndex {
public:
	RGPartitionedIndex(const Identifier &name, const string &index_type, IndexConstraintType index_constraint_type,
	                   const vector<column_t> &column_ids, TableIOManager &table_io_manager,
	                   const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db);
	// Indexes are destroyed before DuckDB's buffer manager (DatabaseInstance::~DatabaseInstance).
	~RGPartitionedIndex() override;

	// Takes the `count` rows of `chunk` that `sel` selects, all of them if it is unset.
	virtual SinkResultType Sink(IndexSinkState &sink_state, DataChunk &chunk, const SelectionVector &sel,
	                            idx_t count) = 0;
	// A sink state for `sink_input.num_rows` rows of one row group.
	virtual unique_ptr<IndexSinkState> InitializeIndexSinkState(ClientContext &context,
	                                                            const IndexSinkInput &sink_input) = 0;
	// Builds the state's row group from the rows it took. Two states of one row group (a row group DuckDB split over
	// scan tasks) append the second's rows.
	virtual void BuildRowGroupIndex(IndexSinkState &sink_state, DataTable &table) = 0;
	// Adds the state's rows to the index's row group at row_group_idx, which already exists.
	virtual void AppendToRowGroupIndex(IndexSinkState &sink_state, idx_t row_group_idx) = 0;

	idx_t GetRowGroupSize() const;

	// The DuckDB row group that holds row_id. Empty if the row is not in the table (yet).
	RowRange GetPhysicalRowGroupRange(DataTable &table, row_t row_id) const;

	// Whether DuckDB appends no more rows to the row group. It appends only to the last one, until it is full or
	// another one follows it (a commit that merges its own row groups, or an append that requires a new one).
	bool IsClosedRowGroup(DataTable &table, RowRange row_group_range) const;

	vector<RowRange> GetPhysicalRowGroupRanges(DataTable &table) const;

	idx_t GetLargestRowGroupCount(DataTable &table) const;

protected:
	// The index's own work at the end of a sync, after the closed row groups are marked.
	virtual void FinishSync() {
	}
	// The memory the index holds outside DuckDB's buffers, which the index reserves itself (UpdateReservedMemory).
	virtual uint64_t GetSelfManagedMemorySizeInBytes() const = 0;

	// Reserves `delta` more bytes, or frees `-delta`. Throws DuckDB's out-of-memory error when nothing more can be
	// evicted.
	void UpdateReservedMemory(int64_t delta);

	// Scans the committed rows of `ranges` into `sink_state` until Sink returns FINISHED. rows_returned: per range.
	void ScanAndSinkRows(DataTable &table, const std::vector<RowRange> &ranges, IndexSinkState &sink_state,
	                     std::vector<idx_t> &rows_returned, bool all_committed = false);

	// As of the last published commit, or with `all_committed` of every commit, published or not.
	TransactionData GetCommittedSnapshot(bool all_committed = false) const;
	// The columns a TableRangeScanner of the indexed rows reads: the indexed column and the row ids.
	vector<StorageIndex> GetScanColumnIds() const;
	vector<LogicalType> GetScanTypes() const;

	// The committed rows of `range`, ascending, written to `row_ids`. Scans only the row id column, so it reads no
	// column data.
	idx_t FetchCommittedRowIds(DataTable &table, RowRange range, row_t *row_ids);

	BufferManager &buffer_manager;
	std::atomic<idx_t> reserved_memory_bytes {0};
};

} // namespace duckdb
