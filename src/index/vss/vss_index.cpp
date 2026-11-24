#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction_data.hpp"
#include "pdx/common.hpp"
#include "pdx/utils.hpp"
#include "index/vss/vss_index.hpp"

#include <random>
#include "index/vss/vss_parallel_executor.hpp"

#include "index/vss/vss_module.hpp"

#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

VSSIndex::VSSIndex(const Identifier &name, IndexConstraintType index_constraint_type,
                   const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                   const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                   const case_insensitive_map_t<Value> &index_creation_options,
                   const IndexStorageInfo &persistence_info)
    : BoundIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db),
      memory_allocator(BufferAllocator::Get(db.GetDatabase())), buffer_manager(BufferManager::GetBufferManager(db)) {
	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("DuckIR VSS indexes do not support unique or primary key constraints");
	}

	// We only support one ARRAY column
	D_ASSERT(logical_types.size() == 1);
	const auto &embedding_type = logical_types[0];
	D_ASSERT(embedding_type.id() == LogicalTypeId::ARRAY);

	const auto num_dimensions = ArrayType::GetSize(embedding_type);

	// DuckDB v1.5.6 does not persist the WITH options in its catalog, so a
	// persisted index reads them from its storage info (MakeStorageInfo).
	const auto &options = persistence_info.IsValid() ? persistence_info.options : index_creation_options;

	// Try to get the vector metric from the options, this parameter should be verified during binding.
	auto dist_metric = VSSWrapper::DEFAULT_DISTANCE_METRIC;
	const auto dist_metric_opt = options.find("metric");
	if (dist_metric_opt != options.end()) {
		const auto dist_metric_val = VSSIndex::DISTANCE_METRIC_MAP.find(dist_metric_opt->second.GetValue<string>());
		if (dist_metric_val != VSSIndex::DISTANCE_METRIC_MAP.end()) {
			dist_metric = dist_metric_val->second;
		}
	}

	auto quantization = VSSWrapper::DEFAULT_QUANTIZATION;
	const auto quantization_opt = options.find("quantization");
	if (quantization_opt != options.end()) {
		const auto quantization_val = VSSIndex::QUANTIZATION_MAP.find(quantization_opt->second.GetValue<string>());
		if (quantization_val != VSSIndex::QUANTIZATION_MAP.end()) {
			quantization = quantization_val->second;
		}
	}

	auto n_probe = VSSWrapper::DEFAULT_N_PROBE;
	const auto n_probe_opt = options.find("n_probe");
	if (n_probe_opt != options.end()) {
		n_probe = n_probe_opt->second.GetValue<int32_t>();
	}

	// A persisted index always has its seed: only a new one draws (std::random_device may throw).
	int32_t seed;
	const auto seed_opt = options.find("seed");
	if (seed_opt != options.end()) {
		seed = seed_opt->second.GetValue<int32_t>();
	} else {
		std::random_device random_device;
		seed = std::uniform_int_distribution<int32_t>(VSSWrapper::MIN_SEED, VSSWrapper::MAX_SEED)(random_device);
	}

	auto build_cluster_sketches = VSSWrapper::DEFAULT_CLUSTER_SKETCHES;
	const auto cluster_sketches_opt = options.find("cluster_sketches");
	if (cluster_sketches_opt != options.end()) {
		build_cluster_sketches = cluster_sketches_opt->second.GetValue<bool>();
	}

	Value max_cluster_size_setting;
	db.GetDatabase().TryGetCurrentSetting("duckir_vss_experimental_max_cluster_size", max_cluster_size_setting);
	const auto max_cluster_size = max_cluster_size_setting.IsNull() ? VSSWrapper::DEFAULT_MAX_CLUSTER_SIZE
	                                                                : max_cluster_size_setting.GetValue<uint32_t>();
	Value cluster_sketch_check_setting;
	db.GetDatabase().TryGetCurrentSetting("duckir_vss_cluster_sketch_check", cluster_sketch_check_setting);
	const auto cluster_sketch_check = cluster_sketch_check_setting.IsNull()
	                                      ? PDX::ClusterSketchCheck::WHEN_CLUSTER_NOT_CACHED
	                                      : CLUSTER_SKETCH_CHECK_MAP.at(cluster_sketch_check_setting.ToString());

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	allocator = make_uniq<FixedSizeAllocator>(VSSBlockChain::GetSegmentSize(block_manager), block_manager);
	// A persisted index keeps its rotation. Its row groups are loaded once the wrapper exists. If its storage cannot be
	// opened (bind_error), the wrapper gets the seed's rotation, only for planning until the error is raised.
	VSSDirectory directory;
	unique_ptr<float[]> rotation_matrix;
	std::optional<PDX::SAQModel> saq_model;
	if (persistence_info.IsValid()) {
		persisted_storage_info = CopyStorageInfo(persistence_info);
		try {
			storage_reader = OpenStorage(persistence_info, static_cast<uint32_t>(num_dimensions), directory,
			                             rotation_matrix, saq_model);
		} catch (std::exception &ex) {
			bind_error = ErrorData(ex);
			rotation_matrix.reset();
			saq_model.reset();
		}
	}

	const idx_t row_group_size = table_io_manager.GetRowGroupSize();
	if (quantization == PDX::Quantization::F32) {
		D_ASSERT(ArrayType::GetChildType(embedding_type).id() == LogicalTypeId::FLOAT);

		vss_wrapper = make_uniq<VSSWrapperF32>(dist_metric, num_dimensions, n_probe, seed, max_cluster_size,
		                                       build_cluster_sketches, cluster_sketch_check, row_group_size,
		                                       std::move(rotation_matrix), std::nullopt, memory_allocator);
	} else if (quantization == PDX::Quantization::U8) {
		vss_wrapper = make_uniq<VSSWrapperU8>(dist_metric, num_dimensions, n_probe, seed, max_cluster_size,
		                                      build_cluster_sketches, cluster_sketch_check, row_group_size,
		                                      std::move(rotation_matrix), std::nullopt, memory_allocator);
	} else if (quantization == PDX::Quantization::SAQ) {
		vss_wrapper = make_uniq<VSSWrapperSAQ>(dist_metric, num_dimensions, n_probe, seed, max_cluster_size,
		                                       build_cluster_sketches, cluster_sketch_check, row_group_size,
		                                       std::move(rotation_matrix), std::move(saq_model), memory_allocator);
	} else {
		throw InternalException("Unsupported quantization: %s", quantization);
	}
	// The rotation matrix and the pruner, before any row group. A persisted index creates and reserves them as it
	// loads, below.
	if (!persistence_info.IsValid()) {
		vss_wrapper->CreatePruner();
		UpdateReservedMemory(static_cast<int64_t>(vss_wrapper->GetSelfManagedMemorySizeInBytes()));
	}
	Value cluster_paging_setting;
	db.GetDatabase().TryGetCurrentSetting("duckir_vss_cluster_paging", cluster_paging_setting);
	cluster_paging = cluster_paging_setting.IsNull() || cluster_paging_setting.GetValue<bool>();
	Value paging_counters_setting;
	db.GetDatabase().TryGetCurrentSetting("duckir_vss_paging_counters", paging_counters_setting);
	record_paging_counters = !paging_counters_setting.IsNull() && paging_counters_setting.GetValue<bool>();
	Value cache_tiers_setting;
	db.GetDatabase().TryGetCurrentSetting("duckir_vss_cache_tiers", cache_tiers_setting);
	cache_tiers = cache_tiers_setting.IsNull() || cache_tiers_setting.GetValue<bool>();
	// An index created in this session pages its row groups from their temporary chains.
	if (cluster_paging && !storage_reader) {
		storage_reader = make_uniq<VSSBlockChainReader>(table_io_manager.GetIndexBlockManager(), allocator->GetInfo());
	}
	if (storage_reader) {
		storage_reader->counters.enabled = record_paging_counters;
		storage_reader->cache_tiers = cache_tiers;
	}
	if (persistence_info.IsValid() && !bind_error.HasError()) {
		try {
			vss_wrapper->CreatePruner();
			UpdateReservedMemory(static_cast<int64_t>(vss_wrapper->GetSelfManagedMemorySizeInBytes()));
			LoadRowGroups(*storage_reader, directory, cluster_paging);
			persisted_row_end = directory.row_groups.empty() ? 0 : directory.row_groups.back().row_end;
			// If first_reused_row_id is among the persisted options, a row id was reused. The reconcile must rebuild
			// the affected row groups from the table. See SerializeToDisk for more details.
			const auto first_reused = persistence_info.options.find("first_reused_row_id");
			if (first_reused != persistence_info.options.end()) {
				first_reused_row_id = first_reused->second.GetValue<row_t>();
				RemoveRowGroupsOverlapping(first_reused_row_id, NumericLimits<row_t>::Maximum());
			}
			needs_reconciliation = true;
		} catch (std::exception &ex) {
			bind_error = ErrorData(ex);
		}
	}

	function_matcher = MakeFunctionMatcher(*vss_wrapper.get());
}

// Indexes are destroyed before DuckDB's buffer manager (DatabaseInstance::~DatabaseInstance).
VSSIndex::~VSSIndex() {
	UpdateReservedMemory(-static_cast<int64_t>(reserved_memory_bytes.load()));
	buffer_manager.FreeReservedMemory(table_scan_reserved_bytes.load());
}

void VSSIndex::ReserveTableScanMetadata(const idx_t num_rows) {
	const idx_t block_size = table_io_manager.GetBlockManagerForRowData().GetBlockSize();
	const idx_t column_bytes = num_rows * vss_wrapper->GetNumDimensions() * sizeof(float);
	const idx_t bytes = (column_bytes + block_size - 1) / block_size * DUCKDB_TABLE_SCAN_METADATA_BYTES_PER_BLOCK;
	buffer_manager.ReserveMemory(bytes);
	table_scan_reserved_bytes += bytes;
}

void VSSIndex::UpdateReservedMemory(const int64_t delta) {
	if (delta > 0) {
		buffer_manager.ReserveMemory(static_cast<idx_t>(delta));
		reserved_memory_bytes += static_cast<idx_t>(delta);
	} else if (delta < 0) {
		buffer_manager.FreeReservedMemory(static_cast<idx_t>(-delta));
		reserved_memory_bytes -= static_cast<idx_t>(-delta);
	}
}

/******************************************************************
 * Index creation and search methods specific to the parallel implementation
 ******************************************************************/

void VSSIndex::FitSAQModelForCreate(ClientContext &context, DataTable &table) {
	// No row group can reach MIN_EMBEDDINGS_FOR_CLUSTERING rows: they are all Flat, and the first sync that clusters
	// one fits.
	if (!vss_wrapper->NeedsSAQFit() || table.GetTotalRows() < VSSWrapperSAQ::MIN_EMBEDDINGS_FOR_CLUSTERING) {
		return;
	}
	// The first non-NULL rows of every row group, SAQ_FIT_SAMPLE_SIZE in all.
	const auto physical_row_groups = GetPhysicalRowGroups(table);
	const idx_t rows_per_row_group =
	    (PDX::SAQ_FIT_SAMPLE_SIZE + physical_row_groups.size() - 1) / physical_row_groups.size();
	idx_t max_sampled = 0;
	for (const auto &row_group : physical_row_groups) {
		max_sampled += MinValue<idx_t>(row_group.count, rows_per_row_group);
	}
	const auto num_dimensions = GetNumDimensions();
	auto sample = skmeans::AllocateBuffer<float>(memory_allocator, max_sampled * num_dimensions);
	DuckDBParallelExecutor executor(context,
	                                static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads()));
	idx_t num_sampled = 0;
	std::vector<idx_t> rows_returned;
	for (const auto &row_group : physical_row_groups) {
		const row_t end = row_group.row_start + static_cast<row_t>(row_group.count);
		// The identity rotation of an index not fitted yet leaves the rows as they are.
		num_sampled += FetchRows(table, {{row_group.row_start, end}}, sample.get() + num_sampled * num_dimensions,
		                         rows_returned, /*row_ids=*/nullptr, /*rotate=*/false, &executor,
		                         MinValue<idx_t>(row_group.count, rows_per_row_group));
	}
	if (num_sampled == 0) {
		return;
	}
	DispatchWrapper([&](auto &wrapper) { wrapper.FitSAQIfNeeded(sample.get(), num_sampled, &executor); });
}

// Called by CREATE INDEX's threads in parallel: each reserves the self-managed memory its row group grew by. PDX's
// buffers are charged as it allocates them.
void VSSIndex::SetUpIndexForRowGroup(const row_t *const row_ids, float *const vectors, const idx_t num_vectors,
                                     const row_t row_start, const idx_t count, const bool is_closed,
                                     PDX::ParallelExecutor *const executor) {
	// Without paging, the row groups stay in memory.
	const auto reader = cluster_paging ? storage_reader.get() : nullptr;
	const int64_t self_managed_growth_bytes = DispatchWrapper([&](auto &wrapper) {
		return wrapper.SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_start, count, is_closed, executor,
		                                     reader, buffer_manager);
	});
	UpdateReservedMemory(self_managed_growth_bytes);
}

uint64_t VSSIndex::EstimateBuildPeakBytes(const idx_t num_embeddings) const {
	return DispatchWrapper([&](auto &wrapper) { return wrapper.EstimateBuildPeakBytes(num_embeddings); });
}

unique_ptr<PDX::ClusterSketchQueryTable>
VSSIndex::CreateSketchQueryTable(const float *const preprocessed_query_embedding) const {
	const auto check = vss_wrapper->GetClusterSketchCheck();
	// Without paging every cluster is in memory, so WHEN_CLUSTER_NOT_CACHED never checks
	if (!vss_wrapper->BuildsClusterSketches() || check == PDX::ClusterSketchCheck::NEVER ||
	    (check == PDX::ClusterSketchCheck::WHEN_CLUSTER_NOT_CACHED && !cluster_paging)) {
		return nullptr;
	}
	return DispatchWrapper([&](auto &wrapper) { return wrapper.CreateSketchQueryTable(preprocessed_query_embedding); });
}

unique_ptr<PDX::IIterativeSearch>
VSSIndex::BeginSearchForRowGroup(const idx_t row_group_idx, const float *const preprocessed_query, const idx_t limit,
                                 PDX::TopKHeap &top_k_heap, const std::vector<size_t> *const passing_row_ids,
                                 const std::vector<uint32_t> *const clusters_access_order,
                                 const PDX::CentroidRankingQuery *const ranking_query,
                                 const PDX::ClusterSketchQueryTable *const sketch_query_table) {
	return DispatchWrapper([&](auto &wrapper) {
		return wrapper.BeginSearchForRowGroup(row_group_idx, preprocessed_query, limit, top_k_heap, passing_row_ids,
		                                      clusters_access_order, ranking_query, sketch_query_table);
	});
}

std::vector<uint32_t>
VSSIndex::GetClustersAccessOrderForRowGroup(const idx_t row_group_idx, const float *const preprocessed_query,
                                            const PDX::CentroidRankingQuery *const ranking_query) {
	return DispatchWrapper([&](auto &wrapper) {
		return wrapper.GetClustersAccessOrderForRowGroup(row_group_idx, preprocessed_query, ranking_query);
	});
}

/******************************************************************
 * Index maintenance
 ******************************************************************/

// Called by DROP INDEX.
// Dropping the allocator's buffers marks their blocks modified; DuckDB frees them at the next checkpoint.
void VSSIndex::ResetStorage(IndexLock &lock) {
	// DuckDB destroys the index next (it rebuilds only ART indexes in place): first, the queries searching it finish.
	{
		std::unique_lock<std::mutex> pins(search_pins_lock);
		is_retired = true;
		search_pins_released.wait(pins, [&] { return search_pins == 0; });
	}
	auto _lock = rwlock.GetExclusiveLock();

	allocator->Reset();
	directory_chain = VSSBlockChain();
	rotation_chain = VSSBlockChain();
	saq_model_chain = VSSBlockChain();
	ResetPersistedChains();
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

idx_t VSSIndex::GetInMemorySize(IndexLock &state) const {
	auto _lock = rwlock.GetSharedLock();

	// The allocator only holds the buffers that were not written to disk yet.
	return vss_wrapper->GetInMemorySizeInBytes() + allocator->GetInMemorySize();
}

unique_ptr<VSSIndexStats> VSSIndex::GetStats(const ClientContext &context) const {
	// A concurrent sync changes the mirrors under the search lock. A query that searches this index holds it already:
	// taking it again would wait for any writer queued behind the query.
	unique_ptr<StorageLockKey> search_lock;
	if (!HasSearchGuard(context, *this)) {
		search_lock = rwlock.GetSharedLock();
	}
	auto result = make_uniq<VSSIndexStats>();

	// Make sure to sync any changes made here to `vss_index_info_function.cpp`.
	result->metric = GetDistanceMetric();
	result->quantization = GetQuantization();
	result->num_dimensions = static_cast<int64_t>(GetNumDimensions());
	result->n_probe = static_cast<int64_t>(vss_wrapper->GetNProbe());
	result->seed = static_cast<int64_t>(vss_wrapper->GetSeed());
	result->is_normalized = IsNormalized();
	result->cluster_sketches = vss_wrapper->BuildsClusterSketches();
	result->approximate_lower_bound_memory_usage_bytes = static_cast<int64_t>(vss_wrapper->GetInMemorySizeInBytes());
	result->table_scan_reserved_bytes = static_cast<int64_t>(table_scan_reserved_bytes.load());
	if (storage_reader) {
		const auto &counters = storage_reader->counters;
		result->cluster_acquires = static_cast<int64_t>(counters.cluster_acquires.load());
		result->cluster_cache_misses = static_cast<int64_t>(counters.cluster_cache_misses.load());
		result->cluster_bytes_fetched = static_cast<int64_t>(counters.cluster_bytes_fetched.load());
		result->cluster_sketch_acquires = static_cast<int64_t>(counters.cluster_sketch_acquires.load());
		result->cluster_sketch_cache_misses = static_cast<int64_t>(counters.cluster_sketch_cache_misses.load());
		result->cluster_sketch_bytes_fetched = static_cast<int64_t>(counters.cluster_sketch_bytes_fetched.load());
		result->blocks_read = static_cast<int64_t>(counters.blocks_read.load());
	}

	return result;
}

VSSIndexPlanInfo VSSIndex::GetPlanInfo(const idx_t oid) const {
	auto _lock = rwlock.GetSharedLock();
	VSSIndexPlanInfo info;
	info.oid = oid;
	info.name = GetIndexName().GetIdentifierName();
	info.num_dimensions = GetNumDimensions();
	info.total_num_clusters = GetTotalNumClusters();
	info.num_row_groups = GetNumRowGroups();
	// The allocator only holds the buffers that were not written to disk yet.
	info.in_memory_size = vss_wrapper->GetInMemorySizeInBytes() + allocator->GetInMemorySize();
	return info;
}

/******************************************************************
 * Misc.
 ******************************************************************/

bool VSSIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                        vector<reference<Expression>> &bindings) const {
	return function_matcher->Match(*expr, bindings);
}

static void TryBindIndexExpressionInternal(Expression &expr, TableIndex table_idx,
                                           const vector<column_t> &index_columns,
                                           const vector<ColumnIndex> &table_columns, bool &success, bool &found) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();

		// Rewrite the column reference to fit in the current set of bound column ids
		ref.BindingMutable().table_index = table_idx;

		const auto referenced_column = index_columns[ref.Binding().column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == referenced_column) {
				ref.BindingMutable().column_index = ProjectionIndex(i);
				return;
			}
		}
		success = false;
	}

	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		TryBindIndexExpressionInternal(child, table_idx, index_columns, table_columns, success, found);
	});
}

bool VSSIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr_ptr = unbound_expressions.back()->Copy();

	auto &expr = *expr_ptr;
	auto &index_columns = GetColumnIds();
	auto &table_columns = get.GetColumnIds();

	auto success = true;
	auto found = false;

	TryBindIndexExpressionInternal(expr, get.table_index, index_columns, table_columns, success, found);

	if (success && found) {
		result = std::move(expr_ptr);
		return true;
	}
	return false;
}

string VSSIndex::GetQuantization() const {
	switch (vss_wrapper->GetQuantization()) {
	case PDX::Quantization::F32:
		return "f32";
	case PDX::Quantization::U8:
		return "u8";
	case PDX::Quantization::SAQ:
		return "u4";
	default:
		throw InternalException("Unknown quantization");
	}
}

const case_insensitive_map_t<PDX::Quantization> VSSIndex::QUANTIZATION_MAP = {
    {"f32", PDX::Quantization::F32},
    {"u8", PDX::Quantization::U8},
    {"u4", PDX::Quantization::SAQ},
};

const case_insensitive_map_t<PDX::ClusterSketchCheck> VSSIndex::CLUSTER_SKETCH_CHECK_MAP = {
    {"never", PDX::ClusterSketchCheck::NEVER},
    {"when_cluster_not_cached", PDX::ClusterSketchCheck::WHEN_CLUSTER_NOT_CACHED},
};

string VSSIndex::GetDistanceMetric() const {
	switch (vss_wrapper->GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		return "l2sq";
	case PDX::DistanceMetric::COSINE:
		return "cosine";
	// case PDX::DistanceMetric::IP:
	// 	return "ip";
	default:
		throw InternalException("Unknown distance metric");
	}
}

const case_insensitive_map_t<PDX::DistanceMetric> VSSIndex::DISTANCE_METRIC_MAP = {
    {"l2sq", PDX::DistanceMetric::L2SQ}, {"cosine", PDX::DistanceMetric::COSINE},
    // {"ip", PDX::DistanceMetric::IP},
};

unique_ptr<ExpressionMatcher> VSSIndex::MakeFunctionMatcher(const VSSWrapper &vss_wrapper) {
	identifier_set_t distance_functions;

	switch (vss_wrapper.GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		distance_functions = {"array_distance", "<->"};
		break;
	case PDX::DistanceMetric::COSINE:
		distance_functions = {"array_cosine_distance", "<=>"};
		break;
	// case PDX::DistanceMetric::IP:
	// 	distance_functions = {"array_negative_inner_product", "<#>"};
	//  break;
	default:
		throw NotImplementedException("Unknown distance metric");
	}

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto array_type = LogicalType::ARRAY(LogicalType::FLOAT, vss_wrapper.GetNumDimensions());

	auto lhs_matcher = make_uniq<ExpressionMatcher>();
	lhs_matcher->type = make_uniq<SetTypesMatcher>(vector<LogicalType>({array_type, LogicalType::BLOB}));
	matcher->matchers.push_back(std::move(lhs_matcher));

	auto rhs_matcher = make_uniq<ExpressionMatcher>();
	rhs_matcher->type = make_uniq<SetTypesMatcher>(vector<LogicalType>({array_type, LogicalType::BLOB}));
	matcher->matchers.push_back(std::move(rhs_matcher));

	return std::move(matcher);
}

string ColumnNamesToString(const TableCatalogEntry &table, const vector<ColumnIndex> &column_ids) {
	string result;
	for (auto &column_id : column_ids) {
		if (!result.empty()) {
			result += "\n";
		}
		if (column_id.IsRowIdColumn()) {
			result += "rowid";
		} else {
			result += table.GetColumn(LogicalIndex(column_id.GetPrimaryIndex())).Name();
		}
	}
	return result;
}

void VSSModule::RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;

	index_type.name = VSSIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<VSSIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                               input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = VSSIndex::CreatePlan;

	db.config.AddExtensionOption("duckir_vss_n_probe",
	                             "override the n_probe parameter when scanning DuckIR VSS indexes (default: " +
	                                 to_string(VSSWrapper::DEFAULT_N_PROBE) + ", must be >= 0)",
	                             LogicalType::INTEGER, Value());
	db.config.AddExtensionOption("duckir_vss_rank_clusters_across_row_groups",
	                             "LATERAL index joins probe clusters nearest to the "
	                             "query across all row groups (default: true); false probes the n_probe nearest of "
	                             "each row group",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true));
	db.config.AddExtensionOption(
	    "duckir_vss_bruteforce_rows_per_cluster",
	    "filtered LATERAL index joins search the passing rows by brute force, without the index, when they number "
	    "at most this many per cluster of the row groups they are in, and at most duckir_vss_bruteforce_max_rows "
	    "(default: 2, 0 disables it)",
	    LogicalType::DOUBLE, Value::DOUBLE(2.0));
	db.config.AddExtensionOption("duckir_vss_bruteforce_max_rows",
	                             "the most passing rows a filtered LATERAL index join searches by brute force "
	                             "(default: 100000, 0 disables it)",
	                             LogicalType::UBIGINT, Value::UBIGINT(100000));
	db.config.AddExtensionOption(
	    "duckir_vss_experimental_on_the_fly_indexing",
	    "filtered LATERAL index joins whose passing rows are too many for a brute-force search build indexes over only "
	    "the passing rows and search them instead of the row groups: 'always', 'never' or 'auto' (when there are at "
	    "most duckir_vss_on_the_fly_indexing_threshold passing rows per query, queries as estimated; default: never)",
	    LogicalType::VARCHAR, Value("never"), [](ClientContext &, SetScope, Value &parameter) {
		    const auto mode = StringUtil::Lower(parameter.ToString());
		    if (mode != "always" && mode != "never" && mode != "auto") {
			    throw InvalidInputException(
			        "duckir_vss_experimental_on_the_fly_indexing must be 'always', 'never' or 'auto', not '%s'",
			        parameter.ToString());
		    }
		    parameter = Value(mode);
	    });
	db.config.AddExtensionOption(
	    "duckir_vss_on_the_fly_indexing_threshold",
	    "with duckir_vss_experimental_on_the_fly_indexing = 'auto', the indexes over the passing rows are built when "
	    "there are at most this many passing rows per query, queries as estimated (default: 20)",
	    LogicalType::UBIGINT, Value::UBIGINT(20));
	db.config.AddExtensionOption("duckir_vss_on_the_fly_indexing_max_row_groups",
	                             "the most consecutive row groups whose passing rows go into one index built on the "
	                             "fly (default: 0, no limit: the memory left decides)",
	                             LogicalType::UBIGINT, Value::UBIGINT(0));
	db.config.AddExtensionOption("duckir_vss_cluster_paging",
	                             "indexes loaded from a database file read their clusters from it as searches need "
	                             "them, so they can be larger than memory_limit; false loads them fully when they "
	                             "load (default: true)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true), nullptr, SetScope::GLOBAL);
	db.config.AddExtensionOption("duckir_vss_paging_counters",
	                             "indexes loaded with it on count their cluster cache's acquires, misses and bytes "
	                             "read, of the clusters and of their sketches, and the blocks read, in "
	                             "duckir_vss_index_info (default: false)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(false), nullptr, SetScope::GLOBAL);
	db.config.AddExtensionOption("duckir_vss_cache_tiers",
	                             "in indexes loaded with it on, the clusters searches read most often are the last "
	                             "DuckDB evicts from memory (default: true)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true), nullptr, SetScope::GLOBAL);
	db.config.AddExtensionOption(
	    "duckir_vss_cluster_sketch_check",
	    "when searches of indexes loaded with it check a cluster's sketch before probing it: 'never' or "
	    "'when_cluster_not_cached', when the cluster must be read from disk (default: when_cluster_not_cached)",
	    LogicalType::VARCHAR, Value("when_cluster_not_cached"),
	    [](ClientContext &, SetScope, Value &parameter) {
		    const auto policy = StringUtil::Lower(parameter.ToString());
		    if (VSSIndex::CLUSTER_SKETCH_CHECK_MAP.find(policy) == VSSIndex::CLUSTER_SKETCH_CHECK_MAP.end()) {
			    throw InvalidInputException("duckir_vss_cluster_sketch_check must be 'never' or "
			                                "'when_cluster_not_cached', not '%s'",
			                                parameter.ToString());
		    }
		    parameter = Value(policy);
	    },
	    SetScope::GLOBAL);
	db.config.AddExtensionOption(
	    "duckir_vss_experimental_max_cluster_size",
	    "incremental maintenance splits clusters when they reach this size, between 32 and 2048 (default: 384)",
	    LogicalType::UBIGINT, Value::UBIGINT(VSSWrapper::DEFAULT_MAX_CLUSTER_SIZE),
	    [](ClientContext &, SetScope, Value &parameter) {
		    const auto max_cluster_size = parameter.GetValue<uint64_t>();
		    if (max_cluster_size < PDX::MAX_CLUSTER_SIZE_LOWER_BOUND ||
		        max_cluster_size > VSSWrapper::MAX_CLUSTER_SIZE_UPPER_BOUND) {
			    throw InvalidInputException(
			        "duckir_vss_experimental_max_cluster_size must be between %d and %d, not %d",
			        PDX::MAX_CLUSTER_SIZE_LOWER_BOUND, VSSWrapper::MAX_CLUSTER_SIZE_UPPER_BOUND, max_cluster_size);
		    }
	    },
	    SetScope::GLOBAL);

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}

} // namespace duckdb
