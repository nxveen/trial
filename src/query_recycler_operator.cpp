#include "query_recycler_operator.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// LogicalQueryRecycler
//===--------------------------------------------------------------------===//

LogicalQueryRecycler::LogicalQueryRecycler(unique_ptr<LogicalOperator> child,
                                           shared_ptr<RecyclingCache> cache_p,
                                           shared_ptr<RecyclingCacheEntry> entry_p,
                                           idx_t table_index_p)
    : LogicalExtensionOperator(), cache(std::move(cache_p)), entry(std::move(entry_p)),
      table_index(table_index_p) {
	D_ASSERT(child);
	D_ASSERT(cache);
	D_ASSERT(entry);

	children.push_back(std::move(child));
}

vector<ColumnBinding> LogicalQueryRecycler::GetColumnBindings() {
	return GenerateColumnBindings(table_index, types.size());
}

void LogicalQueryRecycler::ResolveTypes() {
	D_ASSERT(children.size() == 1);

	types = children[0]->types;
}

PhysicalOperator &LogicalQueryRecycler::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	D_ASSERT(children.size() == 1);
	D_ASSERT(cache);
	D_ASSERT(entry);

	auto &child = planner.CreatePlan(*children[0]);

	return planner.Make<PhysicalQueryRecycler>(types, estimated_cardinality, child, cache, entry);
}

//===--------------------------------------------------------------------===//
// Physical states
//===--------------------------------------------------------------------===//

class QueryRecyclerGlobalSinkState : public GlobalSinkState {
public:
	explicit QueryRecyclerGlobalSinkState(ClientContext &context, const shared_ptr<RecyclingCacheEntry> &entry_p)
	    : entry(entry_p) {
		D_ASSERT(entry);
		D_ASSERT(entry->collection);

		entry->collection->InitializeAppend(append_state);
	}

	shared_ptr<RecyclingCacheEntry> entry;
	ColumnDataAppendState append_state;
};

class QueryRecyclerLocalSinkState : public LocalSinkState {
};

class QueryRecyclerGlobalSourceState : public GlobalSourceState {
public:
	explicit QueryRecyclerGlobalSourceState(const shared_ptr<RecyclingCacheEntry> &entry_p)
	    : entry(entry_p) {
		D_ASSERT(entry);
		D_ASSERT(entry->collection);

		entry->collection->InitializeScan(scan_state);
	}

	shared_ptr<RecyclingCacheEntry> entry;
	ColumnDataScanState scan_state;
};

class QueryRecyclerLocalSourceState : public LocalSourceState {
};

//===--------------------------------------------------------------------===//
// PhysicalQueryRecycler
//===--------------------------------------------------------------------===//

PhysicalQueryRecycler::PhysicalQueryRecycler(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                             idx_t estimated_cardinality, PhysicalOperator &child,
                                             shared_ptr<RecyclingCache> cache_p,
                                             shared_ptr<RecyclingCacheEntry> entry_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      cache(std::move(cache_p)), entry(std::move(entry_p)) {
	children.push_back(child);

	D_ASSERT(cache);
	D_ASSERT(entry);
	D_ASSERT(entry->collection);
}

PhysicalQueryRecycler::~PhysicalQueryRecycler() {
	if (!finalized && cache && entry) {
		cache->MarkFailed(entry);
	}
}

unique_ptr<GlobalSinkState> PhysicalQueryRecycler::GetGlobalSinkState(ClientContext &context) const {
	D_ASSERT(cache);
	D_ASSERT(entry);

	cache->BeginMaterialization(entry);

	return make_uniq<QueryRecyclerGlobalSinkState>(context, entry);
}

unique_ptr<LocalSinkState> PhysicalQueryRecycler::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<QueryRecyclerLocalSinkState>();
}

SinkResultType PhysicalQueryRecycler::Sink(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<QueryRecyclerGlobalSinkState>();

	gstate.entry->collection->Append(gstate.append_state, chunk);

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalQueryRecycler::Combine(ExecutionContext &context,
                                                     OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalQueryRecycler::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                 OperatorSinkFinalizeInput &input) const {
	D_ASSERT(cache);
	D_ASSERT(entry);

	cache->MarkReady(entry);

	finalized = true;

	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSourceState> PhysicalQueryRecycler::GetGlobalSourceState(ClientContext &context) const {
	D_ASSERT(entry);
	D_ASSERT(entry->collection);
	D_ASSERT(cache->IsReady(entry));

	return make_uniq<QueryRecyclerGlobalSourceState>(entry);
}

unique_ptr<LocalSourceState> PhysicalQueryRecycler::GetLocalSourceState(ExecutionContext &context,
                                                                       GlobalSourceState &gstate) const {
	return make_uniq<QueryRecyclerLocalSourceState>();
}

SourceResultType PhysicalQueryRecycler::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<QueryRecyclerGlobalSourceState>();

	entry->collection->Scan(gstate.scan_state, chunk);

	if (chunk.size() == 0) {
		return SourceResultType::FINISHED;
	}

	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalQueryRecycler::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;

	if (entry && entry->collection) {
		result["Cache Rows"] = StringUtil::Format("%llu", entry->collection->Count());
		result["Cache Bytes"] = StringUtil::Format("%llu", entry->collection->SizeInBytes());
	}

	SetEstimatedCardinality(result, estimated_cardinality);

	return result;
}

} // namespace duckdb