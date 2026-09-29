#include "query_recycler_operator.hpp"

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/parallel/pipeline.hpp"

namespace duckdb {

// -------------------------------------------------------------------------
// Physical states
// -------------------------------------------------------------------------

class RecyclerGlobalSinkState : public GlobalSinkState {
public:
	explicit RecyclerGlobalSinkState(
	    const shared_ptr<RecyclingCacheEntry> &entry_p)
	    : entry(entry_p) {

		D_ASSERT(entry);
		D_ASSERT(entry->collection);

		entry->collection->InitializeAppend(append_state);
	}

	shared_ptr<RecyclingCacheEntry> entry;
	ColumnDataAppendState append_state;
};

class RecyclerLocalSinkState : public LocalSinkState {
};

class RecyclerGlobalSourceState : public GlobalSourceState {
public:
	explicit RecyclerGlobalSourceState(
	    const shared_ptr<RecyclingCacheEntry> &entry_p)
	    : entry(entry_p) {

		D_ASSERT(entry);
		D_ASSERT(entry->collection);

		entry->collection->InitializeScan(scan_state);
	}

	shared_ptr<RecyclingCacheEntry> entry;
	ColumnDataScanState scan_state;
};

class RecyclerLocalSourceState : public LocalSourceState {
};

// -------------------------------------------------------------------------
// LogicalRecyclerMaterialize
// -------------------------------------------------------------------------

LogicalRecyclerMaterialize::LogicalRecyclerMaterialize(
    unique_ptr<LogicalOperator> child,
    shared_ptr<RecyclingCache> cache_p,
    shared_ptr<RecyclingCacheEntry> entry_p)
    : LogicalExtensionOperator(),
      cache(std::move(cache_p)),
      entry(std::move(entry_p)) {

	D_ASSERT(child);
	D_ASSERT(cache);
	D_ASSERT(entry);

	bindings = child->GetColumnBindings();

	children.push_back(std::move(child));
}

void LogicalRecyclerMaterialize::ResolveTypes() {
	D_ASSERT(children.size() == 1);

	types = children[0]->types;
}

void LogicalRecyclerMaterialize::ResolveColumnBindings(
    ColumnBindingResolver &res,
    vector<ColumnBinding> &result) {

	D_ASSERT(children.size() == 1);

	// Resolve the child first so that references inside the original
	// subtree remain valid.
	res.VisitOperator(*children[0]);

	result = bindings;
}

vector<ColumnBinding> LogicalRecyclerMaterialize::GetColumnBindings() {
	return bindings;
}

PhysicalOperator &LogicalRecyclerMaterialize::CreatePlan(
    ClientContext &context,
    PhysicalPlanGenerator &planner) {

	D_ASSERT(children.size() == 1);

	auto &child = planner.CreatePlan(*children[0]);

	return planner.Make<PhysicalRecyclerMaterialize>(
	    types,
	    estimated_cardinality,
	    child,
	    cache,
	    entry);
}

// -------------------------------------------------------------------------
// LogicalRecyclerScan
// -------------------------------------------------------------------------

LogicalRecyclerScan::LogicalRecyclerScan(
    shared_ptr<RecyclingCacheEntry> entry_p,
    vector<ColumnBinding> current_bindings)
    : LogicalExtensionOperator(),
      bindings(std::move(current_bindings)),
      entry(std::move(entry_p)) {

	D_ASSERT(entry);
	D_ASSERT(entry->collection);

	// The cached collection provides the physical data and its types.
	// ColumnBindings, however, belong to the current query.
	types = entry->types;

	D_ASSERT(types.size() == bindings.size());
}

void LogicalRecyclerScan::ResolveTypes() {
	D_ASSERT(entry);

	types = entry->types;
}

void LogicalRecyclerScan::ResolveColumnBindings(
    ColumnBindingResolver &res,
    vector<ColumnBinding> &result) {

	(void)res;

	// Use bindings belonging to the current query, not the query that
	// originally populated the cache.
	result = bindings;
}

vector<ColumnBinding> LogicalRecyclerScan::GetColumnBindings() {
	return bindings;
}

PhysicalOperator &LogicalRecyclerScan::CreatePlan(
    ClientContext &context,
    PhysicalPlanGenerator &planner) {

	D_ASSERT(entry);
	D_ASSERT(entry->collection);

	return planner.Make<PhysicalRecyclerScan>(
	    types,
	    NumericCast<idx_t>(entry->collection->Count()),
	    entry);
}

// -------------------------------------------------------------------------
// PhysicalRecyclerMaterialize
// -------------------------------------------------------------------------

PhysicalRecyclerMaterialize::PhysicalRecyclerMaterialize(
    PhysicalPlan &physical_plan,
    vector<LogicalType> types,
    idx_t estimated_cardinality,
    PhysicalOperator &child,
    shared_ptr<RecyclingCache> cache_p,
    shared_ptr<RecyclingCacheEntry> entry_p)
    : PhysicalOperator(
          physical_plan,
          PhysicalOperatorType::EXTENSION,
          std::move(types),
          estimated_cardinality),
      cache(std::move(cache_p)),
      entry(std::move(entry_p)) {

	children.push_back(child);

	D_ASSERT(cache);
	D_ASSERT(entry);
	D_ASSERT(entry->collection);
}

PhysicalRecyclerMaterialize::~PhysicalRecyclerMaterialize() {
	if (!finalized && cache && entry) {
		cache->MarkFailed(entry);
	}
}

unique_ptr<GlobalSinkState>
PhysicalRecyclerMaterialize::GetGlobalSinkState(
    ClientContext &context) const {

	D_ASSERT(entry);
	D_ASSERT(entry->collection);

	return make_uniq<RecyclerGlobalSinkState>(entry);
}

unique_ptr<LocalSinkState>
PhysicalRecyclerMaterialize::GetLocalSinkState(
    ExecutionContext &context) const {

	return make_uniq<RecyclerLocalSinkState>();
}

SinkResultType PhysicalRecyclerMaterialize::Sink(
    ExecutionContext &context,
    DataChunk &chunk,
    OperatorSinkInput &input) const {

	auto &state =
	    input.global_state.Cast<RecyclerGlobalSinkState>();

	state.entry->collection->Append(
	    state.append_state,
	    chunk);

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalRecyclerMaterialize::Combine(
    ExecutionContext &context,
    OperatorSinkCombineInput &input) const {

	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalRecyclerMaterialize::Finalize(
    Pipeline &pipeline,
    Event &event,
    ClientContext &context,
    OperatorSinkFinalizeInput &input) const {

	D_ASSERT(cache);
	D_ASSERT(entry);

	cache->MarkReady(entry);

	finalized = true;

	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSourceState>
PhysicalRecyclerMaterialize::GetGlobalSourceState(
    ClientContext &context) const {

	D_ASSERT(entry);
	D_ASSERT(entry->collection);

	return make_uniq<RecyclerGlobalSourceState>(entry);
}

unique_ptr<LocalSourceState>
PhysicalRecyclerMaterialize::GetLocalSourceState(
    ExecutionContext &context,
    GlobalSourceState &gstate) const {

	return make_uniq<RecyclerLocalSourceState>();
}

SourceResultType PhysicalRecyclerMaterialize::GetDataInternal(
    ExecutionContext &context,
    DataChunk &chunk,
    OperatorSourceInput &input) const {

	auto &state =
	    input.global_state.Cast<RecyclerGlobalSourceState>();

	bool has_more =
	    state.entry->collection->Scan(
	        state.scan_state,
	        chunk);

	if (!has_more || chunk.size() == 0) {
		return SourceResultType::FINISHED;
	}

	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string>
PhysicalRecyclerMaterialize::ParamsToString() const {

	InsertionOrderPreservingMap<string> result;

	if (entry && entry->collection) {
		result["Recycler"] = "MATERIALIZE";
		result["Rows"] =
		    StringUtil::Format(
		        "%llu",
		        entry->collection->Count());

		result["Bytes"] =
		    StringUtil::Format(
		        "%llu",
		        entry->collection->SizeInBytes());
	}

	SetEstimatedCardinality(
	    result,
	    estimated_cardinality);

	return result;
}

// -------------------------------------------------------------------------
// PhysicalRecyclerScan
// -------------------------------------------------------------------------

PhysicalRecyclerScan::PhysicalRecyclerScan(
    PhysicalPlan &physical_plan,
    vector<LogicalType> types,
    idx_t estimated_cardinality,
    shared_ptr<RecyclingCacheEntry> entry_p)
    : PhysicalOperator(
          physical_plan,
          PhysicalOperatorType::EXTENSION,
          std::move(types),
          estimated_cardinality),
      entry(std::move(entry_p)) {

	D_ASSERT(entry);
	D_ASSERT(entry->collection);
}

unique_ptr<GlobalSourceState>
PhysicalRecyclerScan::GetGlobalSourceState(
    ClientContext &context) const {

	D_ASSERT(entry);
	D_ASSERT(entry->collection);

	return make_uniq<RecyclerGlobalSourceState>(entry);
}

unique_ptr<LocalSourceState>
PhysicalRecyclerScan::GetLocalSourceState(
    ExecutionContext &context,
    GlobalSourceState &gstate) const {

	return make_uniq<RecyclerLocalSourceState>();
}

SourceResultType PhysicalRecyclerScan::GetDataInternal(
    ExecutionContext &context,
    DataChunk &chunk,
    OperatorSourceInput &input) const {

	auto &state =
	    input.global_state.Cast<RecyclerGlobalSourceState>();

	bool has_more =
	    state.entry->collection->Scan(
	        state.scan_state,
	        chunk);

	if (!has_more || chunk.size() == 0) {
		return SourceResultType::FINISHED;
	}

	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string>
PhysicalRecyclerScan::ParamsToString() const {

	InsertionOrderPreservingMap<string> result;

	result["Recycler"] = "CACHE HIT";

	if (entry && entry->collection) {
		result["Rows"] =
		    StringUtil::Format(
		        "%llu",
		        entry->collection->Count());

		result["Bytes"] =
		    StringUtil::Format(
		        "%llu",
		        entry->collection->SizeInBytes());

		result["Hits"] =
		    StringUtil::Format(
		        "%llu",
		        entry->hits);
	}

	SetEstimatedCardinality(
	    result,
	    estimated_cardinality);

	return result;
}

} // namespace duckdb