#include "cache_operators.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Scanning a collection, shared by the operators below
//===--------------------------------------------------------------------===//

class CollectionScanGlobalState : public GlobalSourceState {
public:
	CollectionScanGlobalState(shared_ptr<ColumnDataCollection> collection_p, bool ordered)
	    : collection(std::move(collection_p)),
	      max_threads(ordered ? 1 : MaxValue<idx_t>(collection->ChunkCount(), 1)) {
		collection->InitializeScan(scan_state);
	}

	idx_t MaxThreads() override {
		return max_threads;
	}

	//! Held here so the rows outlive anything else that lets go of them.
	shared_ptr<ColumnDataCollection> collection;
	ColumnDataParallelScanState scan_state;
	const idx_t max_threads;
};

class CollectionScanLocalState : public LocalSourceState {
public:
	ColumnDataLocalScanState scan_state;
};

static SourceResultType ScanCollection(DataChunk &chunk, OperatorSourceInput &input) {
	auto &gstate = input.global_state.Cast<CollectionScanGlobalState>();
	auto &lstate = input.local_state.Cast<CollectionScanLocalState>();
	gstate.collection->Scan(gstate.scan_state, lstate.scan_state, chunk);
	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//! Hands a complete result to the recycler, which admits it or not.
static void Publish(Recycler &recycler, const StoreDecision &decision, const vector<LogicalType> &types,
                    shared_ptr<ColumnDataCollection> data, bool ordered) {
	auto result = make_shared_ptr<CachedResult>();
	result->types = types;
	result->bytes = data->SizeInBytes();
	result->data = std::move(data);
	result->ordered = ordered;
	recycler.Store(decision.fingerprint, std::move(result), decision.estimated_bcost, decision.speculative,
	               decision.force, decision.now);
}

//===--------------------------------------------------------------------===//
// Materializing store operator
//===--------------------------------------------------------------------===//

class CaptureGlobalSinkState : public GlobalSinkState {
public:
	mutex lock;
	//! Everything the subtree produced, buffer-managed so it respects
	//! memory_limit and can spill while the query runs.
	unique_ptr<ColumnDataCollection> collected;
	//! What the source half emits, set in Finalize.
	shared_ptr<ColumnDataCollection> output;
};

class CaptureLocalSinkState : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collected;
	ColumnDataAppendState append_state;
};

class PhysicalCaptureResult : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalCaptureResult(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                      shared_ptr<Recycler> recycler, StoreDecision decision, bool ordered)
	    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality),
	      recycler(std::move(recycler)), decision(decision), ordered(ordered) {
	}

	shared_ptr<Recycler> recycler;
	StoreDecision decision;
	bool ordered;

	string GetName() const override {
		return "STORE";
	}

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		// One thread keeps rows in the order the subtree produced them.
		return !ordered;
	}
	bool SinkOrderDependent() const override {
		return ordered;
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		auto state = make_uniq<CaptureGlobalSinkState>();
		state->collected = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), types);
		return std::move(state);
	}

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override {
		auto state = make_uniq<CaptureLocalSinkState>();
		state->collected = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context.client), types);
		state->collected->InitializeAppend(state->append_state);
		return std::move(state);
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &lstate = input.local_state.Cast<CaptureLocalSinkState>();
		lstate.collected->Append(lstate.append_state, chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override {
		auto &gstate = input.global_state.Cast<CaptureGlobalSinkState>();
		auto &lstate = input.local_state.Cast<CaptureLocalSinkState>();
		lock_guard<mutex> guard(gstate.lock);
		gstate.collected->Combine(*lstate.collected);
		return SinkCombineResultType::FINISHED;
	}

	//! Runs once, after every pipeline feeding this operator has finished.
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override {
		auto &gstate = input.global_state.Cast<CaptureGlobalSinkState>();
		if (gstate.collected->SizeInBytes() > decision.max_bytes || gstate.collected->Count() > decision.max_rows) {
			// Too big to be worth keeping: just pass the rows on.
			recycler->MarkOversized(decision.fingerprint);
			gstate.output = std::move(gstate.collected);
			return SinkFinalizeType::READY;
		}
		// The cached copy must not live in the buffer manager: cached results
		// outlive the query that made them, and DuckDB tears the buffer manager
		// down before extension state.
		auto copy = make_shared_ptr<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
		ColumnDataAppendState append_state;
		copy->InitializeAppend(append_state);
		ColumnDataScanState scan_state;
		gstate.collected->InitializeScan(scan_state);
		DataChunk chunk;
		gstate.collected->InitializeScanChunk(scan_state, chunk);
		while (gstate.collected->Scan(scan_state, chunk)) {
			copy->Append(append_state, chunk);
		}
		gstate.collected.reset();
		Publish(*recycler, decision, types, copy, ordered);
		gstate.output = std::move(copy);
		return SinkFinalizeType::READY;
	}

	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return !ordered;
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		auto &gstate = sink_state->Cast<CaptureGlobalSinkState>();
		return make_uniq<CollectionScanGlobalState>(gstate.output, ordered);
	}

	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override {
		return make_uniq<CollectionScanLocalState>();
	}

protected:
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		return ScanCollection(chunk, input);
	}
};

//===--------------------------------------------------------------------===//
// Streaming store operator
//===--------------------------------------------------------------------===//

class StreamingCaptureGlobalState : public GlobalOperatorState {
public:
	mutex lock;
	//! In-memory (not buffer-managed) from the start, so publishing needs no copy.
	unique_ptr<ColumnDataCollection> collected;
	//! Set once the rows outgrow what the cache would take: stop copying (the
	//! paper's "cancels buffering and reverts to passing its input along").
	atomic<bool> gave_up {false};
	//! Thread states handed out, and thread states whose rows were merged.
	//! Publishing requires the two to match: a thread whose share never
	//! reached FinalExecute (an operator below ended the pipeline on the last
	//! chunk of a task, say) would otherwise leave a hole in the result.
	atomic<idx_t> started {0};
	atomic<idx_t> merged {0};
};

class StreamingCaptureLocalState : public OperatorState {
public:
	unique_ptr<ColumnDataCollection> collected;
	ColumnDataAppendState append_state;
};

class PhysicalStreamingCapture : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalStreamingCapture(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                         shared_ptr<Recycler> recycler, StoreDecision decision, bool ordered)
	    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality),
	      recycler(std::move(recycler)), decision(decision), ordered(ordered) {
	}

	shared_ptr<Recycler> recycler;
	StoreDecision decision;
	bool ordered;

	string GetName() const override {
		return "STORE_STREAMING";
	}

	bool ParallelOperator() const override {
		return !ordered;
	}
	bool RequiresFinalExecute() const override {
		return true;
	}
	bool RequiresOperatorFinalize() const override {
		return true;
	}

	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override {
		auto state = make_uniq<StreamingCaptureGlobalState>();
		state->collected = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
		return std::move(state);
	}

	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override {
		if (op_state) {
			op_state->Cast<StreamingCaptureGlobalState>().started++;
		}
		auto state = make_uniq<StreamingCaptureLocalState>();
		state->collected = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
		state->collected->InitializeAppend(state->append_state);
		return std::move(state);
	}

	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate_p, OperatorState &state_p) const override {
		auto &gstate = gstate_p.Cast<StreamingCaptureGlobalState>();
		auto &lstate = state_p.Cast<StreamingCaptureLocalState>();
		if (lstate.collected && !gstate.gave_up) {
			lstate.collected->Append(lstate.append_state, input);
			if (lstate.collected->SizeInBytes() > decision.max_bytes ||
			    lstate.collected->Count() > decision.max_rows) {
				gstate.gave_up = true;
			}
		}
		if (gstate.gave_up && lstate.collected) {
			lstate.collected.reset();
		}
		chunk.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	//! Runs once per thread, when its share of the pipeline is done.
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate_p,
	                                        OperatorState &state_p) const override {
		auto &gstate = gstate_p.Cast<StreamingCaptureGlobalState>();
		auto &lstate = state_p.Cast<StreamingCaptureLocalState>();
		if (lstate.collected && !gstate.gave_up) {
			lock_guard<mutex> guard(gstate.lock);
			gstate.collected->Combine(*lstate.collected);
		}
		if (lstate.collected) {
			gstate.merged++;
		}
		lstate.collected.reset();
		return OperatorFinalizeResultType::FINISHED;
	}

	//! Runs once, after every thread of the pipeline has finished.
	OperatorFinalResultType OperatorFinalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                                         OperatorFinalizeInput &input) const override {
		auto &gstate = input.global_state.Cast<StreamingCaptureGlobalState>();
		if (gstate.gave_up || gstate.collected->SizeInBytes() > decision.max_bytes ||
		    gstate.collected->Count() > decision.max_rows) {
			recycler->MarkOversized(decision.fingerprint);
			gstate.collected.reset();
			return OperatorFinalResultType::FINISHED;
		}
		if (gstate.started == 0 || gstate.merged != gstate.started) {
			// Some thread's rows are missing: keep nothing rather than half.
			gstate.collected.reset();
			return OperatorFinalResultType::FINISHED;
		}
		shared_ptr<ColumnDataCollection> data = std::move(gstate.collected);
		Publish(*recycler, decision, types, std::move(data), ordered);
		return OperatorFinalResultType::FINISHED;
	}
};

LogicalCaptureResult::LogicalCaptureResult(shared_ptr<Recycler> recycler_p, StoreDecision decision_p, bool ordered_p,
                                           CaptureMode mode_p)
    : recycler(std::move(recycler_p)), decision(decision_p), ordered(ordered_p), mode(mode_p) {
}

void LogicalCaptureResult::ResolveTypes() {
	types = children[0]->types;
}

vector<ColumnBinding> LogicalCaptureResult::GetColumnBindings() {
	// Pass-through: nothing above this operator needs rewriting.
	return children[0]->GetColumnBindings();
}

string LogicalCaptureResult::GetName() const {
	return "STORE";
}

string LogicalCaptureResult::GetExtensionName() const {
	return "waddle";
}

PhysicalOperator &LogicalCaptureResult::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	auto &child = planner.CreatePlan(*children[0]);
	if (mode == CaptureMode::MATERIALIZING) {
		auto &op = planner.Make<PhysicalCaptureResult>(types, estimated_cardinality, recycler, decision, ordered);
		op.children.push_back(child);
		return op;
	}
	auto &op = planner.Make<PhysicalStreamingCapture>(types, estimated_cardinality, recycler, decision,
	                                                  mode == CaptureMode::STREAMING_ORDERED);
	op.children.push_back(child);
	return op;
}

//===--------------------------------------------------------------------===//
// Cached scan
//===--------------------------------------------------------------------===//

class PhysicalCachedScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalCachedScan(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                   shared_ptr<CachedResult> result, bool ordered)
	    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality), result(std::move(result)),
	      ordered(ordered) {
	}

	//! Holding the shared_ptr keeps the rows alive for as long as this plan
	//! exists, even if the recycler evicts the result mid-query.
	shared_ptr<CachedResult> result;
	bool ordered;

	string GetName() const override {
		return "CACHED_SCAN";
	}

	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return !ordered;
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		return make_uniq<CollectionScanGlobalState>(result->data, ordered);
	}

	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override {
		return make_uniq<CollectionScanLocalState>();
	}

protected:
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		return ScanCollection(chunk, input);
	}
};

LogicalCachedScan::LogicalCachedScan(shared_ptr<CachedResult> result_p, idx_t table_index_p, bool ordered_p)
    : result(std::move(result_p)), table_index(table_index_p), ordered(ordered_p) {
	types = result->types;
	SetEstimatedCardinality(result->data->Count());
}

void LogicalCachedScan::ResolveTypes() {
	types = result->types;
}

vector<ColumnBinding> LogicalCachedScan::GetColumnBindings() {
	return GenerateColumnBindings(table_index, types.size());
}

vector<idx_t> LogicalCachedScan::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalCachedScan::GetName() const {
	return "CACHED_SCAN";
}

string LogicalCachedScan::GetExtensionName() const {
	return "waddle";
}

PhysicalOperator &LogicalCachedScan::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	return planner.Make<PhysicalCachedScan>(types, estimated_cardinality, result, ordered);
}

} // namespace duckdb
