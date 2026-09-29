#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

#include "query_recycler_cache.hpp"

namespace duckdb {

class PhysicalPlanGenerator;

//! Logical wrapper inserted around a query on a cache miss.
class LogicalQueryRecycler : public LogicalExtensionOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;

	LogicalQueryRecycler(unique_ptr<LogicalOperator> child, shared_ptr<RecyclingCache> cache,
	                     shared_ptr<RecyclingCacheEntry> entry, idx_t table_index);

	shared_ptr<RecyclingCache> cache;
	shared_ptr<RecyclingCacheEntry> entry;
	idx_t table_index;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

	string GetExtensionName() const override {
		return "waddle_query_recycler";
	}

	bool SupportSerialization() const override {
		return false;
	}

	vector<ColumnBinding> GetColumnBindings() override;

protected:
	void ResolveTypes() override;
};

//! Physical operator that materializes the child into a ColumnDataCollection
//! and then exposes the collection as a source.
class PhysicalQueryRecycler : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalQueryRecycler(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                      PhysicalOperator &child, shared_ptr<RecyclingCache> cache,
	                      shared_ptr<RecyclingCacheEntry> entry);

	~PhysicalQueryRecycler() override;

	// Sink side.
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
	                    OperatorSinkInput &input) const override;

	SinkCombineResultType Combine(ExecutionContext &context,
	                              OperatorSinkCombineInput &input) const override;

	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return false;
	}

	// Source side.
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;

	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return false;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

private:
	shared_ptr<RecyclingCache> cache;
	shared_ptr<RecyclingCacheEntry> entry;

	mutable bool finalized = false;
};

} // namespace duckdb