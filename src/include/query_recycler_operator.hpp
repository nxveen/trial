#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

#include "query_recycler_cache.hpp"

namespace duckdb {

class PhysicalPlanGenerator;

// -------------------------------------------------------------------------
// LogicalRecyclerMaterialize
// -------------------------------------------------------------------------

class LogicalRecyclerMaterialize : public LogicalExtensionOperator {
public:
	LogicalRecyclerMaterialize(
	    unique_ptr<LogicalOperator> child,
	    shared_ptr<RecyclingCache> cache_p,
	    shared_ptr<RecyclingCacheEntry> entry_p);

	void ResolveTypes() override;

	void ResolveColumnBindings(
	    ColumnBindingResolver &res,
	    vector<ColumnBinding> &bindings) override;

	vector<ColumnBinding> GetColumnBindings() override;

	PhysicalOperator &CreatePlan(
	    ClientContext &context,
	    PhysicalPlanGenerator &planner) override;

private:
	vector<ColumnBinding> bindings;

	shared_ptr<RecyclingCache> cache;
	shared_ptr<RecyclingCacheEntry> entry;
};

// -------------------------------------------------------------------------
// LogicalRecyclerScan
// -------------------------------------------------------------------------
//
// On a cache hit, the physical data comes from the cached collection.
//
// IMPORTANT:
// The ColumnBindings belong to the CURRENT query. They must not be copied
// from RecyclingCacheEntry because that entry may have been created by a
// previous query compilation.
// -------------------------------------------------------------------------

class LogicalRecyclerScan : public LogicalExtensionOperator {
public:
	LogicalRecyclerScan(
	    shared_ptr<RecyclingCacheEntry> entry_p,
	    vector<ColumnBinding> current_bindings);

	void ResolveTypes() override;

	void ResolveColumnBindings(
	    ColumnBindingResolver &res,
	    vector<ColumnBinding> &bindings) override;

	vector<ColumnBinding> GetColumnBindings() override;

	PhysicalOperator &CreatePlan(
	    ClientContext &context,
	    PhysicalPlanGenerator &planner) override;

private:
	vector<ColumnBinding> bindings;

	shared_ptr<RecyclingCacheEntry> entry;
};

// -------------------------------------------------------------------------
// PhysicalRecyclerMaterialize
// -------------------------------------------------------------------------

class PhysicalRecyclerMaterialize : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE =
	    PhysicalOperatorType::EXTENSION;

	PhysicalRecyclerMaterialize(
	    PhysicalPlan &physical_plan,
	    vector<LogicalType> types,
	    idx_t estimated_cardinality,
	    PhysicalOperator &child,
	    shared_ptr<RecyclingCache> cache,
	    shared_ptr<RecyclingCacheEntry> entry);

	~PhysicalRecyclerMaterialize() override;

	// Sink.
	unique_ptr<GlobalSinkState> GetGlobalSinkState(
	    ClientContext &context) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(
	    ExecutionContext &context) const override;

	SinkResultType Sink(
	    ExecutionContext &context,
	    DataChunk &chunk,
	    OperatorSinkInput &input) const override;

	SinkCombineResultType Combine(
	    ExecutionContext &context,
	    OperatorSinkCombineInput &input) const override;

	SinkFinalizeType Finalize(
	    Pipeline &pipeline,
	    Event &event,
	    ClientContext &context,
	    OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return false;
	}

	// Source.
	unique_ptr<GlobalSourceState> GetGlobalSourceState(
	    ClientContext &context) const override;

	unique_ptr<LocalSourceState> GetLocalSourceState(
	    ExecutionContext &context,
	    GlobalSourceState &gstate) const override;

	SourceResultType GetDataInternal(
	    ExecutionContext &context,
	    DataChunk &chunk,
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

// -------------------------------------------------------------------------
// PhysicalRecyclerScan
// -------------------------------------------------------------------------

class PhysicalRecyclerScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE =
	    PhysicalOperatorType::EXTENSION;

	PhysicalRecyclerScan(
	    PhysicalPlan &physical_plan,
	    vector<LogicalType> types,
	    idx_t estimated_cardinality,
	    shared_ptr<RecyclingCacheEntry> entry);

	unique_ptr<GlobalSourceState> GetGlobalSourceState(
	    ClientContext &context) const override;

	unique_ptr<LocalSourceState> GetLocalSourceState(
	    ExecutionContext &context,
	    GlobalSourceState &gstate) const override;

	SourceResultType GetDataInternal(
	    ExecutionContext &context,
	    DataChunk &chunk,
	    OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return false;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

private:
	shared_ptr<RecyclingCacheEntry> entry;
};

} // namespace duckdb