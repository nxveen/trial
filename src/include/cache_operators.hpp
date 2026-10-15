#pragma once
#include "cost_model.hpp"
#include "recycler.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
namespace duckdb {
struct StoreDecision {
	hash_t fingerprint = 0;
	bool speculative = false;
	bool force = false;
	double estimated_bcost = 0;
	idx_t now = 0;
	idx_t max_bytes = 0;
	idx_t max_rows = NumericLimits<idx_t>::Maximum();
};

class LogicalCaptureResult : public LogicalExtensionOperator {
public:
	LogicalCaptureResult(shared_ptr<Recycler> recycler, StoreDecision decision, bool ordered, CaptureMode mode);
	shared_ptr<Recycler> recycler;
	StoreDecision decision;
	bool ordered;
	CaptureMode mode;
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	string GetExtensionName() const override;
protected:
	void ResolveTypes() override;
};

class LogicalCachedScan : public LogicalExtensionOperator {
public:
	LogicalCachedScan(shared_ptr<CachedResult> result, idx_t table_index, bool ordered);
	shared_ptr<CachedResult> result;
	idx_t table_index;
	bool ordered;
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<idx_t> GetTableIndex() const override;
	string GetExtensionName() const override;
protected:
	void ResolveTypes() override;
};
}

