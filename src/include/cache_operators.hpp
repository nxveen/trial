#pragma once

#include "cost_model.hpp"
#include "recycler.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

//! How a store operator decides, and what it was told when it was placed.
struct StoreDecision {
	hash_t fingerprint = 0;
	//! Speculative (Section III-D): placed on a result never seen before, to be
	//! decided once the result is known; otherwise chosen at rewrite time from
	//! the recycler graph's history (Section III-C).
	bool speculative = false;
	//! Admit even if the replacement policy would refuse (cost model off).
	bool force = false;
	//! The base cost to record if the node has no measured one yet (wall ns).
	double estimated_bcost = 0;
	//! The query that placed the operator: the clock of the recycler.
	idx_t now = 0;
	//! Give up buffering beyond this many bytes or rows: such a result could
	//! never be admitted, or is not the small result storing aims for. Checked
	//! against the real result while it streams past - for speculative store
	//! operators this is the paper's run-time decision.
	idx_t max_bytes = 0;
	idx_t max_rows = NumericLimits<idx_t>::Maximum();
};

//! The paper's store operator (Section II): sits on top of a subtree, passes
//! its rows on unchanged, and keeps a copy, which the recycler's admission and
//! replacement policies decide on once the subtree has produced everything.
//!
//! Two physical forms, chosen by `mode` (see CaptureMode):
//!  - streaming: copies the rows as they pass, in parallel, without breaking
//!    the pipeline - the paper's "without interrupting the tuple flow".
//!    Publishes in OperatorFinalize, which runs once after every thread is
//!    done. Only correct where one pipeline flows through and nothing after
//!    the operator can end that pipeline early; the rewriter only picks it
//!    there, and it refuses to publish if any thread's rows went missing.
//!  - materializing: a pipeline breaker, correct anywhere: every pipeline
//!    feeding it (a UNION or a RIGHT/FULL join has several) shares one sink
//!    state, and Finalize runs exactly once after all of them.
class LogicalCaptureResult : public LogicalExtensionOperator {
public:
	LogicalCaptureResult(shared_ptr<Recycler> recycler, StoreDecision decision, bool ordered, CaptureMode mode);

	shared_ptr<Recycler> recycler;
	StoreDecision decision;
	//! Collect on one thread, in the order rows arrive, and emit them in that
	//! order. Needed where the order of rows is observable.
	bool ordered;
	CaptureMode mode;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	string GetExtensionName() const override;

protected:
	void ResolveTypes() override;
};

//! Replaces a plan, or a subtree of one, whose result is in the recycler
//! cache. It produces the cached rows and reads nothing from storage.
class LogicalCachedScan : public LogicalExtensionOperator {
public:
	LogicalCachedScan(shared_ptr<CachedResult> result, idx_t table_index, bool ordered);

	shared_ptr<CachedResult> result;
	idx_t table_index;
	//! Emit rows on one thread, in the order they were captured.
	bool ordered;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<idx_t> GetTableIndex() const override;
	string GetExtensionName() const override;

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
