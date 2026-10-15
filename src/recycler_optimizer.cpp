#include "recycler_optimizer.hpp"
#include "duckdb/main/config.hpp"
#include "cache_operators.hpp"
#include "cost_model.hpp"
#include "recycler.hpp"
#include "plan_key.hpp"
#include "plan_rewrite.hpp"
#include "subquery_pushdown.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_explain.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include <algorithm>
#include <chrono>
#include <unordered_map>
namespace duckdb {
// Operators we understand well enough to promise that the same plan always produces the same rows.
static bool IsPureReadOperator(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_WINDOW:
	case LogicalOperatorType::LOGICAL_UNNEST:
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
	case LogicalOperatorType::LOGICAL_CTE_REF:
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		return true;
	default:
		return false;
	}
}

// Statements that change what later queries should see: data, schema, settings, or transaction state.
static bool ChangesDatabaseState(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO:
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
	case LogicalOperatorType::LOGICAL_CREATE_SECRET:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_SET:
	case LogicalOperatorType::LOGICAL_RESET:
	case LogicalOperatorType::LOGICAL_PRAGMA:
	case LogicalOperatorType::LOGICAL_VACUUM:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_COPY_DATABASE:
	case LogicalOperatorType::LOGICAL_EXPORT:
	case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
		return true;
	default:
		return false;
	}
}

// Operators that read something produced elsewhere in the same plan.
static bool DependsOnSurroundingPlan(LogicalOperatorType type) {
	return type == LogicalOperatorType::LOGICAL_DELIM_GET || type == LogicalOperatorType::LOGICAL_CTE_REF;
}

// Regions of a plan we leave alone when working below the root.
static bool BlocksDescent(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		return true;
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return true;
	default:
		return false;
	}
}

static bool ExpressionsAreStable(LogicalOperator &op) {
	bool stable = true;
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) {
		auto &expr = **child;
		if (expr.IsVolatile() || !expr.IsConsistent()) {
			stable = false;
		}
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_PARAMETER) {
			stable = false;
		}
	});
	return stable;
}

// Whether this one operator, ignoring its children, may be part of a cached plan.
static bool OperatorIsCacheable(LogicalOperator &op) {
	if (!IsPureReadOperator(op.type)) {
		return false;
	}
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		if (!op.Cast<LogicalGet>().GetTable()) {
			return false;
		}
	}
	return ExpressionsAreStable(op);
}

static bool ChangesState(LogicalOperator &op) {
	if (ChangesDatabaseState(op.type)) {
		return true;
	}
	for (auto &child : op.children) {
		if (ChangesState(*child)) {
			return true;
		}
	}
	return false;
}

// Whether the order of the query's own result matters: with preserve_insertion_order on it always does...
static bool RootOrderMatters(LogicalOperator &root, bool preserve_insertion_order) {
	if (preserve_insertion_order) {
		return true;
	}
	reference<LogicalOperator> op = root;
	while (true) {
		switch (op.get().type) {
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_TOP_N:
			return true;
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_FILTER:
		case LogicalOperatorType::LOGICAL_LIMIT:
			if (op.get().children.size() != 1) {
				return false;
			}
			op = *op.get().children[0];
			break;
		default:
			return false;
		}
	}
}

// Whether the order in which child `child_idx` produces its rows can change what `parent` produces, given...
static bool ChildOrderMatters(LogicalOperator &parent, idx_t child_idx, bool parent_order_matters,
                              bool preserve_insertion_order) {
	switch (parent.type) {
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return false;
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
	case LogicalOperatorType::LOGICAL_LIMIT:
		return true;
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return child_idx == 0 ? parent_order_matters : false;
	case LogicalOperatorType::LOGICAL_WINDOW:
		return preserve_insertion_order;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		return child_idx == 0 ? preserve_insertion_order : parent_order_matters;
	default:
		return parent_order_matters;
	}
}

struct PlanNode {
	unique_ptr<LogicalOperator> *slot = nullptr;
	idx_t end = 0;
	idx_t parent = DConstants::INVALID_INDEX;
	idx_t child_index = 0;
	bool is_root = false;
	bool eligible = false;
	bool order_matters = false;
	bool pure = false;
	bool dependent = false;
	idx_t operators = 0;
	idx_t work = 0;
	idx_t rows = 0;
	double cost = 0;
	double base_cost = 0;
	double estimated_rows = 0;
	double estimated_bytes = 0;
	bool valid = false;
	bool capturable = false;
	string key;
	hash_t fingerprint = 0;
	bool replaced = false;
	bool existed = false;
	shared_ptr<CachedResult> cached;
	bool reused = false;
	Recycler::Assessment assessment;
	CaptureMode mode = CaptureMode::MATERIALIZING;
};

struct DynamicFilterLink {
	const void *filter;
	idx_t node;
};

struct PlanAnalysis {
	PlanAnalysis(ClientContext &context, bool preserve_insertion_order, CostParameters parameters)
	    : context(context), preserve_insertion_order(preserve_insertion_order), parameters(parameters) {
	}
	ClientContext &context;
	bool preserve_insertion_order;
	CostParameters parameters;
	vector<PlanNode> nodes;
	vector<DynamicFilterLink> producers;
	vector<DynamicFilterLink> targets;
};

// Collects the Top-N dynamic filters inside one pushed-down table filter.
static void FindDynamicFilters(const TableFilter &filter, idx_t node, vector<DynamicFilterLink> &out) {
	switch (filter.filter_type) {
	case TableFilterType::DYNAMIC_FILTER:
		out.push_back(DynamicFilterLink {filter.Cast<DynamicFilter>().filter_data.get(), node});
		break;
	case TableFilterType::OPTIONAL_FILTER: {
		auto &child = filter.Cast<OptionalFilter>().child_filter;
		if (child) {
			FindDynamicFilters(*child, node, out);
		}
		break;
	}
	case TableFilterType::STRUCT_EXTRACT: {
		auto &child = filter.Cast<StructFilter>().child_filter;
		if (child) {
			FindDynamicFilters(*child, node, out);
		}
		break;
	}
	case TableFilterType::CONJUNCTION_OR:
	case TableFilterType::CONJUNCTION_AND:
		for (auto &child : static_cast<const ConjunctionFilter &>(filter).child_filters) {
			FindDynamicFilters(*child, node, out);
		}
		break;
	default:
		break;
	}
}

// True if `op` is a LIMIT whose input is a sort (through projections only).
static bool LimitOverSort(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_LIMIT || op.children.size() != 1) {
		return false;
	}
	reference<LogicalOperator> current = *op.children[0];
	while (current.get().type == LogicalOperatorType::LOGICAL_PROJECTION && current.get().children.size() == 1) {
		current = *current.get().children[0];
	}
	return current.get().type == LogicalOperatorType::LOGICAL_ORDER_BY ||
	       current.get().type == LogicalOperatorType::LOGICAL_TOP_N;
}

static void AnalyzeOperator(unique_ptr<LogicalOperator> &slot, bool is_root, bool blocked, bool order_matters,
                            PlanAnalysis &analysis, idx_t parent = DConstants::INVALID_INDEX, idx_t child_index = 0) {
	auto &op = *slot;
	auto index = analysis.nodes.size();
	analysis.nodes.emplace_back();
	analysis.nodes[index].parent = parent;
	analysis.nodes[index].child_index = child_index;
	bool pure = OperatorIsCacheable(op);
	bool dependent = DependsOnSurroundingPlan(op.type);
	idx_t operators = 1;
	idx_t work = op.has_estimated_cardinality ? op.estimated_cardinality : 0;
	double cost = CostModel::OperatorCost(op, analysis.context, analysis.parameters);
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (get.GetTable()) {
			work += CostModel::TableRows(analysis.context, get);
		}
		if (get.dynamic_filters) {
			analysis.targets.push_back(DynamicFilterLink {get.dynamic_filters.get(), index});
		}
		for (auto &entry : get.table_filters.filters) {
			FindDynamicFilters(*entry.second, index, analysis.targets);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.filter_pushdown) {
			for (auto &probe : join.filter_pushdown->probe_info) {
				analysis.producers.push_back(DynamicFilterLink {probe.dynamic_filters.get(), index});
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_TOP_N: {
		auto &top_n = op.Cast<LogicalTopN>();
		if (top_n.dynamic_filter) {
			analysis.producers.push_back(DynamicFilterLink {top_n.dynamic_filter.get(), index});
		}
		break;
	}
	default:
		break;
	}
	auto blocked_below = blocked || (BlocksDescent(op.type) && !LimitOverSort(op));
	for (idx_t i = 0; i < op.children.size(); i++) {
		auto child_index = analysis.nodes.size();
		auto child_order = ChildOrderMatters(op, i, order_matters, analysis.preserve_insertion_order);
		AnalyzeOperator(op.children[i], false, blocked_below, child_order, analysis, index, i);
		auto &child = analysis.nodes[child_index];
		pure = pure && child.pure;
		dependent = dependent || child.dependent;
		operators += child.operators;
		work += child.work;
		cost += child.cost;
	}
	auto &node = analysis.nodes[index];
	node.slot = &slot;
	node.end = analysis.nodes.size();
	node.is_root = is_root;
	node.eligible = is_root || !blocked;
	node.order_matters = order_matters;
	node.pure = pure;
	node.dependent = dependent;
	node.operators = operators;
	node.work = work;
	node.cost = cost;
	node.base_cost = cost;
	node.estimated_rows = CostModel::EstimatedRows(op);
	node.rows = static_cast<idx_t>(node.estimated_rows);
	node.estimated_bytes = node.estimated_rows * CostModel::RowBytes(op.types);
}

// True if every runtime filter that reaches a scan inside nodes [begin, end) also comes from inside it.
static bool DynamicFiltersStayInside(const PlanAnalysis &analysis, idx_t begin, idx_t end) {
	for (auto &target : analysis.targets) {
		if (target.node < begin || target.node >= end) {
			continue;
		}
		bool found = false;
		for (auto &producer : analysis.producers) {
			if (producer.filter != target.filter) {
				continue;
			}
			found = true;
			if (producer.node < begin || producer.node >= end) {
				return false;
			}
		}
		if (!found) {
			return false;
		}
	}
	return true;
}
// Decides which subtrees may be replaced or captured, and takes their keys.
static void ComputeKeys(PlanAnalysis &analysis) {
	auto &nodes = analysis.nodes;
	vector<string> keys;
	bool single_pass = !nodes.empty() && SerializeSubtrees(**nodes[0].slot, keys) && keys.size() == nodes.size();
	for (idx_t i = 0; i < nodes.size(); i++) {
		auto &node = nodes[i];
		if (!node.eligible || !node.pure) {
			continue;
		}
		if (!node.is_root && node.dependent) {
			continue;
		}
		if (single_pass) {
			node.key = std::move(keys[i]);
		} else if (!SerializePlan(**node.slot, node.key)) {
			continue;
		}
		node.fingerprint = HashPlanKey(node.key);
		node.valid = true;
		node.capturable = node.is_root || DynamicFiltersStayInside(analysis, i, node.end);
	}
}

// The current value of a waddle setting, or its registered default if it is unset or NULL.
static Value GetSetting(ClientContext &context, const string &name) {
	Value value;
	if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) {
		return value;
	}
	ExtensionOption option;
	if (DBConfig::GetConfig(context).TryGetExtensionOption(name, option)) {
		return option.default_value;
	}
	throw InternalException("Unknown waddle setting \"%s\"", name);
}

static bool BoolSetting(ClientContext &context, const string &name) {
	return GetSetting(context, name).GetValue<bool>();
}

static idx_t CountSetting(ClientContext &context, const string &name) {
	auto count = GetSetting(context, name).GetValue<int64_t>();
	return count < 0 ? 0 : NumericCast<idx_t>(count);
}

static double DoubleSetting(ClientContext &context, const string &name) {
	return GetSetting(context, name).GetValue<double>();
}

static void Log(ClientContext &context, const string &message) {
	if (BoolSetting(context, "waddle_log")) {
		Printer::Print("[waddle] " + message);
	}
}

static const char *ModeName(CaptureMode mode) {
	switch (mode) {
	case CaptureMode::STREAMING:
		return "streaming";
	case CaptureMode::STREAMING_ORDERED:
		return "streaming, ordered";
	default:
		return "materializing";
	}
}

static string Describe(const PlanNode &node, const char *why) {
	auto &a = node.assessment;
	return StringUtil::Format("%s %s (%llu ops, %s; h %.3f, cost %.0fus, size %.0f KB%s, B %.3g)", why,
	                          node.is_root ? string("the whole query") : node.slot->get()->GetName(), node.operators,
	                          ModeName(node.mode), a.h, a.true_cost / 1000.0, a.size / 1024.0,
	                          a.size_measured ? "" : " est.", a.benefit);
}

static bool CanReuse(const shared_ptr<CachedResult> &cached, LogicalOperator &op, const PlanNode &node) {
	if (!cached || cached->types != op.types) {
		return false;
	}
	return !node.order_matters || cached->ordered;
}

// Operators that end a pipeline below them and start a new one: whatever is directly above them runs in exactly...
static bool StartsOnePipeline(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		return true;
	default:
		return false;
	}
}

static bool IsRowByRow(LogicalOperatorType type) {
	return type == LogicalOperatorType::LOGICAL_PROJECTION || type == LogicalOperatorType::LOGICAL_FILTER;
}

// A join whose probe side (child 0) streams through it in the same pipeline, and which emits nothing from a...
static bool StreamsProbeSide(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.children.size() != 2) {
		return false;
	}
	auto &join = op.Cast<LogicalComparisonJoin>();
	switch (join.join_type) {
	case JoinType::INNER:
	case JoinType::LEFT:
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
	case JoinType::SINGLE:
		break;
	default:
		return false;
	}
	for (auto &condition : join.conditions) {
		if (condition.comparison == ExpressionType::COMPARE_EQUAL) {
			return true;
		}
	}
	return false;
}

// True if exactly one pipeline produces this operator's rows: below it, down the side rows stream through...
static bool FlowsFromOnePipeline(LogicalOperator &op) {
	reference<LogicalOperator> current = op;
	while (true) {
		auto &node = current.get();
		if (IsRowByRow(node.type) && node.children.size() == 1) {
			current = *node.children[0];
		} else if (StreamsProbeSide(node)) {
			current = *node.children[0];
		} else {
			return StartsOnePipeline(node.type);
		}
	}
}

// True if whatever consumes this node's rows takes every one of them: above it there are only projections and...
static bool ConsumedWhole(const vector<PlanNode> &nodes, idx_t index) {
	idx_t current = index;
	while (true) {
		auto parent = nodes[current].parent;
		if (parent == DConstants::INVALID_INDEX) {
			return true;
		}
		auto &op = **nodes[parent].slot;
		if (IsRowByRow(op.type)) {
			current = parent;
			continue;
		}
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_TOP_N:
		case LogicalOperatorType::LOGICAL_DISTINCT:
			return true;
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
			if (nodes[current].child_index == 1) {
				return true;
			}
			if (StreamsProbeSide(op)) {
				current = parent;
				continue;
			}
			return false;
		default:
			return false;
		}
	}
}

static CaptureMode ChooseCaptureMode(const vector<PlanNode> &nodes, idx_t index) {
	auto &node = nodes[index];
	auto &op = **node.slot;
	if (!FlowsFromOnePipeline(op) || !ConsumedWhole(nodes, index)) {
		return CaptureMode::MATERIALIZING;
	}
	if (!node.order_matters) {
		return CaptureMode::STREAMING;
	}
	if (node.is_root && op.type != LogicalOperatorType::LOGICAL_GET) {
		reference<LogicalOperator> current = op;
		while (IsRowByRow(current.get().type) && current.get().children.size() == 1) {
			current = *current.get().children[0];
		}
		if (current.get().type != LogicalOperatorType::LOGICAL_GET) {
			return CaptureMode::STREAMING_ORDERED;
		}
	}
	return CaptureMode::MATERIALIZING;
}

// Operators the paper places speculative store operators after: ones "expected to be computationally expensive...
static bool IsSpeculationPoint(const PlanNode &node, double estimated_bcost, double expensive_join_cost) {
	if (node.is_root) {
		return true;
	}
	auto type = node.slot->get()->type;
	if (type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY || type == LogicalOperatorType::LOGICAL_DISTINCT) {
		return true;
	}
	return expensive_join_cost > 0 && type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	       estimated_bcost >= expensive_join_cost;
}

class QueryTimer : public ClientContextState {
public:
	QueryTimer(shared_ptr<Recycler> recycler_p, shared_ptr<CostModel> cost_model_p)
	    : recycler(std::move(recycler_p)), cost_model(std::move(cost_model_p)) {
	}

	void QueryBegin(ClientContext &context) override {
		start = std::chrono::steady_clock::now();
		started = true;
		armed = false;
		ClearUnoptimized();
	}
	bool unoptimized_valid = false;
	hash_t unoptimized_fingerprint = 0;
	string unoptimized_key;
	bool answered_early = false;
	void ClearUnoptimized() {
		unoptimized_valid = false;
		unoptimized_key.clear();
		answered_early = false;
	}

	void QueryEnd(ClientContext &context, optional_ptr<ErrorData> error) override {
		if (armed && started && !(error && error->HasError()) && estimated > 0) {
			auto elapsed = std::chrono::steady_clock::now() - start;
			auto measured = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
			cost_model->RecordRatio(measured, estimated);
			auto ratio = measured / estimated;
			for (auto &entry : computed) {
				recycler->SetBaseCost(entry.first, entry.second * ratio);
			}
		}
		armed = false;
		started = false;
		computed.clear();
	}

	// Called while planning: `estimated` is the query's own estimated wall time (after reuse), `computed_p` the...
	void Arm(double estimated_p, vector<pair<hash_t, double>> computed_p) {
		estimated = estimated_p;
		computed = std::move(computed_p);
		armed = true;
	}
private:
	shared_ptr<Recycler> recycler;
	shared_ptr<CostModel> cost_model;
	std::chrono::steady_clock::time_point start;
	bool started = false;
	bool armed = false;
	double estimated = 0;
	vector<pair<hash_t, double>> computed;
};

static bool PlanIsCacheable(LogicalOperator &op) {
	if (!OperatorIsCacheable(op)) {
		return false;
	}
	for (auto &child : op.children) {
		if (!PlanIsCacheable(*child)) {
			return false;
		}
	}
	return true;
}

// The pre-optimize hook.
static void RecycleBeforeOptimizing(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto &state = static_cast<RecyclerState &>(*input.info);
	auto &context = input.context;
	if (BoolSetting(context, "waddle_decorrelate_exists")) {
		state.subqueries_decorrelated += DecorrelateExistsSubqueries(plan);
	}
	if (BoolSetting(context, "waddle_push_subqueries")) {
		state.subqueries_pushed +=
		    PushDownCorrelatedSubqueries(context, plan, BoolSetting(context, "waddle_push_semi_joins"));
	}
	if (!BoolSetting(context, "waddle_cache")) {
		return;
	}
	auto timer =
	    context.registered_state->GetOrCreate<QueryTimer>("waddle_query_timer", state.recycler, state.cost_model);
	timer->ClearUnoptimized();
	reference<unique_ptr<LogicalOperator>> slot = plan;
	bool executes = true;
	while (slot.get()->type == LogicalOperatorType::LOGICAL_EXPLAIN && !slot.get()->children.empty()) {
		if (slot.get()->Cast<LogicalExplain>().explain_type != ExplainType::EXPLAIN_ANALYZE) {
			executes = false;
		}
		slot = slot.get()->children[0];
	}
	auto &query = *slot.get();
	if (ChangesState(query) || !PlanIsCacheable(query)) {
		return;
	}
	string key;
	if (!SerializePlan(query, key)) {
		return;
	}
	auto fingerprint = HashPlanKey(key);
	hash_t root = 0;
	auto cached = state.recycler->LookupAlias(fingerprint, key, root);
	timer->unoptimized_valid = true;
	timer->unoptimized_fingerprint = fingerprint;
	timer->unoptimized_key = std::move(key);
	if (!cached) {
		return;
	}
	query.ResolveOperatorTypes();
	auto order_matters = RootOrderMatters(query, Settings::Get<PreserveInsertionOrderSetting>(context));
	if (cached->types != query.types || (order_matters && !cached->ordered)) {
		return;
	}
	auto rows = cached->data->Count();
	idx_t now = ++state.queries_seen;
	if (executes) {
		state.recycler->Reference(root, now);
	}
	slot.get() = make_uniq<LogicalCachedScan>(std::move(cached), input.optimizer.binder.GenerateTableIndex(),
	                                           order_matters);
	timer->answered_early = true;
	state.whole_query_hits++;
	Log(context, StringUtil::Format("query %llu answered from the recycler cache before optimizing (%llu rows)", now,
	                                rows));
}

static void WrapWithStore(PlanNode &node, RecyclerState &state, const StoreDecision &decision) {
	auto &slot = *node.slot;
	auto store = make_uniq<LogicalCaptureResult>(state.recycler, decision, node.order_matters, node.mode);
	if (slot->has_estimated_cardinality) {
		store->SetEstimatedCardinality(slot->estimated_cardinality);
	}
	store->children.push_back(std::move(slot));
	store->ResolveOperatorTypes();
	slot = std::move(store);
}

static void RecycleResults(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto &state = static_cast<RecyclerState &>(*input.info);
	auto &context = input.context;
	auto &recycler = *state.recycler;
	auto &cost_model = *state.cost_model;
	if (!BoolSetting(context, "waddle_cache")) {
		return;
	}
	auto timer =
	    context.registered_state->GetOrCreate<QueryTimer>("waddle_query_timer", state.recycler, state.cost_model);
	if (timer->answered_early) {
		timer->answered_early = false;
		return;
	}
	auto benefit_policies = BoolSetting(context, "waddle_cost_model");
	auto alpha = MinValue(MaxValue(DoubleSetting(context, "waddle_cost_decay"), 0.0), 1.0);
	recycler.Configure(CountSetting(context, "waddle_cache_limit_mb") * 1024 * 1024, alpha);
	auto root_row_limit = CountSetting(context, "waddle_cache_max_rows");
	auto subtree_row_limit = CountSetting(context, "waddle_subtree_max_rows");
	auto aggregate_row_limit = CountSetting(context, "waddle_aggregate_max_rows");
	auto max_subtree_stores = CountSetting(context, "waddle_max_subtree_captures");
	auto subtrees_enabled = BoolSetting(context, "waddle_subtree_reuse");
	auto expensive_join_cost = DoubleSetting(context, "waddle_speculate_join_us") * 1000.0;
	auto preserve_insertion_order = Settings::Get<PreserveInsertionOrderSetting>(context);
	CostParameters parameters;
	parameters.threads =
	    MaxValue<idx_t>(NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads()), 1);
	auto threads = static_cast<double>(parameters.threads);
	reference<unique_ptr<LogicalOperator>> slot = plan;
	bool executes = true;
	while (slot.get()->type == LogicalOperatorType::LOGICAL_EXPLAIN && !slot.get()->children.empty()) {
		if (slot.get()->Cast<LogicalExplain>().explain_type != ExplainType::EXPLAIN_ANALYZE) {
			executes = false;
		}
		slot = slot.get()->children[0];
	}
	auto &query = *slot.get();
	idx_t now = ++state.queries_seen;
	if (ChangesState(query)) {
		recycler.Invalidate();
		Log(context, StringUtil::Format("query %llu changes the database - recycler cache emptied", now));
		return;
	}
	plan->ResolveOperatorTypes();
	// Key every subtree we could recycle (see ComputeKeys).
	PlanAnalysis analysis(context, preserve_insertion_order, parameters);
	AnalyzeOperator(slot.get(), true, !subtrees_enabled, RootOrderMatters(query, preserve_insertion_order), analysis);
	ComputeKeys(analysis);
	auto &nodes = analysis.nodes;
	auto correction = cost_model.Correction();
	for (idx_t i = nodes.size(); i-- > 0;) {
		auto &node = nodes[i];
		if (!node.valid) {
			continue;
		}
		vector<hash_t> children;
		for (idx_t c = i + 1; c < node.end; c = nodes[c].end) {
			if (nodes[c].valid) {
				children.push_back(nodes[c].fingerprint);
			}
		}
		if (!executes) {
			node.cached = recycler.Peek(node.fingerprint, node.key);
			continue;
		}
		auto match = recycler.MatchOrInsert(node.fingerprint, node.key, children, now);
		if (!match.usable) {
			node.valid = false;
			node.capturable = false;
			continue;
		}
		node.existed = match.existed;
		node.cached = match.cached;
	}
	auto &root = nodes[0];
	if (executes && root.valid && timer->unoptimized_valid) {
		recycler.SetAlias(timer->unoptimized_fingerprint, timer->unoptimized_key, root.fingerprint);
	}
	string reused;
	if (root.valid && CanReuse(root.cached, query, root)) {
		auto rows = root.cached->data->Count();
		if (executes) {
			recycler.Reference(root.fingerprint, now);
		}
		*root.slot =
		    make_uniq<LogicalCachedScan>(root.cached, input.optimizer.binder.GenerateTableIndex(), root.order_matters);
		state.whole_query_hits++;
		Log(context, StringUtil::Format("query %llu answered from the recycler cache (%llu rows)", now, rows));
		return;
	}
	for (idx_t i = 1; i < nodes.size(); i++) {
		auto &node = nodes[i];
		if (node.replaced || !node.valid) {
			continue;
		}
		auto &op = **node.slot;
		if (!CanReuse(node.cached, op, node)) {
			continue;
		}
		auto bindings = op.GetColumnBindings();
		if (bindings.size() != op.types.size()) {
			continue;
		}
		auto rows = node.cached->data->Count();
		reused += StringUtil::Format(" %s (%llu rows)", op.GetName(), rows);
		node.reused = true;
		for (idx_t j = i + 1; j < node.end; j++) {
			nodes[j].replaced = true;
		}
		CostParameters p;
		auto read_cost = static_cast<double>(rows) * p.cached_scan_row +
		                 static_cast<double>(node.cached->bytes) * p.cached_scan_byte;
		for (idx_t j = 0; j < i; j++) {
			if (nodes[j].end > i) {
				nodes[j].cost = MaxValue(nodes[j].cost - node.cost + read_cost, 0.0);
			}
		}
		auto table_index = input.optimizer.binder.GenerateTableIndex();
		*node.slot = make_uniq<LogicalCachedScan>(node.cached, table_index, node.order_matters);
		RemapColumnBindings(*plan, bindings, table_index);
		state.subtree_hits++;
	}
	if (!executes) {
		return;
	}
	std::unordered_map<hash_t, bool> referenced;
	for (auto &node : nodes) {
		if (!node.valid || node.replaced || !node.existed || referenced.count(node.fingerprint)) {
			continue;
		}
		referenced[node.fingerprint] = true;
		recycler.Reference(node.fingerprint, now);
	}
	// Estimated base costs (wall ns) for the nodes this query computes, for nodes not measured yet, and to share...
	auto estimated_bcost = [&](const PlanNode &node) {
		return node.cost / threads * correction;
	};
	auto max_bytes = recycler.MaxEntryBytes();
	// An aggregation's result is compact and already built in a hash table: copying even a large one costs little...
	auto row_limit = [&](const PlanNode &node) {
		if (node.is_root) {
			return root_row_limit;
		}
		auto type = node.slot->get()->type;
		if (type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY || type == LogicalOperatorType::LOGICAL_DISTINCT) {
			return MaxValue(aggregate_row_limit, subtree_row_limit);
		}
		return subtree_row_limit;
	};
	auto eligible = [&](const PlanNode &node) {
		if (node.replaced || node.reused || !node.capturable || node.cached) {
			return false;
		}
		return true;
	};
	// Assesses a candidate.
	auto assess = [&](PlanNode &node, bool speculative) {
		node.mode = ChooseCaptureMode(nodes, static_cast<idx_t>(&node - nodes.data()));
		node.assessment = recycler.Assess(node.fingerprint, estimated_bcost(node), node.estimated_bytes, speculative,
		                                  now);
		auto &a = node.assessment;
		if (a.oversized) {
			return false;
		}
		if (speculative && !a.size_measured) {
			return true;
		}
		auto rows = a.size_measured ? a.measured_rows : node.rows;
		return rows <= row_limit(node) && a.size <= static_cast<double>(max_bytes);
	};
	struct Placement {
		idx_t index;
		bool speculative;
	};
	vector<Placement> stores;
	idx_t subtree_stores = 0;
	auto add_store = [&](idx_t index, bool speculative) {
		if (!nodes[index].is_root) {
			if (subtree_stores >= max_subtree_stores) {
				return;
			}
			subtree_stores++;
		}
		stores.push_back(Placement {index, speculative});
	};
	if (benefit_policies) {
		for (idx_t i = 0; i < nodes.size(); i++) {
			auto &top = nodes[i];
			if (!top.valid || top.replaced || !top.existed) {
				continue;
			}
			if (top.parent != DConstants::INVALID_INDEX && nodes[top.parent].valid && nodes[top.parent].existed) {
				continue;
			}
			idx_t best = DConstants::INVALID_INDEX;
			for (idx_t j = i; j < top.end; j++) {
				auto &candidate = nodes[j];
				if (!candidate.valid || !candidate.existed || !eligible(candidate) || !assess(candidate, false)) {
					continue;
				}
				if (best == DConstants::INVALID_INDEX ||
				    candidate.assessment.benefit > nodes[best].assessment.benefit) {
					best = j;
				}
			}
			if (best == DConstants::INVALID_INDEX || nodes[best].assessment.benefit <= 0) {
				continue;
			}
			if (!recycler.WouldAdmit(nodes[best].assessment.size, nodes[best].assessment.benefit, now)) {
				state.not_admitted++;
				continue;
			}
			add_store(best, false);
		}
		for (idx_t i = 0; i < nodes.size(); i++) {
			auto &node = nodes[i];
			if (!node.valid || node.existed || !IsSpeculationPoint(node, estimated_bcost(node), expensive_join_cost) ||
			    !eligible(node) || !assess(node, true)) {
				continue;
			}
			if (node.mode == CaptureMode::MATERIALIZING && !node.is_root) {
				continue;
			}
			add_store(i, true);
		}
	} else {
		vector<idx_t> candidates;
		for (idx_t i = 1; i < nodes.size(); i++) {
			if (eligible(nodes[i]) && assess(nodes[i], false)) {
				candidates.push_back(i);
			}
		}
		std::stable_sort(candidates.begin(), candidates.end(),
		                 [&](idx_t a, idx_t b) { return nodes[a].work > nodes[b].work; });
		std::unordered_map<hash_t, bool> taken;
		for (auto index : candidates) {
			if (!taken[nodes[index].fingerprint]) {
				taken[nodes[index].fingerprint] = true;
				add_store(index, false);
			}
		}
		if (eligible(root) && assess(root, false)) {
			add_store(0, false);
		}
	}
	vector<pair<hash_t, double>> computed;
	for (auto &node : nodes) {
		if (node.valid && !node.replaced && !node.reused) {
			computed.emplace_back(node.fingerprint, node.base_cost / threads);
		}
	}
	if (root.valid) {
		timer->Arm(root.cost / threads, std::move(computed));
	}
	// 7.
	std::sort(stores.begin(), stores.end(), [](const Placement &a, const Placement &b) { return a.index > b.index; });
	string placed;
	for (auto &store : stores) {
		auto &node = nodes[store.index];
		StoreDecision decision;
		decision.fingerprint = node.fingerprint;
		decision.speculative = store.speculative;
		decision.force = !benefit_policies;
		decision.estimated_bcost = estimated_bcost(node);
		decision.now = now;
		decision.max_bytes = max_bytes;
		decision.max_rows = row_limit(node);
		placed += " " + Describe(node, store.speculative ? "speculative" : "history");
		WrapWithStore(node, state, decision);
		if (node.is_root) {
			state.whole_query_stores++;
		} else {
			state.subtree_stores++;
		}
		if (store.speculative) {
			state.speculative_stores++;
		} else {
			state.history_stores++;
		}
	}
	auto message = StringUtil::Format("query %llu", now);
	if (!reused.empty()) {
		message += " reuses" + reused + ";";
	}
	message += placed.empty() ? " places no store operator" : " stores" + placed;
	Log(context, message);
}

OptimizerExtension CreateRecyclerOptimizer(shared_ptr<RecyclerState> state) {
	OptimizerExtension extension;
	extension.pre_optimize_function = RecycleBeforeOptimizing;
	extension.optimize_function = RecycleResults;
	extension.optimizer_info = std::move(state);
	return extension;
}
}

