#include "subsumption.hpp"
#include "cache_operators.hpp"
#include "plan_key.hpp"
#include "recycler.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include <algorithm>
namespace duckdb {
namespace {
struct Bound {
	LogicalOperator *filter;
	ColumnBinding binding;
	LogicalType type;
	bool upper;
	BoundConstantExpression *constant;
};

BoundConstantExpression *ConstantBelowCasts(Expression &expr) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		return &expr.Cast<BoundConstantExpression>();
	}
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_CAST) {
		return ConstantBelowCasts(*expr.Cast<BoundCastExpression>().child);
	}
	return nullptr;
}

// Every `column >= constant` and `column <= constant` in the plan's filters, in a fixed order.
void CollectBounds(LogicalOperator &op, vector<Bound> &out) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		for (auto &expr : op.expressions) {
			auto type = expr->GetExpressionType();
			if (type != ExpressionType::COMPARE_GREATERTHANOREQUALTO && type != ExpressionType::COMPARE_LESSTHANOREQUALTO) {
				continue;
			}
			auto &comparison = expr->Cast<BoundComparisonExpression>();
			auto constant = ConstantBelowCasts(*comparison.right);
			if (comparison.left->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF && constant) {
				out.push_back(Bound {&op, comparison.left->Cast<BoundColumnRefExpression>().binding,
				                     comparison.left->return_type, type == ExpressionType::COMPARE_LESSTHANOREQUALTO,
				                     constant});
			}
		}
	}
	for (auto &child : op.children) {
		CollectBounds(*child, out);
	}
}

bool FindPath(LogicalOperator &from, LogicalOperator *target, vector<pair<LogicalOperator *, idx_t>> &path) {
	if (&from == target) {
		return true;
	}
	for (idx_t i = 0; i < from.children.size(); i++) {
		path.emplace_back(&from, i);
		if (FindPath(*from.children[i], target, path)) {
			return true;
		}
		path.pop_back();
	}
	return false;
}

bool IsColumn(const Expression &expr, const ColumnBinding &binding) {
	return expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
	       expr.Cast<BoundColumnRefExpression>().binding == binding;
}

// Follows `binding` up through `op`, if filtering on it commutes with `op`.
bool PassUp(LogicalOperator &op, ColumnBinding &binding) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		break;
	case LogicalOperatorType::LOGICAL_DISTINCT:
		if (op.Cast<LogicalDistinct>().distinct_type != DistinctType::DISTINCT) {
			return false;
		}
		break;
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		auto &projection = op.Cast<LogicalProjection>();
		for (idx_t i = 0; i < projection.expressions.size(); i++) {
			if (IsColumn(*projection.expressions[i], binding)) {
				binding = ColumnBinding(projection.table_index, i);
				return true;
			}
		}
		return false;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggregate = op.Cast<LogicalAggregate>();
		if (aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty()) {
			return false;
		}
		for (idx_t i = 0; i < aggregate.groups.size(); i++) {
			if (IsColumn(*aggregate.groups[i], binding)) {
				binding = ColumnBinding(aggregate.group_index, i);
				return true;
			}
		}
		return false;
	}
	default:
		return false;
	}
	auto bindings = op.GetColumnBindings();
	return std::find(bindings.begin(), bindings.end(), binding) != bindings.end();
}
}

bool ComputeRangeSignature(LogicalOperator &plan, RangeSignature &signature) {
	vector<Bound> bounds;
	CollectBounds(plan, bounds);
	if (bounds.empty()) {
		return false;
	}
	signature.constants.clear();
	for (auto &bound : bounds) {
		signature.constants.push_back(bound.constant->value);
		bound.constant->value = Value(bound.constant->value.type());
	}
	auto serialized = SerializePlan(plan, signature.key);
	for (idx_t i = 0; i < bounds.size(); i++) {
		bounds[i].constant->value = signature.constants[i];
	}
	if (!serialized) {
		return false;
	}
	signature.fingerprint = HashPlanKey(signature.key);
	return true;
}

bool TryRangeSubsumption(Binder &binder, unique_ptr<LogicalOperator> &slot, const RangeSignature &current,
                         const vector<Value> &previous, const shared_ptr<CachedResult> &cached, bool order_matters) {
	if (!cached || (order_matters && !cached->ordered)) {
		return false;
	}
	vector<Bound> bounds;
	CollectBounds(*slot, bounds);
	if (current.constants.size() != bounds.size() || previous.size() != bounds.size()) {
		return false;
	}
	idx_t changed = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < bounds.size(); i++) {
		if (previous[i].type() != current.constants[i].type() || previous[i].IsNull() || current.constants[i].IsNull()) {
			return false;
		}
		if (previous[i] == current.constants[i]) {
			continue;
		}
		if (changed != DConstants::INVALID_INDEX &&
		    (bounds[i].filter != bounds[changed].filter || bounds[i].binding != bounds[changed].binding)) {
			return false;
		}
		changed = i;
	}
	if (changed == DConstants::INVALID_INDEX) {
		return false;
	}
	idx_t low = DConstants::INVALID_INDEX, high = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < bounds.size(); i++) {
		if (bounds[i].filter != bounds[changed].filter || bounds[i].binding != bounds[changed].binding) {
			continue;
		}
		auto &side = bounds[i].upper ? high : low;
		if (side != DConstants::INVALID_INDEX) {
			return false;
		}
		side = i;
	}
	if (low == DConstants::INVALID_INDEX || high == DConstants::INVALID_INDEX) {
		return false;
	}
	auto &type = bounds[low].type;
	Value new_low, new_high, old_low, old_high;
	try {
		new_low = current.constants[low].DefaultCastAs(type);
		new_high = current.constants[high].DefaultCastAs(type);
		old_low = previous[low].DefaultCastAs(type);
		old_high = previous[high].DefaultCastAs(type);
	} catch (std::exception &) {
		return false;
	}
	if (new_low < old_low || new_high > old_high) {
		return false;
	}
	auto binding = bounds[low].binding;
	auto filter_op = bounds[low].filter;
	vector<pair<LogicalOperator *, idx_t>> path;
	if (!FindPath(*slot, filter_op, path)) {
		return false;
	}
	for (idx_t i = path.size(); i-- > 0;) {
		if (!PassUp(*path[i].first, binding)) {
			return false;
		}
	}
	slot->ResolveOperatorTypes();
	auto bindings = slot->GetColumnBindings();
	auto column = std::find(bindings.begin(), bindings.end(), binding);
	if (column == bindings.end() || bindings.size() != slot->types.size() || cached->types != slot->types) {
		return false;
	}
	auto index = static_cast<idx_t>(column - bindings.begin());
	auto scan_index = binder.GenerateTableIndex();
	auto filter = make_uniq<LogicalFilter>();
	for (bool upper : {false, true}) {
		filter->expressions.push_back(make_uniq<BoundComparisonExpression>(
		    upper ? ExpressionType::COMPARE_LESSTHANOREQUALTO : ExpressionType::COMPARE_GREATERTHANOREQUALTO,
		    make_uniq<BoundColumnRefExpression>(slot->types[index], ColumnBinding(scan_index, index)),
		    make_uniq<BoundConstantExpression>(upper ? new_high : new_low)));
	}
	filter->children.push_back(make_uniq<LogicalCachedScan>(cached, scan_index, order_matters));
	filter->ResolveOperatorTypes();
	slot = std::move(filter);
	return true;
}
}

