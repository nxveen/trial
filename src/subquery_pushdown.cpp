#include "subquery_pushdown.hpp"

#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

#include <algorithm>
#include <functional>

namespace duckdb {

static void CollectBindings(const Expression &expr, vector<ColumnBinding> &out) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		out.push_back(expr.Cast<BoundColumnRefExpression>().binding);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { CollectBindings(child, out); });
}

//! True if `op` produces every one of `needed`.
static bool Provides(LogicalOperator &op, const vector<ColumnBinding> &needed) {
	auto bindings = op.GetColumnBindings();
	for (auto &binding : needed) {
		if (std::find(bindings.begin(), bindings.end(), binding) == bindings.end()) {
			return false;
		}
	}
	return true;
}

//! The inputs of `op` the subquery may be moved into: inputs whose every row
//! and column `op` passes on unchanged (under the same bindings), only
//! dropping, duplicating or extending rows. Both inputs of an inner join or
//! cross product; only the preserved (left) input of a LEFT join, whose rows
//! all survive it - its right input is padded with NULLs, so it is not.
static vector<idx_t> PassThroughInputs(LogicalOperator &op) {
	if (op.children.size() != 2) {
		return {};
	}
	if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		return {0, 1};
	}
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.HasProjectionMap()) {
			return {};
		}
		if (join.join_type == JoinType::INNER) {
			return {0, 1};
		}
		if (join.join_type == JoinType::LEFT) {
			return {0};
		}
	}
	return {};
}

//! The row count of the largest base table scanned in a subtree: a rough,
//! estimate-free measure of how many rows it can produce. (Estimates built from
//! guessed predicate selectivities were tried and chose worse on this workload.)
static idx_t LargestTable(ClientContext &context, LogicalOperator &op) {
	idx_t largest = 0;
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.cardinality) {
			auto stats = get.function.cardinality(context, get.bind_data.get());
			if (stats && stats->has_estimated_cardinality) {
				largest = stats->estimated_cardinality;
			}
		}
	}
	for (auto &child : op.children) {
		largest = MaxValue(largest, LargestTable(context, *child));
	}
	return largest;
}

static bool TryPushDown(ClientContext &context, unique_ptr<LogicalOperator> &slot) {
	auto &op = *slot;
	if (op.type != LogicalOperatorType::LOGICAL_DELIM_JOIN || op.children.size() != 2) {
		return false;
	}
	auto &delim = op.Cast<LogicalComparisonJoin>();
	if (delim.delim_flipped || delim.HasProjectionMap()) {
		return false;
	}
	switch (delim.join_type) {
	case JoinType::MARK:
	case JoinType::SEMI:
	case JoinType::ANTI:
		break;
	default:
		return false;
	}
	// Every column of the outer query the subquery depends on.
	vector<ColumnBinding> needed;
	for (auto &column : delim.duplicate_eliminated_columns) {
		CollectBindings(*column, needed);
	}
	for (auto &condition : delim.conditions) {
		CollectBindings(*condition.left, needed);
	}
	if (needed.empty()) {
		return false;
	}
	// Walk down the left side while one input of a pass-through join still
	// provides all of them.
	unique_ptr<LogicalOperator> *target = &delim.children[0];
	while (true) {
		unique_ptr<LogicalOperator> *next = nullptr;
		for (auto input : PassThroughInputs(**target)) {
			auto &child = (*target)->children[input];
			if (Provides(*child, needed)) {
				next = &child;
				break;
			}
		}
		if (!next) {
			break;
		}
		target = next;
	}
	if (target == &delim.children[0]) {
		return false;
	}
	// Only worth it if the subquery then sees fewer rows: never move it onto
	// the largest table of the outer query.
	if (LargestTable(context, **target) >= LargestTable(context, *delim.children[0])) {
		return false;
	}
	auto outer = std::move(delim.children[0]);
	delim.children[0] = std::move(*target);
	*target = std::move(slot);
	slot = std::move(outer);
	return true;
}

static idx_t PushDownRecursive(ClientContext &context, unique_ptr<LogicalOperator> &slot) {
	idx_t moved = 0;
	for (auto &child : slot->children) {
		moved += PushDownRecursive(context, child);
	}
	if (TryPushDown(context, slot)) {
		moved++;
	}
	return moved;
}

idx_t PushDownCorrelatedSubqueries(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
	return PushDownRecursive(context, plan);
}


//===--------------------------------------------------------------------===//
// EXISTS / NOT EXISTS on an equality: semi and anti joins
//===--------------------------------------------------------------------===//

//! Counts references to `binding` in the expressions of a whole plan.
static idx_t CountReferences(LogicalOperator &op, const ColumnBinding &binding) {
	idx_t count = 0;
	std::function<void(const Expression &)> visit = [&](const Expression &expr) {
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
		    expr.Cast<BoundColumnRefExpression>().binding == binding) {
			count++;
		}
		ExpressionIterator::EnumerateChildren(expr, visit);
	};
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) { visit(**child); });
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN || op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// Conditions and duplicate-eliminated columns are enumerated above.
	}
	for (auto &child : op.children) {
		count += CountReferences(*child, binding);
	}
	return count;
}

static bool References(const Expression &expr, idx_t table_index) {
	vector<ColumnBinding> bindings;
	CollectBindings(expr, bindings);
	for (auto &binding : bindings) {
		if (binding.table_index == table_index) {
			return true;
		}
	}
	return false;
}

static bool ContainsDelimGet(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		return true;
	}
	for (auto &child : op.children) {
		if (ContainsDelimGet(*child)) {
			return true;
		}
	}
	return false;
}

//! Rewrites `delim_slot` - a decorrelated EXISTS (a MARK delim join) whose
//! mark is used as `mark` or `NOT mark` in `filter` - into a plain SEMI or ANTI
//! join. Only the shape DuckDB produces for an EXISTS correlated by a single
//! equality is accepted:
//!     DELIM_JOIN MARK  [outer.col IS NOT DISTINCT FROM P.k]
//!       outer
//!       P: PROJECTION (..., k: delim.col, ...)
//!            FILTER (inner.col = delim.col AND rest)
//!              CROSS_PRODUCT (inner, DELIM_GET)
//! and becomes
//!     SEMI / ANTI JOIN  [outer.col = P.k]
//!       outer
//!       P: PROJECTION (..., k: inner.col, ...)
//!            FILTER (rest)
//!              inner
//! EXISTS (... WHERE inner.col = outer.col AND rest) is exactly "some row of
//! inner satisfying rest has inner.col = outer.col": a semi join on equality,
//! which, like EXISTS, never matches a NULL; NOT EXISTS is its anti join, which
//! keeps a row with a NULL key just as NOT EXISTS does. The query optimizer can
//! then place and order these joins itself, which it cannot do with a delim
//! join - and the delim join's duplicate elimination disappears.
static bool TryDecorrelate(LogicalOperator &root, LogicalOperator &filter, unique_ptr<LogicalOperator> &delim_slot) {
	auto &delim = delim_slot->Cast<LogicalComparisonJoin>();
	if (delim.join_type != JoinType::MARK || delim.delim_flipped || delim.HasProjectionMap() ||
	    delim.children.size() != 2 || delim.conditions.size() != 1 || delim.duplicate_eliminated_columns.size() != 1 ||
	    delim.conditions[0].comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
		return false;
	}
	// The mark must be used exactly once in the whole plan: as `mark` or
	// `NOT mark`, a conjunct of this filter.
	ColumnBinding mark(delim.mark_index, 0);
	if (CountReferences(root, mark) != 1) {
		return false;
	}
	idx_t conjunct = DConstants::INVALID_INDEX;
	JoinType type = JoinType::INVALID;
	for (idx_t i = 0; i < filter.expressions.size(); i++) {
		auto &expr = *filter.expressions[i];
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
		    expr.Cast<BoundColumnRefExpression>().binding == mark) {
			conjunct = i;
			type = JoinType::SEMI;
		} else if (expr.GetExpressionType() == ExpressionType::OPERATOR_NOT) {
			auto &op = expr.Cast<BoundOperatorExpression>();
			if (op.children.size() == 1 && op.children[0]->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
			    op.children[0]->Cast<BoundColumnRefExpression>().binding == mark) {
				conjunct = i;
				type = JoinType::ANTI;
			}
		}
	}
	if (conjunct == DConstants::INVALID_INDEX) {
		return false;
	}
	// The subquery side: PROJECTION -> FILTER -> CROSS_PRODUCT(inner, DELIM_GET).
	auto &projection = *delim.children[1];
	if (projection.type != LogicalOperatorType::LOGICAL_PROJECTION || projection.children.size() != 1) {
		return false;
	}
	auto &inner_filter = *projection.children[0];
	if (inner_filter.type != LogicalOperatorType::LOGICAL_FILTER || inner_filter.children.size() != 1) {
		return false;
	}
	auto &cross_slot = inner_filter.children[0];
	if (cross_slot->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT || cross_slot->children.size() != 2) {
		return false;
	}
	idx_t delim_side = cross_slot->children[1]->type == LogicalOperatorType::LOGICAL_DELIM_GET ? 1 : 0;
	auto &delim_get = *cross_slot->children[delim_side];
	auto &inner = cross_slot->children[1 - delim_side];
	if (delim_get.type != LogicalOperatorType::LOGICAL_DELIM_GET || ContainsDelimGet(*inner)) {
		return false;
	}
	auto delim_index = delim_get.GetTableIndex()[0];
	ColumnBinding delim_column(delim_index, 0);
	// Exactly one conjunct uses the delim values: inner.col = delim.col.
	idx_t equality = DConstants::INVALID_INDEX;
	unique_ptr<Expression> inner_column;
	for (idx_t i = 0; i < inner_filter.expressions.size(); i++) {
		auto &expr = *inner_filter.expressions[i];
		if (!References(expr, delim_index)) {
			continue;
		}
		if (equality != DConstants::INVALID_INDEX || expr.GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
			return false;
		}
		auto &comparison = expr.Cast<BoundComparisonExpression>();
		auto is_delim = [&](const Expression &side) {
			return side.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
			       side.Cast<BoundColumnRefExpression>().binding == delim_column;
		};
		if (is_delim(*comparison.left) && !References(*comparison.right, delim_index)) {
			inner_column = comparison.right->Copy();
		} else if (is_delim(*comparison.right) && !References(*comparison.left, delim_index)) {
			inner_column = comparison.left->Copy();
		} else {
			return false;
		}
		equality = i;
	}
	if (equality == DConstants::INVALID_INDEX) {
		return false;
	}
	// The projection may only pass the delim column through as is.
	for (auto &expr : projection.expressions) {
		if (References(*expr, delim_index) &&
		    !(expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
		      expr->Cast<BoundColumnRefExpression>().binding == delim_column)) {
			return false;
		}
	}
	// The inner column must stand in for the delim column without a cast.
	auto &delim_types = delim_get.Cast<LogicalDelimGet>().chunk_types;
	if (delim_types.empty() || inner_column->return_type != delim_types[0]) {
		return false;
	}

	// All checks passed: rewrite.
	for (auto &expr : projection.expressions) {
		if (References(*expr, delim_index)) {
			expr = inner_column->Copy();
		}
	}
	inner_filter.expressions.erase(inner_filter.expressions.begin() + static_cast<int64_t>(equality));
	cross_slot = std::move(inner);
	if (inner_filter.expressions.empty()) {
		projection.children[0] = std::move(inner_filter.children[0]);
	}
	auto join = make_uniq<LogicalComparisonJoin>(type);
	JoinCondition condition;
	condition.left = std::move(delim.conditions[0].left);
	condition.right = std::move(delim.conditions[0].right);
	condition.comparison = ExpressionType::COMPARE_EQUAL;
	join->conditions.push_back(std::move(condition));
	join->children.push_back(std::move(delim.children[0]));
	join->children.push_back(std::move(delim.children[1]));
	delim_slot = std::move(join);
	filter.expressions.erase(filter.expressions.begin() + static_cast<int64_t>(conjunct));
	return true;
}

static idx_t DecorrelateRecursive(LogicalOperator &root, unique_ptr<LogicalOperator> &slot) {
	idx_t rewritten = 0;
	for (auto &child : slot->children) {
		rewritten += DecorrelateRecursive(root, child);
	}
	if (slot->type != LogicalOperatorType::LOGICAL_FILTER || slot->children.size() != 1) {
		return rewritten;
	}
	// The delim joins of a WHERE clause's subqueries sit in a chain below its
	// FILTER, each on the left side of the next.
	auto &filter = *slot;
	reference<unique_ptr<LogicalOperator>> current = filter.children[0];
	while (current.get()->type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
	       (current.get()->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN && rewritten > 0 &&
	        (current.get()->Cast<LogicalComparisonJoin>().join_type == JoinType::SEMI ||
	         current.get()->Cast<LogicalComparisonJoin>().join_type == JoinType::ANTI))) {
		if (current.get()->type == LogicalOperatorType::LOGICAL_DELIM_JOIN && TryDecorrelate(root, filter, current.get())) {
			rewritten++;
		}
		if (current.get()->children.empty()) {
			break;
		}
		current = current.get()->children[0];
	}
	if (filter.expressions.empty()) {
		slot = std::move(filter.children[0]);
	}
	return rewritten;
}

idx_t DecorrelateExistsSubqueries(unique_ptr<LogicalOperator> &plan) {
	return DecorrelateRecursive(*plan, plan);
}

} // namespace duckdb

