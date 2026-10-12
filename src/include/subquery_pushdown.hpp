#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class ClientContext;

//! Moves decorrelated EXISTS / NOT EXISTS / IN subqueries down to the part of
//! the query they depend on.
//!
//! DuckDB decorrelates a correlated subquery into a delim join whose left
//! side is the whole FROM clause, so the subquery is evaluated only after every
//! join of the outer query - for `... JOIN title t ... WHERE EXISTS (... WHERE
//! x.movie_id = t.id)` on the full multi-way join result, which can be millions
//! of rows, although its outcome depends on t alone. This pass pushes such a
//! delim join down through the inner joins, cross products and LEFT joins (on
//! their preserved side) of its left side to the smallest subtree that
//! provides every column it correlates on.
//!
//! Correct because a MARK, SEMI or ANTI join's outcome for a row depends only
//! on the columns it correlates on: inner joins and cross products only drop or
//! duplicate rows, a LEFT join keeps every row of its left side, and all of them
//! pass every column through by binding, so applying the
//! subquery before them yields the same rows, and the mark column (for EXISTS
//! used in an expression) still reaches everything that referenced it.
//!
//! Runs before the optimizer, which then plans the joins around the smaller
//! input. Returns the number of subqueries moved.
idx_t PushDownCorrelatedSubqueries(ClientContext &context, unique_ptr<LogicalOperator> &plan);

//! Turns WHERE [NOT] EXISTS subqueries correlated by a single equality into
//! plain SEMI / ANTI joins, which the optimizer can place and order freely,
//! instead of delim joins it cannot. Runs before PushDownCorrelatedSubqueries,
//! which then handles the subqueries this one leaves as delim joins. Returns
//! the number rewritten.
idx_t DecorrelateExistsSubqueries(unique_ptr<LogicalOperator> &plan);

} // namespace duckdb

