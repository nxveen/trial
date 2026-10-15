#pragma once
#include "duckdb/planner/logical_operator.hpp"
namespace duckdb {
class ClientContext;
// Moves decorrelated EXISTS / NOT EXISTS / IN subqueries down to the part of the query they depend on.
idx_t PushDownCorrelatedSubqueries(ClientContext &context, unique_ptr<LogicalOperator> &plan, bool semi_joins);
// Turns WHERE [NOT] EXISTS subqueries correlated by a single equality into plain SEMI / ANTI joins, which the...
idx_t DecorrelateExistsSubqueries(unique_ptr<LogicalOperator> &plan);
}

