#pragma once
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/logical_operator.hpp"
namespace duckdb {
// Points every column reference in `root` that used to come out of a replaced subtree at the cached scan that...
void RemapColumnBindings(LogicalOperator &root, const vector<ColumnBinding> &old_bindings, idx_t new_table_index);
}

