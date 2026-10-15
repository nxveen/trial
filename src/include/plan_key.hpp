#pragma once
#include "duckdb/planner/logical_operator.hpp"
namespace duckdb {
// Serializes a plan into bytes that describe it completely, in a canonical form, and returns false if DuckDB...
bool SerializePlan(const LogicalOperator &op, string &result);
// Hash of a serialized plan, used to find candidates.
hash_t HashPlanKey(const string &key);
// The keys SerializePlan would give the plan and each of its subtrees, in preorder, from a single serialization...
bool SerializeSubtrees(const LogicalOperator &root, vector<string> &keys);
}

