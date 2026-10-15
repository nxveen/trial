#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! Serializes a plan into bytes that describe it completely, in a canonical
//! form, and returns false if DuckDB cannot serialize some part of it.
//!
//! We deliberately do NOT key the cache on the plan's printed form. Several
//! operators leave fields out of EXPLAIN: LogicalLimit and LogicalTopN do not
//! print their limit or offset at all, so "LIMIT 10000 OFFSET 0" and
//! "LIMIT 10000 OFFSET 10000" print identically and would share a cache entry.
//!
//! Plain serialization is not enough either, because it records details that
//! say nothing about the result:
//!  - Table indices. The binder numbers every table, projection and aggregate
//!    in the order it meets them, so the same subtree gets different numbers
//!    in two queries that bind other tables before it. We renumber them in
//!    the order the serializer meets them, which keeps every reference
//!    consistent (a binding is just a pair of these numbers) while making two
//!    plans that differ only in numbering produce identical bytes.
//!  - Query locations: the character offset an expression had in the SQL text.
//! Two plans with identical canonical bytes are the same plan up to a
//! renaming of table indices, so they produce the same rows in the same
//! column order.
bool SerializePlan(const LogicalOperator &op, string &result);

//! Hash of a serialized plan, used to find candidates. The bytes themselves
//! still decide whether a candidate is a match.
hash_t HashPlanKey(const string &key);

//! The keys SerializePlan would give the plan and each of its subtrees, in
//! preorder, from a single serialization of the plan - instead of serializing
//! every subtree on its own, which writes a deep subtree once per ancestor.
//! Returns false if the plan cannot be serialized as a whole.
bool SerializeSubtrees(const LogicalOperator &root, vector<string> &keys);

} // namespace duckdb

