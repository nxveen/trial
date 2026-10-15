#pragma once
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/logical_operator.hpp"
namespace duckdb {
class Binder;
struct CachedResult;
struct RangeSignature {
	string key;
	hash_t fingerprint = 0;
	vector<Value> constants;
};

// An unoptimized plan's key with the bounds of its BETWEEN filters masked, plus those bounds in order.
bool ComputeRangeSignature(LogicalOperator &plan, RangeSignature &signature);

// Replaces the plan at `slot` by `cached` filtered to its range, if that range lies inside the cached one's.
bool TryRangeSubsumption(Binder &binder, unique_ptr<LogicalOperator> &slot, const RangeSignature &current,
                         const vector<Value> &previous, const shared_ptr<CachedResult> &cached, bool order_matters);
}

