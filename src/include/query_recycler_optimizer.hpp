#pragma once

#include "duckdb/optimizer/optimizer_extension.hpp"

#include "query_recycler_cache.hpp"

namespace duckdb {

struct QueryRecyclerOptimizerInfo : public OptimizerExtensionInfo {
	explicit QueryRecyclerOptimizerInfo(shared_ptr<RecyclingCache> cache_p)
	    : cache(std::move(cache_p)) {
	}

	shared_ptr<RecyclingCache> cache;
};

OptimizerExtension CreateQueryRecyclerOptimizer(
    shared_ptr<RecyclingCache> cache);

} // namespace duckdb