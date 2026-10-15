#pragma once
#include "cost_model.hpp"
#include "recycler.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
namespace duckdb {
struct RecyclerState : public OptimizerExtensionInfo {
	shared_ptr<Recycler> recycler = make_shared_ptr<Recycler>();
	shared_ptr<CostModel> cost_model = make_shared_ptr<CostModel>();
	atomic<idx_t> queries_seen {0};
	atomic<idx_t> whole_query_hits {0};
	atomic<idx_t> subtree_hits {0};
	atomic<idx_t> whole_query_stores {0};
	atomic<idx_t> subtree_stores {0};
	atomic<idx_t> history_stores {0};
	atomic<idx_t> speculative_stores {0};
	atomic<idx_t> not_admitted {0};
	atomic<idx_t> subqueries_pushed {0};
	atomic<idx_t> subqueries_decorrelated {0};
	atomic<idx_t> subsumption_hits {0};
	// Empties the cache, forgets the recycler graph and zeroes every counter.
	void Reset() {
		recycler->Reset();
		cost_model->Reset();
		queries_seen = 0;
		whole_query_hits = 0;
		subtree_hits = 0;
		whole_query_stores = 0;
		subtree_stores = 0;
		history_stores = 0;
		speculative_stores = 0;
		not_admitted = 0;
		subqueries_pushed = 0;
		subqueries_decorrelated = 0;
		subsumption_hits = 0;
	}
};

// Optimizer extension implementing the rewriter rules of the recycler.
OptimizerExtension CreateRecyclerOptimizer(shared_ptr<RecyclerState> state);
}

