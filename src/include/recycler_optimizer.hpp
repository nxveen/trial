#pragma once

#include "cost_model.hpp"
#include "recycler.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {

//! State shared by every query on this database.
struct RecyclerState : public OptimizerExtensionInfo {
	shared_ptr<Recycler> recycler = make_shared_ptr<Recycler>();
	shared_ptr<CostModel> cost_model = make_shared_ptr<CostModel>();

	//! What the rewriter decided, per query. Store operators are counted when
	//! placed; the recycler counts what was actually admitted.
	atomic<idx_t> queries_seen {0};
	atomic<idx_t> whole_query_hits {0};
	atomic<idx_t> subtree_hits {0};
	atomic<idx_t> whole_query_stores {0};
	atomic<idx_t> subtree_stores {0};
	atomic<idx_t> history_stores {0};
	atomic<idx_t> speculative_stores {0};
	//! History candidates the admission/replacement policy turned down at
	//! rewrite time.
	atomic<idx_t> not_admitted {0};
	//! Correlated subqueries moved down to what they depend on.
	atomic<idx_t> subqueries_pushed {0};
	//! EXISTS / NOT EXISTS subqueries turned into semi / anti joins.
	atomic<idx_t> subqueries_decorrelated {0};

	//! Empties the cache, forgets the recycler graph and zeroes every counter.
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
	}
};

//! Optimizer extension implementing the rewriter rules of the recycler.
OptimizerExtension CreateRecyclerOptimizer(shared_ptr<RecyclerState> state);

} // namespace duckdb

