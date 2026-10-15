#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <unordered_map>

namespace duckdb {

//! The recycler of Nagel, Boncz and Viglas, "Recycling in Pipelined Query
//! Evaluation" (ICDE 2013): the recycler graph and the recycler cache
//! (Section II), managed with the benefit metric of Section III-C and the
//! cache policies of Section III-E.
//!
//! Nodes are identified by the canonical serialized plan of their subtree
//! (see plan_key.hpp). Two subtrees with equal keys are bisimilar in the
//! paper's sense - same operator, same parameters, exactly matching children -
//! so looking a subtree up by its key is the paper's matchTree (Algorithm 1)
//! done in one step, and the canonical renumbering of table indices plays the
//! part of its name mapping. Equal subtrees from different queries share one
//! node, so the graph is the paper's AND-DAG of optimized plans.

//! A cached result: the rows of a recycler-graph node.
struct CachedResult {
	vector<LogicalType> types;
	shared_ptr<ColumnDataCollection> data;
	idx_t bytes = 0;
	//! Captured by one thread in plan order; only such a result may be used
	//! where the order of rows is observable.
	bool ordered = false;
};

//! One node of the recycler graph.
struct GraphNode {
	hash_t fingerprint = 0;
	//! The canonical plan bytes: the hash only finds candidates, the bytes
	//! decide the match.
	string key;
	//! Graph children: the nodes of this node's direct child subtrees.
	vector<hash_t> children;
	//! The query that inserted the node. Its sightings in that query do not
	//! count as references (Section III-C: "S that has not been inserted by Q").
	idx_t created_by = 0;

	//! Importance factor h_R (Section III-C), aged lazily: the value at query
	//! `h_time`, to be multiplied by alpha^(now - h_time) when next touched.
	double h = 0;
	idx_t h_time = 0;

	//! Base cost bcost(R): wall time to compute the result from base tables
	//! only (ns). Refreshed every time the result is recomputed. 0 = unknown.
	double bcost = 0;
	//! Measured size once the result has been stored; 0 = unknown.
	idx_t measured_bytes = 0;
	idx_t measured_rows = 0;
	//! The result did not fit the cache when stored: do not try again.
	bool oversized = false;

	//! The result, while it is in the recycler cache.
	shared_ptr<CachedResult> cached;
};

//! What the recycler reports through waddle_stats().
struct RecyclerStats {
	idx_t graph_nodes = 0;
	idx_t cache_entries = 0;
	idx_t cache_bytes = 0;
	idx_t admissions = 0;
	idx_t replacements_rejected = 0;
	idx_t evictions = 0;
};

class Recycler {
public:
	//! The speculative importance of a result never seen before (Section
	//! III-D: "a small constant for h_n.R (e.g., 0.001)").
	static constexpr double SPECULATIVE_H = 0.001;

	//! Cache budget (bytes) and the aging weight alpha < 1 (Eq. 5).
	void Configure(idx_t budget_bytes, double alpha);

	//===--------------------------------------------------------------===//
	// Matching and insertion (Sections III-A and III-B)
	//===--------------------------------------------------------------===//
	struct Match {
		//! False if the fingerprint is taken by a different plan (a hash
		//! collision): the subtree is then left out of recycling.
		bool usable = false;
		//! The node existed before this query: an exact match.
		bool existed = false;
		//! The node's cached result, if any.
		shared_ptr<CachedResult> cached;
	};
	//! Matches one subtree against the graph, inserting it if there is no
	//! exact match. Children must have been matched first (bottom-up).
	Match MatchOrInsert(hash_t fingerprint, const string &key, const vector<hash_t> &children, idx_t now);

	//! The cached result of a subtree, without touching the graph (for plans
	//! that are only explained, never run).
	shared_ptr<CachedResult> Peek(hash_t fingerprint, const string &key);

	//! Records that the unoptimized plan with this key optimizes to the graph
	//! node `root`. A later query with the same unoptimized plan is the same
	//! query, so if `root` is cached it can be answered before the optimizer
	//! runs at all.
	void SetAlias(hash_t fingerprint, const string &key, hash_t root);
	//! The cached result an unoptimized plan's alias leads to, if any; sets
	//! `root` to the graph node it found.
	shared_ptr<CachedResult> LookupAlias(hash_t fingerprint, const string &key, hash_t &root);

	//! h += 1 for a node whose result could have answered the current query
	//! (Section III-C, "Importance Factor"), with lazy aging first.
	void Reference(hash_t fingerprint, idx_t now);

	//===--------------------------------------------------------------===//
	// Measurements (Section III-C, "True Cost and Base Cost")
	//===--------------------------------------------------------------===//
	void SetBaseCost(hash_t fingerprint, double bcost);
	double BaseCost(hash_t fingerprint);

	//===--------------------------------------------------------------===//
	// Benefit (Eq. 1 and 2)
	//===--------------------------------------------------------------===//
	struct Assessment {
		double h = 0;
		double true_cost = 0;
		double size = 0;
		double benefit = 0;
		bool size_measured = false;
		idx_t measured_rows = 0;
		bool oversized = false;
	};
	//! B(R) = cost(R) * h_R / size(R) for one node. `estimated_bcost` and
	//! `estimated_bytes` stand in for measurements the node does not have
	//! yet; `speculative` uses the speculative h instead of the node's.
	Assessment Assess(hash_t fingerprint, double estimated_bcost, double estimated_bytes, bool speculative,
	                  idx_t now);

	//===--------------------------------------------------------------===//
	// The recycler cache (Section III-E)
	//===--------------------------------------------------------------===//
	//! The admission/replacement decision, without changing anything: would a
	//! result of this size and benefit get into the cache now?
	bool WouldAdmit(double bytes, double benefit, idx_t now);
	//! Stores a captured result if the admission or replacement policy lets
	//! it in, applying updateHR (Algorithm 2) to the node's descendants.
	//! `estimated_bcost` is used if the node has no measured base cost yet.
	bool Store(hash_t fingerprint, shared_ptr<CachedResult> result, double estimated_bcost, bool speculative,
	           bool force, idx_t now);
	//! Records that a result outgrew the cache while being captured.
	void MarkOversized(hash_t fingerprint);
	//! Evicts every cached result (the database changed).
	void Invalidate();
	//! Forgets everything, the graph included.
	void Reset();

	idx_t MaxEntryBytes();
	RecyclerStats GetStats();

private:
	//! All private helpers expect `lock` to be held.
	GraphNode *Find(hash_t fingerprint);
	double AgedH(GraphNode &node, idx_t now) const;
	void SetH(GraphNode &node, double h, idx_t now);
	//! Algorithm 2 (with sign -1 on admission, Eq. 3) and its counterpart on
	//! eviction (sign +1, Eq. 4): walk the descendants down to and including
	//! the nearest cached ones (DMDs and potential DMDs).
	void UpdateHR(GraphNode &node, double h_delta, idx_t now);
	//! cost(R) = bcost(R) - sum of bcost over the direct materialized descendants.
	double TrueCost(GraphNode &node, double bcost);
	void SubtractDmdCosts(GraphNode &node, double &cost);
	double Benefit(GraphNode &node, idx_t now);
	static idx_t SizeGroup(double bytes);
	//! The replacement policy: the lowest-benefit set of cached results in the
	//! newcomer's size group that frees enough room while its average benefit
	//! stays below the newcomer's. False if there is none.
	bool FindVictims(double bytes, double benefit, idx_t now, bool force, vector<hash_t> &victims);
	void Evict(GraphNode &node, idx_t now);

	struct Alias {
		string key;
		hash_t root;
	};

	mutex lock;
	std::unordered_map<hash_t, GraphNode> graph;
	std::unordered_map<hash_t, Alias> aliases;
	idx_t budget = 1024ULL * 1024 * 1024;
	double alpha = 0.99;
	idx_t bytes_used = 0;
	idx_t latest = 0;
	RecyclerStats stats;
};

} // namespace duckdb

