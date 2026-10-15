#pragma once
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include <unordered_map>
namespace duckdb {
struct CachedResult {
	vector<LogicalType> types;
	shared_ptr<ColumnDataCollection> data;
	idx_t bytes = 0;
	bool ordered = false;
};

struct GraphNode {
	hash_t fingerprint = 0;
	string key;
	vector<hash_t> children;
	idx_t created_by = 0;
	double h = 0;
	idx_t h_time = 0;
	double bcost = 0;
	idx_t measured_bytes = 0;
	idx_t measured_rows = 0;
	bool oversized = false;
	shared_ptr<CachedResult> cached;
};

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
	static constexpr double SPECULATIVE_H = 0.001;
	// Cache budget (bytes) and the aging weight alpha < 1 (Eq.
	void Configure(idx_t budget_bytes, double alpha);
	struct Match {
		bool usable = false;
		bool existed = false;
		shared_ptr<CachedResult> cached;
	};

	// Matches one subtree against the graph, inserting it if there is no exact match.
	Match MatchOrInsert(hash_t fingerprint, const string &key, const vector<hash_t> &children, idx_t now);
	// The cached result of a subtree, without touching the graph (for plans that are only explained, never run).
	shared_ptr<CachedResult> Peek(hash_t fingerprint, const string &key);
	// Records that the unoptimized plan with this key optimizes to the graph node `root`.
	void SetAlias(hash_t fingerprint, const string &key, hash_t root);
	// The cached result an unoptimized plan's alias leads to, if any; sets `root` to the graph node it found.
	shared_ptr<CachedResult> LookupAlias(hash_t fingerprint, const string &key, hash_t &root);
	// h += 1 for a node whose result could have answered the current query (Section III-C, "Importance Factor")...
	void Reference(hash_t fingerprint, idx_t now);
	void SetBaseCost(hash_t fingerprint, double bcost);
	struct Assessment {
		double h = 0;
		double true_cost = 0;
		double size = 0;
		double benefit = 0;
		bool size_measured = false;
		idx_t measured_rows = 0;
		bool oversized = false;
	};

	// B(R) = cost(R) * h_R / size(R) for one node.
	Assessment Assess(hash_t fingerprint, double estimated_bcost, double estimated_bytes, bool speculative,
	                  idx_t now);
	// The admission/replacement decision, without changing anything: would a result of this size and benefit get...
	bool WouldAdmit(double bytes, double benefit, idx_t now);
	// Stores a captured result if the admission or replacement policy lets it in, applying updateHR (Algorithm 2)...
	bool Store(hash_t fingerprint, shared_ptr<CachedResult> result, double estimated_bcost, bool speculative,
	           bool force, idx_t now);
	// Records that a result outgrew the cache while being captured.
	void MarkOversized(hash_t fingerprint);
	// Evicts every cached result (the database changed).
	void Invalidate();
	// Forgets everything, the graph included.
	void Reset();
	idx_t MaxEntryBytes();
	RecyclerStats GetStats();
private:
	// All private helpers expect `lock` to be held.
	GraphNode *Find(hash_t fingerprint);
	double AgedH(GraphNode &node, idx_t now) const;
	void SetH(GraphNode &node, double h, idx_t now);
	// Algorithm 2 (with sign -1 on admission, Eq.
	void UpdateHR(GraphNode &node, double h_delta, idx_t now);
	// cost(R) = bcost(R) - sum of bcost over the direct materialized descendants.
	double TrueCost(GraphNode &node, double bcost);
	void SubtractDmdCosts(GraphNode &node, double &cost);
	double Benefit(GraphNode &node, idx_t now);
	static idx_t SizeGroup(double bytes);
	// The replacement policy: the lowest-benefit set of cached results in the newcomer's size group that frees...
	bool FindVictims(double bytes, double benefit, idx_t now, bool force, vector<hash_t> &victims);
	void Evict(GraphNode &node, idx_t now);
	struct Alias {
		string key;
		hash_t root;
	};
	mutex lock;
	std::unordered_map<hash_t, GraphNode> graph;
	std::unordered_map<hash_t, Alias> aliases;
	idx_t budget = 512ULL * 1024 * 1024;
	double alpha = 0.99;
	idx_t bytes_used = 0;
	idx_t latest = 0;
	RecyclerStats stats;
};
}

