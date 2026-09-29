#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

// A single materialized logical-plan subtree.
struct RecyclingCacheEntry {
	RecyclingCacheEntry(string fingerprint_p, vector<LogicalType> types_p,
	                    vector<ColumnBinding> bindings_p,
	                    shared_ptr<ColumnDataCollection> collection_p)
	    : fingerprint(std::move(fingerprint_p)), types(std::move(types_p)), bindings(std::move(bindings_p)),
	      collection(std::move(collection_p)) {
	}

	string fingerprint;

	// Output schema of the cached subtree.
	vector<LogicalType> types;

	// IMPORTANT:
	// We retain the original logical column bindings. This allows a cached
	// join/subplan to replace the original subtree without breaking references
	// in its parent operator.
	vector<ColumnBinding> bindings;

	shared_ptr<ColumnDataCollection> collection;

	// Cache state.
	bool ready = false;
	bool building = false;

	// LRU metadata.
	idx_t last_used = 0;
	idx_t bytes = 0;
	idx_t hits = 0;
};

// Bounded cache containing materialized logical subplans.
class RecyclingCache {
public:
	static constexpr idx_t MAX_ENTRIES = 128;

	// 2 GiB upper bound for cached results.
	static constexpr idx_t MAX_BYTES = 2ULL * 1024ULL * 1024ULL * 1024ULL;

public:
	RecyclingCache() {
	}

	// Look for a completed entry.
	shared_ptr<RecyclingCacheEntry> FindReady(const string &fingerprint);

	// Start creating a new entry.
	//
	// Returns nullptr if another execution is already materializing this
	// fingerprint. This avoids two concurrent queries writing to the same
	// ColumnDataCollection.
	shared_ptr<RecyclingCacheEntry> BeginBuild(ClientContext &context, const string &fingerprint,
	                                           const vector<LogicalType> &types,
	                                           const vector<ColumnBinding> &bindings);

	void MarkReady(const shared_ptr<RecyclingCacheEntry> &entry);

	void MarkFailed(const shared_ptr<RecyclingCacheEntry> &entry);

	void Touch(const shared_ptr<RecyclingCacheEntry> &entry);

	idx_t EntryCount();

	idx_t CachedBytes();

private:
	void EvictIfNeeded();

private:
	mutex lock;

	unordered_map<string, shared_ptr<RecyclingCacheEntry>> entries;

	idx_t total_bytes = 0;
	idx_t clock = 0;
};

} // namespace duckdb