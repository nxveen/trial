#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

// One cached query result.
struct RecyclingCacheEntry {
	RecyclingCacheEntry(string key_p, shared_ptr<ColumnDataCollection> collection_p)
	    : key(std::move(key_p)), collection(std::move(collection_p)), ready(false), bytes(0) {
	}

	string key;
	shared_ptr<ColumnDataCollection> collection;

	// Protected by RecyclingCache::lock.
	bool ready;
	idx_t bytes;
};

// Bounded cache for completed query results.
//
// The cache is deliberately per extension/DatabaseInstance rather than global
// across databases.
class RecyclingCache {
public:
	// Keep the cache conservative. The benchmark has an 8 GB memory limit.
	static constexpr idx_t MAX_ENTRIES = 32;
	static constexpr idx_t MAX_BYTES = 1024ULL * 1024ULL * 1024ULL; // 1 GiB

	RecyclingCache() = default;

	// Returns an existing entry or creates a new one.
	//
	// The returned entry is initially not ready. The physical recycler fills
	// it during the first execution.
	shared_ptr<RecyclingCacheEntry> GetOrCreate(ClientContext &context, const string &key,
	                                            const vector<LogicalType> &types);

	// True iff the entry contains a complete successful result.
	bool IsReady(const shared_ptr<RecyclingCacheEntry> &entry);

	// Prepare an entry for a fresh materialization.
	void BeginMaterialization(const shared_ptr<RecyclingCacheEntry> &entry);

	// Mark a materialization as successfully completed.
	void MarkReady(const shared_ptr<RecyclingCacheEntry> &entry);

	// Drop an incomplete entry from the cache.
	void MarkFailed(const shared_ptr<RecyclingCacheEntry> &entry);

	// Statistics, useful for debugging.
	idx_t EntryCount();
	idx_t CachedBytes();

private:
	void EvictIfNeeded(const shared_ptr<RecyclingCacheEntry> &keep);

private:
	mutex lock;
	unordered_map<string, shared_ptr<RecyclingCacheEntry>> entries;
	idx_t total_bytes = 0;
};

} // namespace duckdb