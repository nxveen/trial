#include "query_recycler_cache.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"

namespace duckdb {

shared_ptr<RecyclingCacheEntry> RecyclingCache::GetOrCreate(ClientContext &context, const string &key,
                                                            const vector<LogicalType> &types) {
	lock_guard<mutex> guard(lock);

	auto entry = entries.find(key);
	if (entry != entries.end()) {
		return entry->second;
	}

	auto collection = make_shared_ptr<ColumnDataCollection>(context, types);

	auto result = make_shared_ptr<RecyclingCacheEntry>(key, collection);
	entries.emplace(key, result);

	return result;
}

bool RecyclingCache::IsReady(const shared_ptr<RecyclingCacheEntry> &entry) {
	lock_guard<mutex> guard(lock);
	return entry && entry->ready;
}

void RecyclingCache::BeginMaterialization(const shared_ptr<RecyclingCacheEntry> &entry) {
	if (!entry) {
		return;
	}

	lock_guard<mutex> guard(lock);

	// The entry is expected to be incomplete here.
	entry->ready = false;

	if (entry->collection) {
		entry->collection->Reset();
	}

	entry->bytes = 0;
}

void RecyclingCache::MarkReady(const shared_ptr<RecyclingCacheEntry> &entry) {
	if (!entry) {
		return;
	}

	lock_guard<mutex> guard(lock);

	if (!entry->collection) {
		entry->ready = false;
		return;
	}

	entry->bytes = entry->collection->SizeInBytes();
	entry->ready = true;

	// Recalculate defensively.
	total_bytes = 0;
	for (auto &kv : entries) {
		if (kv.second->ready) {
			total_bytes += kv.second->bytes;
		}
	}

	EvictIfNeeded(entry);
}

void RecyclingCache::MarkFailed(const shared_ptr<RecyclingCacheEntry> &entry) {
	if (!entry) {
		return;
	}

	lock_guard<mutex> guard(lock);

	entry->ready = false;
	entry->bytes = 0;

	if (entry->collection) {
		entry->collection->Reset();
	}

	total_bytes = 0;
	for (auto &kv : entries) {
		if (kv.second->ready) {
			total_bytes += kv.second->bytes;
		}
	}
}

void RecyclingCache::EvictIfNeeded(const shared_ptr<RecyclingCacheEntry> &keep) {
	while ((entries.size() > MAX_ENTRIES || total_bytes > MAX_BYTES) && entries.size() > 1) {
		auto victim = entries.end();

		for (auto it = entries.begin(); it != entries.end(); ++it) {
			if (it->second == keep) {
				continue;
			}

			if (!it->second->ready) {
				victim = it;
				break;
			}

			if (victim == entries.end()) {
				victim = it;
			}
		}

		if (victim == entries.end()) {
			break;
		}

		if (victim->second->ready) {
			if (total_bytes >= victim->second->bytes) {
				total_bytes -= victim->second->bytes;
			} else {
				total_bytes = 0;
			}
		}

		entries.erase(victim);
	}

	// A single query result larger than MAX_BYTES is still retained. This is
	// preferable to immediately throwing away the result we just computed.
	total_bytes = 0;
	for (auto &kv : entries) {
		if (kv.second->ready) {
			total_bytes += kv.second->bytes;
		}
	}
}

idx_t RecyclingCache::EntryCount() {
	lock_guard<mutex> guard(lock);
	return entries.size();
}

idx_t RecyclingCache::CachedBytes() {
	lock_guard<mutex> guard(lock);
	return total_bytes;
}

} // namespace duckdb