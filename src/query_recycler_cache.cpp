#include "query_recycler_cache.hpp"

namespace duckdb {

shared_ptr<RecyclingCacheEntry> RecyclingCache::FindReady(const string &fingerprint) {
	lock_guard<mutex> guard(lock);

	auto it = entries.find(fingerprint);
	if (it == entries.end()) {
		return nullptr;
	}

	auto entry = it->second;

	if (!entry->ready || entry->building) {
		return nullptr;
	}

	entry->hits++;
	entry->last_used = ++clock;

	return entry;
}

shared_ptr<RecyclingCacheEntry> RecyclingCache::BeginBuild(ClientContext &context,
                                                            const string &fingerprint,
                                                            const vector<LogicalType> &types,
                                                            const vector<ColumnBinding> &bindings) {
	lock_guard<mutex> guard(lock);

	auto existing = entries.find(fingerprint);

	if (existing != entries.end()) {
		// Somebody else is currently building this exact subplan.
		if (existing->second->building) {
			return nullptr;
		}

		// A completed entry should have been returned by FindReady().
		if (existing->second->ready) {
			return nullptr;
		}

		// Remove a stale failed entry.
		if (existing->second->collection) {
			if (total_bytes >= existing->second->bytes) {
				total_bytes -= existing->second->bytes;
			}
		}

		entries.erase(existing);
	}

	auto collection = make_shared_ptr<ColumnDataCollection>(context, types);

	auto entry = make_shared_ptr<RecyclingCacheEntry>(
	    fingerprint, types, bindings, std::move(collection));

	entry->building = true;
	entry->ready = false;
	entry->last_used = ++clock;

	entries.emplace(fingerprint, entry);

	return entry;
}

void RecyclingCache::MarkReady(const shared_ptr<RecyclingCacheEntry> &entry) {
	if (!entry) {
		return;
	}

	lock_guard<mutex> guard(lock);

	if (!entry->collection) {
		entry->building = false;
		entry->ready = false;
		return;
	}

	entry->bytes = entry->collection->SizeInBytes();

	entry->building = false;
	entry->ready = true;
	entry->last_used = ++clock;

	total_bytes = 0;

	for (auto &kv : entries) {
		if (kv.second->ready) {
			total_bytes += kv.second->bytes;
		}
	}

	EvictIfNeeded();
}

void RecyclingCache::MarkFailed(const shared_ptr<RecyclingCacheEntry> &entry) {
	if (!entry) {
		return;
	}

	lock_guard<mutex> guard(lock);

	auto it = entries.find(entry->fingerprint);

	if (it == entries.end()) {
		return;
	}

	if (it->second != entry) {
		return;
	}

	if (entry->ready) {
		return;
	}

	entries.erase(it);

	total_bytes = 0;

	for (auto &kv : entries) {
		if (kv.second->ready) {
			total_bytes += kv.second->bytes;
		}
	}
}

void RecyclingCache::Touch(const shared_ptr<RecyclingCacheEntry> &entry) {
	if (!entry) {
		return;
	}

	lock_guard<mutex> guard(lock);

	entry->last_used = ++clock;
}

void RecyclingCache::EvictIfNeeded() {
	while ((entries.size() > MAX_ENTRIES || total_bytes > MAX_BYTES) && !entries.empty()) {
		auto victim = entries.end();

		for (auto it = entries.begin(); it != entries.end(); ++it) {
			auto &entry = it->second;

			// Never evict an entry while it is being materialized.
			if (entry->building) {
				continue;
			}

			if (victim == entries.end() ||
			    entry->last_used < victim->second->last_used) {
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