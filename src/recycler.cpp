#include "recycler.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {

void Recycler::Configure(idx_t budget_bytes, double alpha_p) {
	lock_guard<mutex> guard(lock);
	budget = budget_bytes;
	alpha = alpha_p;
	if (bytes_used > budget) {
		vector<hash_t> victims;
		FindVictims(0, 0, latest, true, victims);
		for (auto fingerprint : victims) {
			Evict(*Find(fingerprint), latest);
		}
	}
}

GraphNode *Recycler::Find(hash_t fingerprint) {
	auto entry = graph.find(fingerprint);
	return entry == graph.end() ? nullptr : &entry->second;
}

//===--------------------------------------------------------------------===//
// Importance factor and aging (Section III-C, Eq. 5)
//===--------------------------------------------------------------------===//

double Recycler::AgedH(GraphNode &node, idx_t now) const {
	if (node.h <= 0 || now <= node.h_time) {
		return node.h;
	}
	return node.h * std::pow(alpha, static_cast<double>(now - node.h_time));
}

void Recycler::SetH(GraphNode &node, double h, idx_t now) {
	node.h = MaxValue(h, 0.0);
	node.h_time = MaxValue(node.h_time, now);
}

//===--------------------------------------------------------------------===//
// Matching and insertion (Sections III-A, III-B)
//===--------------------------------------------------------------------===//

Recycler::Match Recycler::MatchOrInsert(hash_t fingerprint, const string &key, const vector<hash_t> &children,
                                        idx_t now) {
	lock_guard<mutex> guard(lock);
	latest = MaxValue(latest, now);
	Match match;
	auto node = Find(fingerprint);
	if (node) {
		if (node->key != key) {
			// Another plan already holds this hash: leave this one out.
			return match;
		}
		match.usable = true;
		match.existed = node->created_by != now;
		match.cached = node->cached;
		return match;
	}
	GraphNode inserted;
	inserted.fingerprint = fingerprint;
	inserted.key = key;
	inserted.children = children;
	inserted.created_by = now;
	inserted.h_time = now;
	graph.emplace(fingerprint, std::move(inserted));
	match.usable = true;
	return match;
}

shared_ptr<CachedResult> Recycler::Peek(hash_t fingerprint, const string &key) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	return node && node->key == key ? node->cached : nullptr;
}

void Recycler::Reference(hash_t fingerprint, idx_t now) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	if (!node || node->created_by == now) {
		return;
	}
	SetH(*node, AgedH(*node, now) + 1.0, now);
}

//===--------------------------------------------------------------------===//
// Base cost and true cost (Eq. 2)
//===--------------------------------------------------------------------===//

void Recycler::SetBaseCost(hash_t fingerprint, double bcost) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	if (node && bcost > 0) {
		node->bcost = bcost;
	}
}

double Recycler::BaseCost(hash_t fingerprint) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	return node ? node->bcost : 0;
}

void Recycler::SubtractDmdCosts(GraphNode &node, double &cost) {
	for (auto child_fingerprint : node.children) {
		auto child = Find(child_fingerprint);
		if (!child) {
			continue;
		}
		if (child->cached) {
			// A direct materialized descendant: its result replaces its subtree.
			cost -= child->bcost;
		} else {
			SubtractDmdCosts(*child, cost);
		}
	}
}

double Recycler::TrueCost(GraphNode &node, double bcost) {
	double cost = bcost;
	SubtractDmdCosts(node, cost);
	return MaxValue(cost, 0.0);
}

double Recycler::Benefit(GraphNode &node, idx_t now) {
	auto size = static_cast<double>(MaxValue<idx_t>(node.cached ? node.cached->bytes : node.measured_bytes, 1));
	return TrueCost(node, node.bcost) * AgedH(node, now) / size;
}

Recycler::Assessment Recycler::Assess(hash_t fingerprint, double estimated_bcost, double estimated_bytes,
                                      bool speculative, idx_t now) {
	lock_guard<mutex> guard(lock);
	Assessment assessment;
	auto node = Find(fingerprint);
	if (!node) {
		return assessment;
	}
	auto bcost = node->bcost > 0 ? node->bcost : estimated_bcost;
	assessment.size_measured = node->measured_bytes > 0;
	assessment.measured_rows = node->measured_rows;
	assessment.size = MaxValue(assessment.size_measured ? static_cast<double>(node->measured_bytes) : estimated_bytes,
	                           1.0);
	assessment.oversized = node->oversized;
	assessment.h = speculative ? SPECULATIVE_H : AgedH(*node, now);
	assessment.true_cost = TrueCost(*node, bcost);
	assessment.benefit = assessment.true_cost * assessment.h / assessment.size;
	return assessment;
}

//===--------------------------------------------------------------------===//
// updateHR (Algorithm 2, Eq. 3 and 4)
//===--------------------------------------------------------------------===//

void Recycler::UpdateHR(GraphNode &node, double h_delta, idx_t now) {
	SetH(node, AgedH(node, now) + h_delta, now);
	if (node.cached) {
		return;
	}
	for (auto child_fingerprint : node.children) {
		auto child = Find(child_fingerprint);
		if (child) {
			UpdateHR(*child, h_delta, now);
		}
	}
}

//===--------------------------------------------------------------------===//
// Cache policies (Section III-E)
//===--------------------------------------------------------------------===//

idx_t Recycler::SizeGroup(double bytes) {
	return static_cast<idx_t>(std::floor(std::log2(MaxValue(bytes, 1.0))));
}

bool Recycler::FindVictims(double bytes, double benefit, idx_t now, bool force, vector<hash_t> &victims) {
	victims.clear();
	if (bytes > static_cast<double>(budget)) {
		return false;
	}
	auto free_bytes = static_cast<double>(budget) - static_cast<double>(bytes_used);
	if (bytes <= free_bytes) {
		// Admission policy: there is still room.
		return true;
	}
	// Replacement policy: only the newcomer's size group is considered (when
	// forced, or when shrinking the budget, every group is).
	auto group = SizeGroup(bytes);
	vector<pair<double, hash_t>> candidates;
	for (auto &entry : graph) {
		auto &node = entry.second;
		if (node.cached && (force || SizeGroup(static_cast<double>(node.cached->bytes)) == group)) {
			candidates.emplace_back(Benefit(node, now), entry.first);
		}
	}
	std::sort(candidates.begin(), candidates.end());
	double freed = 0;
	double benefit_sum = 0;
	for (auto &candidate : candidates) {
		benefit_sum += candidate.first;
		victims.push_back(candidate.second);
		freed += static_cast<double>(Find(candidate.second)->cached->bytes);
		if (!force && benefit_sum / static_cast<double>(victims.size()) >= benefit) {
			// The set to give up is worth as much as the newcomer, on average.
			victims.clear();
			return false;
		}
		if (bytes <= free_bytes + freed) {
			return true;
		}
	}
	victims.clear();
	return false;
}

void Recycler::Evict(GraphNode &node, idx_t now) {
	if (!node.cached) {
		return;
	}
	bytes_used -= node.cached->bytes;
	node.cached.reset();
	stats.evictions++;
	// Eq. 4: queries that would have used this result fall back on its
	// (potential) direct materialized descendants.
	auto h = AgedH(node, now);
	for (auto child_fingerprint : node.children) {
		auto child = Find(child_fingerprint);
		if (child) {
			UpdateHR(*child, h, now);
		}
	}
}

bool Recycler::WouldAdmit(double bytes, double benefit, idx_t now) {
	lock_guard<mutex> guard(lock);
	vector<hash_t> victims;
	return FindVictims(bytes, benefit, now, false, victims);
}

bool Recycler::Store(hash_t fingerprint, shared_ptr<CachedResult> result, double estimated_bcost, bool speculative,
                     bool force, idx_t now) {
	lock_guard<mutex> guard(lock);
	latest = MaxValue(latest, now);
	auto node = Find(fingerprint);
	if (!node) {
		return false;
	}
	node->measured_bytes = MaxValue<idx_t>(result->bytes, 1);
	node->measured_rows = result->data->Count();
	if (node->bcost <= 0) {
		node->bcost = estimated_bcost;
	}
	if (node->cached) {
		// Captured twice (two connections, or a subtree that occurs twice in one
		// query): keep what is there, unless only the new copy is ordered.
		if (node->cached->ordered || !result->ordered) {
			return true;
		}
		bytes_used -= node->cached->bytes;
		bytes_used += result->bytes;
		node->cached = std::move(result);
		return true;
	}
	auto size = static_cast<double>(MaxValue<idx_t>(result->bytes, 1));
	auto h = speculative ? SPECULATIVE_H : AgedH(*node, now);
	auto benefit = TrueCost(*node, node->bcost) * h / size;
	vector<hash_t> victims;
	if (!FindVictims(size, benefit, now, force, victims)) {
		stats.replacements_rejected++;
		return false;
	}
	for (auto victim : victims) {
		Evict(*Find(victim), now);
	}
	bytes_used += result->bytes;
	node->cached = std::move(result);
	stats.admissions++;
	// Eq. 3 / Algorithm 2: queries that used a descendant now use this result.
	auto own_h = AgedH(*node, now);
	for (auto child_fingerprint : node->children) {
		auto child = Find(child_fingerprint);
		if (child) {
			UpdateHR(*child, -own_h, now);
		}
	}
	return true;
}

void Recycler::MarkOversized(hash_t fingerprint) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	if (node) {
		node->oversized = true;
	}
}

void Recycler::Invalidate() {
	lock_guard<mutex> guard(lock);
	for (auto &entry : graph) {
		Evict(entry.second, latest);
	}
	bytes_used = 0;
}

void Recycler::Reset() {
	lock_guard<mutex> guard(lock);
	graph.clear();
	bytes_used = 0;
	latest = 0;
	stats = RecyclerStats();
}

idx_t Recycler::MaxEntryBytes() {
	lock_guard<mutex> guard(lock);
	return budget / 4;
}

RecyclerStats Recycler::GetStats() {
	lock_guard<mutex> guard(lock);
	auto result = stats;
	result.graph_nodes = graph.size();
	result.cache_bytes = bytes_used;
	for (auto &entry : graph) {
		result.cache_entries += entry.second.cached ? 1 : 0;
	}
	return result;
}

} // namespace duckdb
