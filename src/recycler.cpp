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

Recycler::Match Recycler::MatchOrInsert(hash_t fingerprint, const string &key, const vector<hash_t> &children,
                                        idx_t now) {
	lock_guard<mutex> guard(lock);
	latest = MaxValue(latest, now);
	Match match;
	auto node = Find(fingerprint);
	if (node) {
		if (node->key != key) {
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

void Recycler::SetAlias(hash_t fingerprint, const string &key, hash_t root) {
	lock_guard<mutex> guard(lock);
	auto &alias = aliases[fingerprint];
	if (alias.key.empty() || alias.key == key) {
		alias.key = key;
		alias.root = root;
	}
}

shared_ptr<CachedResult> Recycler::LookupAlias(hash_t fingerprint, const string &key, hash_t &root) {
	lock_guard<mutex> guard(lock);
	auto alias = aliases.find(fingerprint);
	if (alias == aliases.end() || alias->second.key != key) {
		return nullptr;
	}
	auto node = Find(alias->second.root);
	if (!node) {
		return nullptr;
	}
	root = alias->second.root;
	return node->cached;
}

void Recycler::Reference(hash_t fingerprint, idx_t now) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	if (!node || node->created_by == now) {
		return;
	}
	SetH(*node, AgedH(*node, now) + 1.0, now);
}

void Recycler::SetBaseCost(hash_t fingerprint, double bcost) {
	lock_guard<mutex> guard(lock);
	auto node = Find(fingerprint);
	if (node && bcost > 0) {
		node->bcost = bcost;
	}
}

void Recycler::SubtractDmdCosts(GraphNode &node, double &cost) {
	for (auto child_fingerprint : node.children) {
		auto child = Find(child_fingerprint);
		if (!child) {
			continue;
		}
		if (child->cached) {
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
		return true;
	}
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

void Recycler::RegisterRangeVariant(hash_t signature, const string &key, hash_t node,
                                    const vector<Value> &constants) {
	lock_guard<mutex> guard(lock);
	auto &group = range_index[signature];
	if (group.key.empty()) {
		group.key = key;
	} else if (group.key != key) {
		return;
	}
	for (auto &variant : group.variants) {
		if (variant.node == node) {
			return;
		}
	}
	group.variants.push_back(RangeVariant {node, constants});
}

vector<pair<shared_ptr<CachedResult>, vector<Value>>> Recycler::CachedRangeVariants(hash_t signature,
                                                                                   const string &key) {
	lock_guard<mutex> guard(lock);
	vector<pair<shared_ptr<CachedResult>, vector<Value>>> found;
	auto entry = range_index.find(signature);
	if (entry == range_index.end() || entry->second.key != key) {
		return found;
	}
	auto &variants = entry->second.variants;
	for (idx_t i = variants.size(); i-- > 0;) {
		auto node = Find(variants[i].node);
		if (node && node->cached) {
			found.emplace_back(node->cached, variants[i].constants);
		}
	}
	return found;
}

void Recycler::Reset() {
	lock_guard<mutex> guard(lock);
	graph.clear();
	range_index.clear();
	aliases.clear();
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
}

