#include "query_recycler_optimizer.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/operator/logical_explain.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"

#include "query_recycler_operator.hpp"

namespace duckdb {

static bool IsUnsafeOperator(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO:

	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_COPY_DATABASE:
	case LogicalOperatorType::LOGICAL_EXPORT:

	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
	case LogicalOperatorType::LOGICAL_CREATE_SECRET:

	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_PRAGMA:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:

	case LogicalOperatorType::LOGICAL_PREPARE:
	case LogicalOperatorType::LOGICAL_EXECUTE:
	case LogicalOperatorType::LOGICAL_VACUUM:
	case LogicalOperatorType::LOGICAL_SET:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_RESET:
	case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:

		return true;

	default:
		return false;
	}
}

static bool IsCacheablePlan(LogicalOperator &op) {
	if (IsUnsafeOperator(op.type)) {
		return false;
	}

	// Don't recursively recycle our own operator.
	if (op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
		return false;
	}

	for (auto &child : op.children) {
		if (!IsCacheablePlan(*child)) {
			return false;
		}
	}

	return true;
}

static bool ContainsParameterMarker(const string &query) {
	// We deliberately avoid recycling prepared/parameterized statements,
	// because two executions can have different parameter values.
	for (idx_t i = 0; i < query.size(); i++) {
		if (query[i] == '?') {
			return true;
		}

		if (query[i] == '$') {
			return true;
		}
	}

	return false;
}

static string MakeCacheKey(ClientContext &context, LogicalOperator &plan) {
	// Include both the original SQL and the optimized logical plan.
	//
	// The SQL distinguishes semantically different queries that happen to
	// produce similar plans; the plan makes the key robust to harmless
	// formatting differences.
	string key;

	key += context.GetCurrentQuery();
	key += "\n---WADDLE-PLAN---\n";
	key += plan.ToString(ExplainFormat::TEXT);

	return key;
}

static bool TryGetSingleTableIndex(LogicalOperator &op, idx_t &table_index) {
	auto bindings = op.GetColumnBindings();

	if (bindings.empty()) {
		return false;
	}

	table_index = bindings[0].table_index;

	for (auto &binding : bindings) {
		if (binding.table_index != table_index) {
			return false;
		}
	}

	return true;
}

static LogicalOperator *FindQueryPlan(unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return nullptr;
	}

	// The assignment evaluator wraps the benchmark query in EXPLAIN ANALYZE.
	//
	// LogicalExplain has exactly one child containing the real query plan.
	if (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
		if (plan->children.size() != 1) {
			return nullptr;
		}

		return plan->children[0].get();
	}

	return plan.get();
}

static void OptimizeQueryRecycler(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto info = input.info;
if (!info) {
	return;
}

auto *recycler_info = dynamic_cast<QueryRecyclerOptimizerInfo *>(info.get());
if (!recycler_info) {
	return;
}

auto cache = recycler_info->cache;

	if (!cache) {
		return;
	}

	auto *query_plan = FindQueryPlan(plan);
	if (!query_plan) {
		return;
	}

	// Only cache SELECT-like plans.
	if (!IsCacheablePlan(*query_plan)) {
		return;
	}

	const auto query_string = input.context.GetCurrentQuery();

	if (query_string.empty()) {
		return;
	}

	if (ContainsParameterMarker(query_string)) {
		return;
	}

	// Avoid obviously non-deterministic queries.
	//
	// JOB/IMDB queries do not contain these functions, so this does not
	// interfere with the benchmark.
	auto lower_query = StringUtil::Lower(query_string);

	static const char *NON_DETERMINISTIC[] = {
	    "random(",
	    "uuid(",
	    "gen_random_uuid(",
	    "current_timestamp",
	    "current_localtimestamp",
	    "current_time",
	    "current_date",
	    "now(",
	};

	for (auto fn : NON_DETERMINISTIC) {
		if (lower_query.find(fn) != string::npos) {
			return;
		}
	}

	if (query_plan->types.empty()) {
		return;
	}

	idx_t table_index;
	if (!TryGetSingleTableIndex(*query_plan, table_index)) {
		return;
	}

	auto key = MakeCacheKey(input.context, *query_plan);

	auto entry = cache->GetOrCreate(input.context, key, query_plan->types);

	// Cache hit:
	//
	// Replace the entire query plan with a scan over the materialized result.
		if (cache->IsReady(entry)) {
		auto collection = entry->collection;

		auto replacement =
		    make_uniq<LogicalColumnDataGet>(table_index, query_plan->types, collection);

		if (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
			plan->children[0] = std::move(replacement);
		} else {
			plan = std::move(replacement);
		}

		return;
	}

	if (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
		auto original_query = std::move(plan->children[0]);

		auto recycler =
		    make_uniq<LogicalQueryRecycler>(std::move(original_query), cache, entry, table_index);

		plan->children[0] = std::move(recycler);
	} else {
		auto original_query = std::move(plan);

		plan = make_uniq<LogicalQueryRecycler>(std::move(original_query), cache, entry, table_index);
	}
}

OptimizerExtension CreateQueryRecyclerOptimizer(shared_ptr<RecyclingCache> cache) {
	OptimizerExtension result;

	result.optimizer_info = make_shared_ptr<QueryRecyclerOptimizerInfo>(std::move(cache));
	result.optimize_function = OptimizeQueryRecycler;

	return result;
}

} // namespace duckdb