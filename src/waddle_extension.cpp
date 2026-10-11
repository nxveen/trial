#define DUCKDB_EXTENSION_MAIN

#include "waddle_extension.hpp"
#include "recycler_optimizer.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// waddle_stats() and waddle_clear()
//===--------------------------------------------------------------------===//

//! Gives the table functions the same state the optimizer uses.
struct RecyclerFunctionInfo : public TableFunctionInfo {
	explicit RecyclerFunctionInfo(shared_ptr<RecyclerState> state) : state(std::move(state)) {
	}
	shared_ptr<RecyclerState> state;
};

struct RecyclerStatsData : public TableFunctionData {
	vector<Value> row;
};

struct RecyclerStatsState : public GlobalTableFunctionState {
	bool done = false;
};

static RecyclerState &GetRecyclerState(TableFunctionBindInput &input) {
	return *input.info->Cast<RecyclerFunctionInfo>().state;
}

//! The counters are read at bind time, so a query on them does not count itself.
static unique_ptr<FunctionData> StatsBind(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	auto &state = GetRecyclerState(input);
	auto recycler = state.recycler->GetStats();
	auto result = make_uniq<RecyclerStatsData>();
	auto add = [&](const string &name, idx_t value) {
		names.push_back(name);
		return_types.push_back(LogicalType::UBIGINT);
		result->row.push_back(Value::UBIGINT(value));
	};
	add("queries", state.queries_seen.load());
	add("whole_query_hits", state.whole_query_hits.load());
	add("subtree_hits", state.subtree_hits.load());
	add("whole_query_stores", state.whole_query_stores.load());
	add("subtree_stores", state.subtree_stores.load());
	add("history_stores", state.history_stores.load());
	add("speculative_stores", state.speculative_stores.load());
	add("not_admitted", state.not_admitted.load());
	add("admissions", recycler.admissions);
	add("replacements_rejected", recycler.replacements_rejected);
	add("evictions", recycler.evictions);
	add("graph_nodes", recycler.graph_nodes);
	add("cache_entries", recycler.cache_entries);
	add("cache_bytes", recycler.cache_bytes);
	names.emplace_back("cost_correction");
	return_types.emplace_back(LogicalType::DOUBLE);
	result->row.push_back(Value::DOUBLE(state.cost_model->Correction()));
	return std::move(result);
}

static unique_ptr<FunctionData> ClearBind(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	GetRecyclerState(input).Reset();
	names.emplace_back("success");
	return_types.emplace_back(LogicalType::BOOLEAN);
	auto result = make_uniq<RecyclerStatsData>();
	result->row.push_back(Value::BOOLEAN(true));
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> StatsInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<RecyclerStatsState>();
}

static void EmitSingleRow(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<RecyclerStatsState>();
	if (state.done) {
		return;
	}
	auto &bind_data = data.bind_data->Cast<RecyclerStatsData>();
	for (idx_t i = 0; i < bind_data.row.size(); i++) {
		output.SetValue(i, 0, bind_data.row[i]);
	}
	output.SetCardinality(1);
	state.done = true;
}

static void LoadInternal(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("waddle_cache", "Reuse results computed earlier in this session", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(true));
	config.AddExtensionOption("waddle_cache_limit_mb", "Memory the result cache may use, in MB", LogicalType::BIGINT,
	                          Value::BIGINT(512));
	config.AddExtensionOption("waddle_cache_max_rows", "Largest whole-query result worth caching, in rows",
	                          LogicalType::BIGINT, Value::BIGINT(200000));
	config.AddExtensionOption("waddle_subtree_reuse", "Also cache and reuse subtrees shared between queries",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption("waddle_subtree_max_rows", "Largest subtree result worth caching, in rows",
	                          LogicalType::BIGINT, Value::BIGINT(200000));
	config.AddExtensionOption("waddle_min_subtree_operators", "Smallest subtree worth caching, in operators",
	                          LogicalType::BIGINT, Value::BIGINT(1));
	config.AddExtensionOption("waddle_min_subtree_work",
	                          "Smallest subtree worth caching, in estimated rows scanned and processed inside it",
	                          LogicalType::BIGINT, Value::BIGINT(0));
	config.AddExtensionOption("waddle_max_subtree_captures", "Most store operators on subtrees in one query",
	                          LogicalType::BIGINT, Value::BIGINT(4));
	config.AddExtensionOption("waddle_cost_model",
	                          "Use the recycler's benefit-based policies (off: keep everything that fits)",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption("waddle_cost_decay",
	                          "Aging weight alpha < 1: a reference keeps this fraction of its weight per query",
	                          LogicalType::DOUBLE, Value::DOUBLE(0.99));
	config.AddExtensionOption("waddle_speculate_join_us",
	                          "Also store new joins speculatively when their estimated cost reaches this many "
	                          "microseconds (0: only final results and aggregations, as in the paper's examples)",
	                          LogicalType::DOUBLE, Value::DOUBLE(2000.0));
	config.AddExtensionOption("waddle_log", "Print what the recycler does with each query", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(false));

	auto state = make_shared_ptr<RecyclerState>();
	OptimizerExtension::Register(config, CreateRecyclerOptimizer(state));

	TableFunction stats("waddle_stats", {}, EmitSingleRow, StatsBind, StatsInit);
	stats.function_info = make_shared_ptr<RecyclerFunctionInfo>(state);
	loader.RegisterFunction(stats);

	TableFunction clear("waddle_clear", {}, EmitSingleRow, ClearBind, StatsInit);
	clear.function_info = make_shared_ptr<RecyclerFunctionInfo>(state);
	loader.RegisterFunction(clear);
}

void WaddleExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string WaddleExtension::Name() {
	return "waddle";
}
std::string WaddleExtension::Version() const {
#ifdef EXT_VERSION_WADDLE
	return EXT_VERSION_WADDLE;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(waddle, loader) {
	duckdb::LoadInternal(loader);
}
}
