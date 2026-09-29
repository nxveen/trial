#define DUCKDB_EXTENSION_MAIN

#include "waddle_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

#include "query_recycler_cache.hpp"
#include "query_recycler_optimizer.hpp"

namespace duckdb {

inline void WaddleScalarFun(
    DataChunk &args,
    ExpressionState &state,
    Vector &result) {

	auto &name_vector = args.data[0];

	UnaryExecutor::Execute<string_t, string_t>(
	    name_vector,
	    result,
	    args.size(),
	    [&](string_t name) {
		    return StringVector::AddString(
		        result,
		        "...........🦆 " + name.GetString());
	    });
}

static void LoadInternal(ExtensionLoader &loader) {

	auto waddle_scalar_function =
	    ScalarFunction(
	        "waddle",
	        {LogicalType::VARCHAR},
	        LogicalType::VARCHAR,
	        WaddleScalarFun);

	loader.RegisterFunction(waddle_scalar_function);

	// One recycler cache per DuckDB database instance.
	auto cache =
	    make_shared_ptr<RecyclingCache>();

	auto optimizer =
	    CreateQueryRecyclerOptimizer(cache);

	OptimizerExtension::Register(
	    loader.GetDatabaseInstance().config,
	    std::move(optimizer));
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