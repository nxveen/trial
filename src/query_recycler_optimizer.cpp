#include "query_recycler_optimizer.hpp"

#include "query_recycler_operator.hpp"

namespace duckdb {

namespace {

// -----------------------------------------------------------------------------
// Candidate selection
//
// We deliberately keep this conservative.
//
// Aggregate/group-by and DISTINCT are self-contained relational results.
// Their output does not carry ORDER BY / WINDOW semantics, and they are the
// most useful candidates for the repeated derived subqueries in JOB.
//
// We intentionally DO NOT recycle:
//   - joins
//   - filters
//   - projections
//   - ORDER BY
//   - TOP N
//   - WINDOW
//   - set operations
//
// This avoids changing optimizer-visible properties of large parts of the
// outer query and greatly reduces the chance of an accidental cache match.
// -----------------------------------------------------------------------------

static bool IsRecyclerCandidate(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_DISTINCT:
		return true;

	default:
		return false;
	}
}

// -----------------------------------------------------------------------------
// Fingerprinting
//
// The fingerprint describes the logical subtree, rather than its physical
// execution state.
//
// We intentionally do not include the ColumnBinding values themselves.
// Bindings are query-instance-specific and are therefore not suitable for
// matching the same logical subquery across separate executions.
//
// The current query's bindings are installed separately when a cache hit is
// converted into LogicalRecyclerScan.
// -----------------------------------------------------------------------------

static string BuildFingerprint(const LogicalOperator &op) {
	string result;

	// Logical operator identity.
	result += "OP=";
	result += op.GetName();
	result += "|";

	// Operator-specific parameters.
	//
	// ParamsToString() contains information specific to the particular
	// logical operator, such as aggregate/grouping configuration.
	result += "PARAMS=";

	auto params = op.ParamsToString();

	for (auto &param : params) {
		result += param.first;
		result += "=";
		result += param.second;
		result += ";";
	}

	result += "|";

	// Output schema.
	result += "TYPES=";

	for (auto &type : op.types) {
		result += type.ToString();
		result += ";";
	}

	result += "|";

	// Expressions belonging to this operator.
	result += "EXPRESSIONS=";

	for (auto &expr : op.expressions) {
		result += expr->ToString();
		result += ";";
	}

	result += "|";

	// Recursively fingerprint the complete input subtree.
	result += "CHILDREN=";

	for (auto &child : op.children) {
		result += "{";
		result += BuildFingerprint(*child);
		result += "}";
	}

	return result;
}

// -----------------------------------------------------------------------------
// Rewrite
// -----------------------------------------------------------------------------

static void RewritePlan(unique_ptr<LogicalOperator> &node,
                        const shared_ptr<RecyclingCache> &cache,
                        ClientContext &context) {
	if (!node) {
		return;
	}

	// EXPLAIN / EXPLAIN ANALYZE contains the actual query plan as a child.
	//
	// We must recurse into it rather than replacing the EXPLAIN operator.
	if (node->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
		for (auto &child : node->children) {
			RewritePlan(child, cache, context);
		}
		return;
	}

	// -------------------------------------------------------------------------
	// First rewrite children.
	//
	// This allows nested recyclable subqueries to be considered before their
	// parent aggregate.
	// -------------------------------------------------------------------------

	for (auto &child : node->children) {
		RewritePlan(child, cache, context);
	}

	if (!IsRecyclerCandidate(*node)) {
		return;
	}

	// -------------------------------------------------------------------------
	// Compute the fingerprint from the original logical subtree.
	//
	// This is done after recursively visiting children, but the fingerprint
	// itself is based on the current logical tree. Recycler operators created
	// below this node are therefore represented consistently when this node
	// becomes a cache entry.
	// -------------------------------------------------------------------------

	auto fingerprint = BuildFingerprint(*node);

	// -------------------------------------------------------------------------
	// CACHE HIT
	// -------------------------------------------------------------------------

	auto hit = cache->FindReady(fingerprint);

	if (hit) {
		// IMPORTANT:
		//
		// Never use the bindings stored in the old query's cache entry.
		//
		// The same logical subquery can be bound to different ColumnBinding
		// values in a later query.
		//
		// The cached collection contains only data, while the scan must expose
		// the bindings expected by THIS query's parent operator.
		auto current_bindings = node->GetColumnBindings();

		cache->Touch(hit);

		node = make_uniq<LogicalRecyclerScan>(
		    hit,
		    std::move(current_bindings));

		return;
	}

	// -------------------------------------------------------------------------
	// CACHE MISS
	// -------------------------------------------------------------------------

	auto entry = cache->BeginBuild(
	    context,
	    fingerprint,
	    node->types,
	    node->GetColumnBindings());

	if (!entry) {
		// Another execution is already building this exact subtree.
		//
		// Do not try to use an unfinished ColumnDataCollection.
		// Leave this query's original subtree intact.
		return;
	}

	// The original subtree becomes the child of the materializer.
	//
	// PhysicalRecyclerMaterialize will execute the child, append its result
	// into the cache collection, mark the entry ready, and expose the same
	// collection as its source.
	node = make_uniq<LogicalRecyclerMaterialize>(
	    std::move(node),
	    cache,
	    entry);
}

} // namespace

// -----------------------------------------------------------------------------
// Optimizer callback
// -----------------------------------------------------------------------------

static void QueryRecyclerOptimize(
    OptimizerExtensionInput &input,
    unique_ptr<LogicalOperator> &plan) {

	if (!plan) {
		return;
	}

	auto *info =
	    dynamic_cast<QueryRecyclerOptimizerInfo *>(input.info.get());

	if (!info || !info->cache) {
		return;
	}

	RewritePlan(
	    plan,
	    info->cache,
	    input.context);
}

// -----------------------------------------------------------------------------
// Optimizer factory
//
// This matches the architecture already present in your waddle_extension.cpp:
//
//     auto optimizer = CreateQueryRecyclerOptimizer(cache);
//
//     OptimizerExtension::Register(
//         loader.GetDatabaseInstance().config,
//         std::move(optimizer));
// -----------------------------------------------------------------------------

OptimizerExtension CreateQueryRecyclerOptimizer(
    shared_ptr<RecyclingCache> cache) {

	OptimizerExtension optimizer;

	optimizer.optimize_function = QueryRecyclerOptimize;

	optimizer.optimizer_info =
	    make_shared_ptr<QueryRecyclerOptimizerInfo>(
	        std::move(cache));

	return optimizer;
}

} // namespace duckdb