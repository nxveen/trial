#include "cost_model.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include <cmath>
namespace duckdb {
double CostModel::EstimatedRows(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		return 0;
	}
	if (op.has_estimated_cardinality) {
		return static_cast<double>(op.estimated_cardinality);
	}
	double rows = 0;
	for (auto &child : op.children) {
		rows = MaxValue(rows, EstimatedRows(*child));
	}
	return rows;
}

idx_t CostModel::TableRows(ClientContext &context, LogicalGet &get) {
	if (!get.function.cardinality) {
		return 0;
	}
	auto stats = get.function.cardinality(context, get.bind_data.get());
	if (!stats || !stats->has_estimated_cardinality) {
		return 0;
	}
	return stats->estimated_cardinality;
}

static double BaseTableRows(ClientContext &context, LogicalGet &get) {
	auto rows = CostModel::TableRows(context, get);
	return rows ? static_cast<double>(rows) : CostModel::EstimatedRows(get);
}

// Number of operations (function calls, comparisons, casts...) an expression performs per row.
static double ExpressionOperations(const Expression &expression) {
	double operations = 0;
	switch (expression.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
	case ExpressionClass::BOUND_REF:
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_PARAMETER:
		break;
	default:
		operations = 1;
		break;
	}
	ExpressionIterator::EnumerateChildren(expression, [&](const Expression &child) {
		operations += ExpressionOperations(child);
	});
	return operations;
}

// Operations a filter pushed into a table scan performs per row.
static double TableFilterOperations(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::EXPRESSION_FILTER:
		return MaxValue(ExpressionOperations(*filter.Cast<ExpressionFilter>().expr), 1.0);
	case TableFilterType::CONJUNCTION_AND:
	case TableFilterType::CONJUNCTION_OR: {
		double operations = 0;
		for (auto &child : static_cast<const ConjunctionFilter &>(filter).child_filters) {
			operations += TableFilterOperations(*child);
		}
		return operations;
	}
	case TableFilterType::OPTIONAL_FILTER:
	case TableFilterType::DYNAMIC_FILTER:
		return 0;
	default:
		return 1;
	}
}

// Per-row cost of hashing rows into `groups` groups: cheap while the hash table fits in cache, about five times...
static double GroupedRowCost(double groups, const CostParameters &p) {
	auto position = (std::log10(MaxValue(groups, 1.0)) - 3.0) / 2.0;
	position = MinValue(MaxValue(position, 0.0), 1.0);
	return p.aggregate_few_groups + position * (p.aggregate_many_groups - p.aggregate_few_groups);
}

static double SortCost(double rows, double per_row_log) {
	return rows <= 1 ? 0 : rows * std::log2(rows) * per_row_log;
}

double CostModel::OperatorCost(LogicalOperator &op, ClientContext &context, const CostParameters &p) {
	auto output = EstimatedRows(op);
	double input = 0;
	for (auto &child : op.children) {
		input += EstimatedRows(*child);
	}
	auto expressions = static_cast<double>(MaxValue<idx_t>(op.expressions.size(), 1));
	double operations = 0;
	for (auto &expression : op.expressions) {
		operations += ExpressionOperations(*expression);
	}
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (!get.GetTable()) {
			return output * p.default_row;
		}
		auto base = BaseTableRows(context, get);
		auto columns = static_cast<double>(MaxValue<idx_t>(get.GetColumnIds().size(), 1));
		double filter_operations = 0;
		for (auto &entry : get.table_filters.filters) {
			filter_operations += TableFilterOperations(*entry.second);
		}
		return base * (columns * p.scan_row_column + filter_operations * p.scan_filter_row);
	}
	case LogicalOperatorType::LOGICAL_FILTER:
		return input * MaxValue(operations, 1.0) * p.filter_row;
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return input * operations * p.projection_row;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		if (op.children.size() != 2) {
			return input * p.default_row;
		}
		auto left = EstimatedRows(*op.children[0]);
		auto right = EstimatedRows(*op.children[1]);
		bool has_equality = false;
		for (auto &condition : op.Cast<LogicalComparisonJoin>().conditions) {
			if (condition.comparison == ExpressionType::COMPARE_EQUAL ||
			    condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
				has_equality = true;
			}
		}
		if (has_equality) {
			return right * p.hash_build_row + left * p.hash_probe_row + output * p.join_output_row;
		}
		return SortCost(left, p.sort_row_log) + SortCost(right, p.sort_row_log) + output * p.join_output_row;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN: {
		if (op.children.size() != 2) {
			return input * p.default_row;
		}
		if (op.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN) {
			return input * p.default_row;
		}
		auto left = EstimatedRows(*op.children[0]);
		auto right = EstimatedRows(*op.children[1]);
		return left * right * p.nested_loop_pair + output * p.join_output_row;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggregate = op.Cast<LogicalAggregate>();
		auto functions = static_cast<double>(aggregate.expressions.size()) * p.aggregate_function;
		if (aggregate.groups.empty()) {
			return input * functions;
		}
		return input * (GroupedRowCost(output, p) + functions);
	}
	case LogicalOperatorType::LOGICAL_DISTINCT:
		return input * GroupedRowCost(output, p);
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		return SortCost(input, p.sort_row_log);
	case LogicalOperatorType::LOGICAL_TOP_N:
		return input * p.top_n_row;
	case LogicalOperatorType::LOGICAL_WINDOW:
		return SortCost(input, p.sort_row_log) + input * expressions * p.window_row;
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return input * p.hash_build_row;
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		return 0;
	default:
		return MaxValue(input, output) * p.default_row;
	}
}

static double TypeBytes(const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::VARCHAR:
		return 32;
	case PhysicalType::LIST:
	case PhysicalType::STRUCT:
	case PhysicalType::ARRAY:
		return 64;
	default:
		return static_cast<double>(GetTypeIdSize(type.InternalType()));
	}
}

double CostModel::RowBytes(const vector<LogicalType> &types) {
	double bytes = 0;
	for (auto &type : types) {
		bytes += TypeBytes(type) + 0.125;
	}
	return MaxValue(bytes, 1.0);
}

void CostModel::RecordRatio(double measured, double estimated) {
	if (estimated < 200000.0 && measured < 1000000.0) {
		return;
	}
	auto ratio = MinValue(MaxValue(measured / MaxValue(estimated, 1.0), 1.0 / 64), 64.0);
	lock_guard<mutex> guard(lock);
	log_ratio_sum += std::log(ratio);
	log_ratio_count++;
}

double CostModel::Correction() {
	lock_guard<mutex> guard(lock);
	auto mean = log_ratio_sum / (log_ratio_count + 2.0);
	return MinValue(MaxValue(std::exp(mean), 0.25), 64.0);
}

void CostModel::Reset() {
	lock_guard<mutex> guard(lock);
	log_ratio_sum = 0;
	log_ratio_count = 0;
}
}

