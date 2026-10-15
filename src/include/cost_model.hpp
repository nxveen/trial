#pragma once
#include "duckdb/common/mutex.hpp"
#include "duckdb/planner/logical_operator.hpp"
namespace duckdb {
class LogicalGet;
class ClientContext;
enum class CaptureMode : uint8_t {
	STREAMING,
	STREAMING_ORDERED,
	MATERIALIZING
};

struct CostParameters {
	idx_t threads = 1;
	double scan_row_column = 0.7;
	double scan_filter_row = 1.8;
	double filter_row = 1.8;
	double projection_row = 0.8;
	double hash_build_row = 10.0;
	double hash_probe_row = 3.0;
	double join_output_row = 1.5;
	double nested_loop_pair = 1.0;
	double aggregate_few_groups = 3.0;
	double aggregate_many_groups = 15.0;
	double aggregate_function = 0.5;
	double sort_row_log = 3.0;
	double top_n_row = 2.0;
	double window_row = 10.0;
	double default_row = 1.0;
	double cached_scan_row = 1.0;
	double cached_scan_byte = 0.05;
};

class CostModel {
public:
	// Estimated CPU cost (ns) of recomputing just this operator, given its children's cardinalities.
	static double OperatorCost(LogicalOperator &op, ClientContext &context, const CostParameters &p);
	// Row count of the table a scan reads, or 0 if unknown.
	static idx_t TableRows(ClientContext &context, LogicalGet &get);
	// Estimated rows an operator produces, falling back on its children when the optimizer left no estimate (it...
	static double EstimatedRows(LogicalOperator &op);
	// Estimated bytes one row of these types takes in a cached result.
	static double RowBytes(const vector<LogicalType> &types);
	// One executed query's measured / estimated wall time.
	void RecordRatio(double measured, double estimated);
	// The learned correction: measured / estimated (geometric mean, clamped).
	double Correction();
	void Reset();
private:
	mutex lock;
	double log_ratio_sum = 0;
	double log_ratio_count = 0;
};
}

