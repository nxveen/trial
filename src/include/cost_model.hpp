#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class ClientContext;

//! How a result is collected while the query runs. The choice changes what
//! keeping it costs, by an order of magnitude.
enum class CaptureMode : uint8_t {
	//! Copies rows as they stream past, on every thread, with no pipeline
	//! break: only safe where exactly one pipeline flows through the point and
	//! nothing after it can stop that pipeline early.
	STREAMING,
	//! The same, but on one thread so the rows keep their order. Only used at
	//! the top of a query that ends in an ORDER BY, where the last pipeline
	//! just reads out an already sorted result.
	STREAMING_ORDERED,
	//! A pipeline breaker: collects everything, copies it out of the buffer
	//! manager on one thread, then scans it back out. Works everywhere.
	MATERIALIZING
};

//! The knobs of the cost model.
//!
//! Costs are in nanoseconds of CPU work, calibrated single-threaded on DuckDB
//! 1.5.5; dividing by the thread count turns them into wall-clock time.
struct CostParameters {
	idx_t threads = 1;

	//! Recomputing a subtree.
	double scan_row_column = 0.7; //!< per base-table row, per column read (0.8 ns measured)
	double scan_filter_row = 1.8; //!< per base-table row, per operation in pushed-down filters
	double filter_row = 1.8;      //!< per input row, per operation in the predicates (~1.9 ns measured)
	double projection_row = 0.8;  //!< per input row, per operation in the expressions (~0.75 ns measured)
	double hash_build_row = 10.0;
	double hash_probe_row = 3.0; //!< probe + output measured at ~4.4 ns per probe row
	double join_output_row = 1.5;
	double nested_loop_pair = 1.0;       //!< per left x right pair (cross product, any-join)
	double aggregate_few_groups = 3.0;   //!< per input row, up to ~1000 groups (3 ns measured)
	double aggregate_many_groups = 15.0; //!< per input row, 100k+ groups (15 ns measured)
	double aggregate_function = 0.5;     //!< per input row, per aggregate function
	double sort_row_log = 3.0;           //!< per row, times log2(rows) (3.3 ns measured)
	double top_n_row = 2.0;
	double window_row = 10.0;
	double default_row = 1.0;

	//! Reading a cached result back out (only used to estimate the true cost
	//! of a node above a cached one before it has been measured).
	double cached_scan_row = 1.0;
	double cached_scan_byte = 0.05;
};

//! Estimates of what the paper's recycler measures: the base cost and the size
//! of a result (Section III-C). The recycler uses them only until it has
//! measured the real values, and for the speculative decisions of Section
//! III-D, which by definition concern results it has never seen.
//!
//! Estimates come from the optimizer's cardinalities, which can be off by an
//! order of magnitude, so the model also learns a correction factor: the
//! geometric mean of measured over estimated time for executed queries.
class CostModel {
public:
	//! Estimated CPU cost (ns) of recomputing just this operator, given its
	//! children's cardinalities. A subtree's cost is the sum over its operators.
	static double OperatorCost(LogicalOperator &op, ClientContext &context, const CostParameters &p);
	//! Estimated rows an operator produces, falling back on its children when
	//! the optimizer left no estimate (it does not estimate operators that
	//! later optimizer passes add, such as compressed-materialization projections).
	static double EstimatedRows(LogicalOperator &op);
	//! Estimated bytes one row of these types takes in a cached result.
	static double RowBytes(const vector<LogicalType> &types);

	//! One executed query's measured / estimated wall time.
	void RecordRatio(double measured, double estimated);
	//! The learned correction: measured / estimated (geometric mean, clamped).
	double Correction();
	void Reset();

private:
	mutex lock;
	double log_ratio_sum = 0;
	double log_ratio_count = 0;
};

} // namespace duckdb
