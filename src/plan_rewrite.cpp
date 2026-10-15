#include "plan_rewrite.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include <map>
namespace duckdb {
namespace {
class BindingRemapper : public LogicalOperatorVisitor {
public:
	BindingRemapper(const vector<ColumnBinding> &old_bindings, idx_t new_table_index) {
		for (idx_t i = 0; i < old_bindings.size(); i++) {
			auto &old_binding = old_bindings[i];
			mapping[std::make_pair(old_binding.table_index, old_binding.column_index)] = i;
		}
		table_index = new_table_index;
	}
protected:
	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		auto entry = mapping.find(std::make_pair(expr.binding.table_index, expr.binding.column_index));
		if (entry != mapping.end()) {
			expr.binding = ColumnBinding(table_index, entry->second);
		}
		return nullptr;
	}
private:
	std::map<std::pair<idx_t, idx_t>, idx_t> mapping;
	idx_t table_index;
};
}
void RemapColumnBindings(LogicalOperator &root, const vector<ColumnBinding> &old_bindings, idx_t new_table_index) {
	// Visiting the whole plan is safe: the retired table indices exist nowhere else, so the only references the...
	BindingRemapper remapper(old_bindings, new_table_index);
	remapper.VisitOperator(root);
}
}

