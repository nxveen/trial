#include "plan_key.hpp"

#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/hash.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace duckdb {

namespace {

//! Every serialized field that holds a table index. They all live in one
//! numbering space (ColumnBinding's "table_index" refers to any of them), so
//! they are renumbered together.
bool IsTableIndexTag(const char *tag) {
	static const char *const TAGS[] = {"table_index",  "group_index",  "aggregate_index", "groupings_index",
	                                   "window_index", "unnest_index", "cte_index",       "pivot_index"};
	for (auto candidate : TAGS) {
		if (strcmp(tag, candidate) == 0) {
			return true;
		}
	}
	return false;
}

//! Fields that do not affect the result but can differ between plans that do
//! the same thing:
//!  - query_location: where the expression stood in the SQL text.
//!  - mark_index: a join's table index for its MARK column. Joins of every
//!    other type leave a leftover value there that nothing reads, and giving
//!    it a canonical number would shift every number after it. A MARK join's
//!    column is still matched through the references to it, which are
//!    renumbered like every other binding.
bool IsIgnoredTag(const char *tag) {
	return strcmp(tag, "query_location") == 0 || strcmp(tag, "mark_index") == 0;
}

//! Writes a compact, unambiguous byte encoding of whatever DuckDB serializes,
//! with table indices renumbered and ignored fields dropped. Only ever used to
//! compare plans, never read back.
class CanonicalPlanSerializer : public Serializer {
public:
	CanonicalPlanSerializer() {
		// Write every field, defaults included. Otherwise a field whose value
		// happens to be the default (table index 0, say) is left out in one
		// plan and present in an otherwise identical one.
		options.serialize_default_values = true;
	}

	string &Result() {
		return out;
	}

	//! Single-pass mode (see SerializeSubtrees): table indices are written as
	//! they are and their positions recorded, and the byte range of every
	//! operator is recorded, in preorder.
	void RecordSubtrees() {
		record = true;
		frames.push_back(Frame {true, nullptr, 0});
	}
	struct IndexPosition {
		idx_t offset;
		idx_t value;
	};
	struct Range {
		idx_t begin;
		idx_t end;
	};
	vector<IndexPosition> positions;
	vector<Range> ranges;

protected:
	void OnPropertyBegin(const field_id_t field_id, const char *tag) override {
		SetProperty(tag);
		BeginTag(tag);
		if (!Skipping()) {
			Byte('P');
			Raw(field_id);
		}
	}
	void OnPropertyEnd() override {
		if (!Skipping()) {
			Byte('p');
		}
		EndTag();
		SetProperty(nullptr);
	}
	void OnOptionalPropertyBegin(const field_id_t field_id, const char *tag, bool present) override {
		SetProperty(tag);
		BeginTag(tag);
		if (!Skipping()) {
			Byte('O');
			Raw(field_id);
			Byte(present ? 1 : 0);
		}
	}
	void OnOptionalPropertyEnd(bool present) override {
		if (!Skipping()) {
			Byte('o');
		}
		EndTag();
		SetProperty(nullptr);
	}
	void OnObjectBegin() override {
		Marker('{');
		if (record) {
			// An operator's children are serialized as the elements of its
			// "children" property; expressions have one too, but only the
			// children of an operator are operators.
			auto &parent = frames.back();
			bool is_operator = parent.is_operator && parent.property && strcmp(parent.property, "children") == 0;
			frames.push_back(Frame {is_operator, nullptr, 0});
			if (is_operator) {
				frames.back().range = ranges.size();
				ranges.push_back(Range {out.size(), 0});
			}
		}
	}
	void OnObjectEnd() override {
		if (record) {
			if (frames.back().is_operator) {
				ranges[frames.back().range].end = out.size();
			}
			frames.pop_back();
		}
		Marker('}');
	}
	void OnListBegin(idx_t count) override {
		if (!Skipping()) {
			Byte('[');
			Raw(count);
		}
	}
	void OnListEnd() override {
		Marker(']');
	}
	void OnNullableBegin(bool present) override {
		if (!Skipping()) {
			Byte('N');
			Byte(present ? 1 : 0);
		}
	}
	void OnNullableEnd() override {
		Marker('n');
	}

	void WriteNull() override {
		Marker('0');
	}
	void WriteValue(char value) override {
		Scalar('c', value);
	}
	void WriteValue(bool value) override {
		Scalar('b', static_cast<uint8_t>(value ? 1 : 0));
	}
	void WriteValue(uint8_t value) override {
		Scalar('B', value);
	}
	void WriteValue(int8_t value) override {
		Scalar('C', value);
	}
	void WriteValue(uint16_t value) override {
		Scalar('W', value);
	}
	void WriteValue(int16_t value) override {
		Scalar('w', value);
	}
	void WriteValue(uint32_t value) override {
		Scalar('I', value);
	}
	void WriteValue(int32_t value) override {
		Scalar('i', value);
	}
	void WriteValue(uint64_t value) override {
		if (!tags.empty() && IsTableIndexTag(tags.back())) {
			if (record) {
				// Renumbered per subtree afterwards; remember where it is.
				if (!Skipping()) {
					positions.push_back(IndexPosition {out.size() + 1, value});
				}
			} else {
				value = Canonical(value);
			}
		}
		Scalar('L', value);
	}
	void WriteValue(int64_t value) override {
		Scalar('l', value);
	}
	void WriteValue(hugeint_t value) override {
		Scalar('H', value);
	}
	void WriteValue(uhugeint_t value) override {
		Scalar('U', value);
	}
	void WriteValue(float value) override {
		Scalar('f', value);
	}
	void WriteValue(double value) override {
		Scalar('d', value);
	}
	void WriteValue(const string_t value) override {
		Bytes(value.GetData(), value.GetSize());
	}
	void WriteValue(const string &value) override {
		Bytes(value.data(), value.size());
	}
	void WriteValue(const char *str) override {
		Bytes(str, strlen(str));
	}
	void WriteDataPtr(const_data_ptr_t ptr, idx_t count) override {
		Bytes(const_char_ptr_cast(ptr), count);
	}

private:
	bool Skipping() const {
		return skip_depth != 0;
	}
	void BeginTag(const char *tag) {
		tags.push_back(tag);
		if (!Skipping() && IsIgnoredTag(tag)) {
			skip_depth = tags.size();
		}
	}
	void EndTag() {
		if (skip_depth == tags.size()) {
			skip_depth = 0;
		}
		tags.pop_back();
	}
	idx_t Canonical(idx_t table_index) {
		auto entry = renumbered.find(table_index);
		if (entry != renumbered.end()) {
			return entry->second;
		}
		auto number = renumbered.size();
		renumbered[table_index] = number;
		return number;
	}
	void Byte(uint8_t byte) {
		out.push_back(static_cast<char>(byte));
	}
	template <class T>
	void Raw(const T &value) {
		out.append(reinterpret_cast<const char *>(&value), sizeof(T));
	}
	void Marker(uint8_t byte) {
		if (!Skipping()) {
			Byte(byte);
		}
	}
	template <class T>
	void Scalar(uint8_t type_tag, const T &value) {
		if (!Skipping()) {
			Byte(type_tag);
			Raw(value);
		}
	}
	void Bytes(const char *data, idx_t size) {
		if (!Skipping()) {
			Byte('s');
			Raw(size);
			out.append(data, size);
		}
	}

	struct Frame {
		bool is_operator;
		const char *property;
		idx_t range;
	};
	void SetProperty(const char *tag) {
		if (record) {
			frames.back().property = tag;
		}
	}

	bool record = false;
	vector<Frame> frames;
	string out;
	vector<const char *> tags;
	//! Depth of the tag stack at which an ignored field began; 0 when none.
	idx_t skip_depth = 0;
	std::unordered_map<idx_t, idx_t> renumbered;
};

} // namespace

bool SerializePlan(const LogicalOperator &op, string &result) {
	try {
		CanonicalPlanSerializer serializer;
		op.Serialize(serializer);
		result = std::move(serializer.Result());
		return true;
	} catch (std::exception &) {
		// Not every operator or function can be serialized. Those plans are
		// simply not cached.
		return false;
	}
}

bool SerializeSubtrees(const LogicalOperator &root, vector<string> &keys) {
	try {
		CanonicalPlanSerializer serializer;
		serializer.RecordSubtrees();
		root.Serialize(serializer);
		auto &out = serializer.Result();
		// The root's range is the whole output; the others were recorded in
		// preorder as their objects began.
		vector<CanonicalPlanSerializer::Range> ranges;
		ranges.push_back(CanonicalPlanSerializer::Range {0, out.size()});
		ranges.insert(ranges.end(), serializer.ranges.begin(), serializer.ranges.end());
		auto &positions = serializer.positions;
		keys.clear();
		keys.reserve(ranges.size());
		for (auto &range : ranges) {
			// A subtree's bytes are exactly what serializing it alone writes,
			// except that its table indices must be numbered from its own root:
			// renumber them in order of first appearance within the range.
			string key = out.substr(range.begin, range.end - range.begin);
			std::unordered_map<idx_t, idx_t> local;
			auto first = std::lower_bound(positions.begin(), positions.end(), range.begin,
			                              [](const CanonicalPlanSerializer::IndexPosition &position, idx_t offset) {
				                              return position.offset < offset;
			                              });
			for (auto position = first; position != positions.end() && position->offset < range.end; ++position) {
				auto entry = local.find(position->value);
				idx_t number = entry != local.end() ? entry->second : local.size();
				if (entry == local.end()) {
					local[position->value] = number;
				}
				memcpy(&key[position->offset - range.begin], &number, sizeof(number));
			}
			keys.push_back(std::move(key));
		}
		return true;
	} catch (std::exception &) {
		return false;
	}
}

hash_t HashPlanKey(const string &key) {
	return Hash(key.c_str(), key.size());
}

} // namespace duckdb

