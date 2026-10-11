#include "plan_key.hpp"

#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/hash.hpp"

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

protected:
	void OnPropertyBegin(const field_id_t field_id, const char *tag) override {
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
	}
	void OnOptionalPropertyBegin(const field_id_t field_id, const char *tag, bool present) override {
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
	}
	void OnObjectBegin() override {
		Marker('{');
	}
	void OnObjectEnd() override {
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
			value = Canonical(value);
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

hash_t HashPlanKey(const string &key) {
	return Hash(key.c_str(), key.size());
}

} // namespace duckdb
