#pragma once

#include "duckdb/common/types.hpp"

namespace duckdb {

// [start, end)
template <class T>
struct Range {
	T start;
	T end;

	static Range Empty() {
		return {0, 0};
	}
	static Range FromStartAndSize(const T start, const idx_t size) {
		return {start, start + static_cast<T>(size)};
	}

	bool Contains(const T value) const {
		return start <= value && value < end;
	}
	bool IsEmpty() const {
		return start >= end;
	}
	idx_t Size() const {
		return static_cast<idx_t>(end - start);
	}
	bool Overlaps(const Range &other) const {
		return start < other.end && other.start < end;
	}
	// Whether the range starts where `other` starts and ends inside `other`.
	bool IsPrefixOf(const Range &other) const {
		return start == other.start && end <= other.end;
	}
	// Whether the range reaches past `point`.
	bool EndsAfter(const T point) const {
		return end > point;
	}
};

using RowRange = Range<row_t>;

} // namespace duckdb
