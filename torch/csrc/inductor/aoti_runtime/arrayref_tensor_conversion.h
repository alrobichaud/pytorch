#pragma once

// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Zero-copy conversion utilities between ArrayRefTensor<T> (C++ template) and
// AOTInductorArrayRefTensor (plain C struct).
//
// These helpers allow the host process to marshal ArrayRefTensor objects into
// the C-compatible AOTInductorArrayRefTensor descriptors before calling into a
// DSO, and to unmarshal the descriptors back after the call.  Because only
// C types cross the DSO boundary, the host and DSO can be linked against
// different C++ standard libraries (e.g. libc++ vs libstdc++) without ABI
// conflicts.
//
// IMPORTANT: Both sides share the same underlying data buffers -- no copies
// are made.  The caller must ensure the data remains valid for the lifetime
// of the descriptor.

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>

#include <cassert>
#include <stdexcept>
#include <type_traits>

namespace torch::aot_inductor {

inline void validate_arrayref_tensor_ndim(int32_t ndim) {
  if (ndim < 0 || ndim > AOTI_ARRAYREF_TENSOR_MAX_DIMS) {
    throw std::runtime_error(
        "AOTInductorArrayRefTensor ndim exceeds AOTI_ARRAYREF_TENSOR_MAX_DIMS");
  }
}

// -------------------------------------------------------------------------
// ArrayRefTensor<T> --> AOTInductorArrayRefTensor  (zero-copy)
// -------------------------------------------------------------------------
template <typename T>
inline void arrayref_tensor_to_c(
    const ArrayRefTensor<T>& src,
    AOTInductorArrayRefTensor& dst) {
  const auto sizes = src.sizes();
  const auto strides = src.strides();
  dst.data = const_cast<void*>(static_cast<const void*>(src.data()));
  dst.numel = static_cast<int64_t>(src.numel());
  dst.ndim = static_cast<int32_t>(sizes.size());
  dst.dtype = aoti_torch_dtype<std::remove_const_t<T>>();
  dst.device_type = src.device_type();
  dst.device_idx = src.device_idx();

  validate_arrayref_tensor_ndim(dst.ndim);
  assert(dst.ndim <= AOTI_ARRAYREF_TENSOR_MAX_DIMS);
  int32_t i = 0;
  for (; i < dst.ndim; ++i) {
    dst.sizes[i] = sizes[i];
    dst.strides[i] = strides[i];
  }
  for (; i < AOTI_ARRAYREF_TENSOR_MAX_DIMS; ++i) {
    dst.sizes[i] = 0;
    dst.strides[i] = 0;
  }
}

template <typename T>
inline AOTInductorArrayRefTensor arrayref_tensor_to_c(
    const ArrayRefTensor<T>& src) {
  AOTInductorArrayRefTensor dst;
  arrayref_tensor_to_c(src, dst);
  return dst;
}

// -------------------------------------------------------------------------
// AOTInductorArrayRefTensor --> ArrayRefTensor<T>  (zero-copy)
// -------------------------------------------------------------------------
template <typename T>
inline ArrayRefTensor<T> c_to_arrayref_tensor(
    const AOTInductorArrayRefTensor& src) {
  validate_arrayref_tensor_ndim(src.ndim);
  return ArrayRefTensor<T>(
      MiniArrayRef<T>(
          static_cast<T*>(const_cast<void*>(src.data)),
          static_cast<size_t>(src.numel)),
      MiniArrayRef<const int64_t>(src.sizes, static_cast<size_t>(src.ndim)),
      MiniArrayRef<const int64_t>(src.strides, static_cast<size_t>(src.ndim)),
      src.device_type,
      src.device_idx);
}

} // namespace torch::aot_inductor
