// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// NPU-resident host memory for the XRT ELF launchers.
//
// An ordinary XRT launch stages every pointer argument: it copies the tensor
// into a buffer object, syncs the whole BO, runs, then syncs and copies the
// result back. For a model that is most of the time a launch takes -- weights
// copied in on every call, an int32 output copied in and out.
//
// Memory allocated here is a BO from the start, so a launch can name it
// directly. libtriton_npu_xrt_resident.so holds one process-wide registry of
// those allocations, keyed by host address, and every generated ELF launcher
// links it. When a pointer argument lands inside a registered allocation the
// launcher passes that BO (or a sub-buffer, for a view that does not start at
// the allocation's base), syncs only the bytes the argument covers, and copies
// nothing. Arguments outside the registry take the staging path as before.
//
// An allocation marked static is synced to the device when it is marked and
// then left alone: it holds weights the host does not write. If the host does
// write it, tnpu_flush() pushes the change.
//
// The device is shared too: launchers open it through tnpu::device(), so one
// xrt::device backs every BO and every hw_context in the process.

#pragma once

#include <cstddef>
#include <cstdint>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"

namespace tnpu {

// The process-wide device, opened on first use.
xrt::device &device();

// Where a pointer argument lives. `bo` is the BO to pass to the kernel: the
// allocation itself when the argument starts at its base, otherwise a cached
// sub-buffer starting at the argument. Sync calls go through `parent` with
// `offset`, which is the argument's byte offset in the allocation.
struct Hit {
  xrt::bo *bo = nullptr;
  xrt::bo *parent = nullptr;
  size_t offset = 0;
  bool is_static = false;
  // Written by the NPU only: the host does not write it, so a launch need not
  // flush it before the run.
  bool device_writes = false;
  // Whether a launch syncs it back after the run. Off when the host
  // invalidates the lines itself before reading them (in parallel, say).
  bool sync_back = true;
};

// Look up [ptr, ptr + nbytes). True when the whole range lies inside one
// registered allocation; `hit` is then filled in. Thread-safe.
bool find(const void *ptr, size_t nbytes, Hit &hit);

} // namespace tnpu

extern "C" {
// Allocate `nbytes` of NPU-resident memory and return its host address, or
// null on failure. The memory is page aligned and zero filled.
void *tnpu_alloc(size_t nbytes, int is_static);
// Release an allocation returned by tnpu_alloc. Unknown pointers are ignored.
void tnpu_free(void *ptr);
// Change whether an allocation is static. Marking one static syncs it.
int tnpu_set_static(void *ptr, int is_static);
int tnpu_set_device_writes(void *ptr, int device_writes, int sync_back);
// Push host writes in [ptr, ptr + nbytes) to the device.
int tnpu_flush(void *ptr, size_t nbytes);
// Base address and size of the allocation containing ptr; 0 if none.
size_t tnpu_lookup(const void *ptr, void **base);
// Bytes currently allocated, and the number of allocations.
void tnpu_usage(size_t *bytes, size_t *count);
}
