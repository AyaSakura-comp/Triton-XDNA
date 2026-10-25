// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// See XrtResident.h.

#include "XrtResident.h"

#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include <xrt/experimental/xrt_ext.h>

namespace {

struct Allocation {
  xrt::bo bo;
  size_t nbytes;
  bool is_static;
  // Sub-buffers handed out for views, by (offset, size). A view is usually
  // launched many times, and building a sub-BO is an ioctl.
  std::map<std::pair<size_t, size_t>, std::unique_ptr<xrt::bo>> views;
};

std::mutex &registry_mutex() {
  static std::mutex m;
  return m;
}

// By base address, so the allocation holding an address is the last one at
// or below it.
std::map<uintptr_t, Allocation> &registry() {
  static std::map<uintptr_t, Allocation> r;
  return r;
}

// Caller holds registry_mutex().
Allocation *containing(const void *ptr, size_t nbytes, uintptr_t *base) {
  auto &r = registry();
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  auto it = r.upper_bound(p);
  if (it == r.begin())
    return nullptr;
  --it;
  if (p + nbytes > it->first + it->second.nbytes)
    return nullptr;
  if (base)
    *base = it->first;
  return &it->second;
}

} // namespace

namespace tnpu {

xrt::device &device() {
  // Opened once and never closed: BOs from it outlive any one launcher, and
  // the process exit tears it down after them.
  static xrt::device *d = new xrt::device(0u);
  return *d;
}

bool find(const void *ptr, size_t nbytes, Hit &hit) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  uintptr_t base;
  Allocation *a = containing(ptr, nbytes, &base);
  if (!a)
    return false;
  size_t offset = reinterpret_cast<uintptr_t>(ptr) - base;
  hit.parent = &a->bo;
  hit.offset = offset;
  hit.is_static = a->is_static;
  if (offset == 0) {
    // A prefix view needs no sub-buffer: the kernel only touches the bytes
    // the argument covers, and those start at the BO's base.
    hit.bo = &a->bo;
    return true;
  }
  auto key = std::make_pair(offset, nbytes);
  auto &view = a->views[key];
  if (!view)
    view = std::make_unique<xrt::bo>(a->bo, nbytes, offset);
  hit.bo = view.get();
  return true;
}

} // namespace tnpu

extern "C" {

void *tnpu_alloc(size_t nbytes, int is_static) {
  if (nbytes == 0)
    nbytes = 1;
  try {
    xrt::bo bo = xrt::ext::bo{tnpu::device(), nbytes};
    void *p = bo.map<void *>();
    std::memset(p, 0, nbytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().emplace(reinterpret_cast<uintptr_t>(p),
                       Allocation{std::move(bo), nbytes, is_static != 0, {}});
    return p;
  } catch (const std::exception &) {
    return nullptr;
  }
}

void tnpu_free(void *ptr) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().erase(reinterpret_cast<uintptr_t>(ptr));
}

int tnpu_set_static(void *ptr, int is_static) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  Allocation *a = containing(ptr, 1, nullptr);
  if (!a)
    return -1;
  a->is_static = is_static != 0;
  if (a->is_static)
    a->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return 0;
}

int tnpu_flush(void *ptr, size_t nbytes) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  uintptr_t base;
  Allocation *a = containing(ptr, nbytes, &base);
  if (!a)
    return -1;
  a->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, nbytes,
             reinterpret_cast<uintptr_t>(ptr) - base);
  return 0;
}

size_t tnpu_lookup(const void *ptr, void **base) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  uintptr_t b;
  Allocation *a = containing(ptr, 1, &b);
  if (!a)
    return 0;
  if (base)
    *base = reinterpret_cast<void *>(b);
  return a->nbytes;
}

void tnpu_usage(size_t *bytes, size_t *count) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  size_t total = 0;
  for (auto &[_, a] : registry())
    total += a.nbytes;
  *bytes = total;
  *count = registry().size();
}
}
