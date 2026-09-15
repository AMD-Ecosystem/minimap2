/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef GPU_DEVICE_LIST_FWD_H
#define GPU_DEVICE_LIST_FWD_H

#include <memory>
#include <vector>

struct GPU_device;
using gpu_device_list_t = std::vector<std::shared_ptr<GPU_device>>;

/* HIP-free declaration — implementation in gpu_device_descriptor.cpp (HIP TU). */
gpu_device_list_t mm_gpu_create_device_list(const int* device_ids, int num_devices);

#endif // GPU_DEVICE_LIST_FWD_H
