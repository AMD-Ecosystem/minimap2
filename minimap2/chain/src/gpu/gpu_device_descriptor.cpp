/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * gpu_device_descriptor.cpp
 *
 * Implementation of GPU device descriptor abstraction for multi-GPU support.
 * Creates and validates a homogeneous list of GPU device descriptors, each
 * owning its own HIP memory pool and cached properties.
 */

#include <cstring>
#include <cstdlib>
#include "gpu_device_descriptor.h"
#include "mm_log.h"

/**
 * @brief Check if two devices are homogeneous (same product/model)
 *
 * Homogeneity is defined as: same device_name and same major/minor compute
 * capability. This ensures devices have compatible memory bandwidth, CU
 * counts, and architecture-specific tuning.
 */
static int gpu_devices_homogeneous(const hipDeviceProp_t* prop1, const char* name1,
                                   const hipDeviceProp_t* prop2, const char* name2)
{
	/* Compare device name (product model) */
	if (strcmp(name1, name2) != 0) {
		mm_log_warn("[GPU] Device heterogeneity detected: '%s' vs '%s'", name1, name2);
		return 0;
	}

	/* Compare compute capability */
	if (prop1->major != prop2->major || prop1->minor != prop2->minor) {
		mm_log_warn("[GPU] Device heterogeneity detected: compute capability %d.%d vs %d.%d",
		    prop1->major, prop1->minor, prop2->major, prop2->minor);
		return 0;
	}

	return 1;
}

/**
 * @brief Query device name (product) from device properties
 *
 * Extracts and normalizes the device name for homogeneity checking.
 */
static void gpu_get_device_name(int device_id, char* name_buf, size_t buf_size)
{
	hipDeviceProp_t prop;
	if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
		strncpy(name_buf, prop.name, buf_size - 1);
		name_buf[buf_size - 1] = '\0';
	} else {
		snprintf(name_buf, buf_size, "unknown_device_%d", device_id);
	}
}

/**
 * @brief Get visible device ids from runtime environment
 *
 * If HIP_VISIBLE_DEVICES is set, parse it; otherwise enumerate all visible devices.
 * Returns a newly allocated array that the caller must free.
 */
static int* gpu_get_visible_devices(int* out_count)
{
	int device_count = 0;
	if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
		mm_log_error("[GPU] No visible GPU devices found");
		*out_count = 0;
		return NULL;
	}

	/* TODO: Parse HIP_VISIBLE_DEVICES environment variable if set.
	 * For now, use all visible devices. */

	int* device_ids = (int*)malloc(device_count * sizeof(int));
	if (!device_ids) {
		mm_log_error("[GPU] Memory allocation failed for device list");
		*out_count = 0;
		return NULL;
	}

	for (int i = 0; i < device_count; i++) {
		device_ids[i] = i;
	}

	*out_count = device_count;
	return device_ids;
}

gpu_device_list_t mm_gpu_create_device_list(const int* device_ids, int num_devices)
{
	int actual_count = num_devices;
	int* devices_to_use = NULL;
	int should_free_devices = 0;

	/* Auto-discover if no explicit list provided */
	if (device_ids == NULL || num_devices == 0) {
		devices_to_use = gpu_get_visible_devices(&actual_count);
		should_free_devices = 1;
		if (actual_count <= 0) {
			mm_log_error("[GPU] Failed to auto-discover visible GPU devices");
			return {};
		}
		mm_log_debug("[GPU] Auto-discovered {} devices (no explicit --gpu-devices)", actual_count);
	} else {
		devices_to_use = (int*)device_ids;
		mm_log_debug("[GPU] Using {} explicit device(s)", num_devices);
		for (int i = 0; i < num_devices; i++) {
			mm_log_debug("[GPU]   Device[{}] = {}", i, devices_to_use[i]);
		}
	}

	gpu_device_list_t list;

	/* Query first device properties for homogeneity baseline */
	hipDeviceProp_t baseline_prop;
	char baseline_name[256];
	if (hipSetDevice(devices_to_use[0]) != hipSuccess ||
	    hipGetDeviceProperties(&baseline_prop, devices_to_use[0]) != hipSuccess) {
		mm_log_error("[GPU] Failed to query properties for device %d", devices_to_use[0]);
		if (should_free_devices) free(devices_to_use);
		return {};
	}
	gpu_get_device_name(devices_to_use[0], baseline_name, sizeof(baseline_name));

	/* Create device descriptor for each device */
	for (int i = 0; i < actual_count; i++) {
		int dev_id = devices_to_use[i];

		if (hipSetDevice(dev_id) != hipSuccess) {
			mm_log_error("[GPU] Failed to set device %d", dev_id);
			if (should_free_devices) free(devices_to_use);
			return {};
		}

		hipDeviceProp_t prop;
		if (hipGetDeviceProperties(&prop, dev_id) != hipSuccess) {
			mm_log_error("[GPU] Failed to query properties for device %d", dev_id);
			if (should_free_devices) free(devices_to_use);
			return {};
		}

		char dev_name[256];
		gpu_get_device_name(dev_id, dev_name, sizeof(dev_name));
		mm_log_debug("[GPU] Device {}: initialized HIP device_id={}, product={}, compute_cap={}.{}", i, dev_id, dev_name, prop.major, prop.minor);

		if (i > 0 && !gpu_devices_homogeneous(&baseline_prop, baseline_name, &prop, dev_name)) {
			mm_log_error("[GPU] Heterogeneous GPU list not supported. "
			             "Use --gpu-devices or HIP_VISIBLE_DEVICES to select a homogeneous set.");
			if (should_free_devices) free(devices_to_use);
			return {};
		}

		hipMemPool_t mempool = NULL;
		if (prop.memoryPoolsSupported) {
			if (hipDeviceGetDefaultMemPool(&mempool, dev_id) != hipSuccess) {
				mm_log_error("[GPU] Failed to get mempool for device %d", dev_id);
				if (should_free_devices) free(devices_to_use);
				return {};
			}
			uint64_t threshold = UINT64_MAX;
			if (hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &threshold) != hipSuccess) {
				mm_log_warn("[GPU] Failed to set mempool threshold for device %d (non-critical)", dev_id);
			}
		}

		auto device = std::make_shared<GPU_device>();
		device->device_id = dev_id;
		device->mempool = mempool;
		device->mempool_owned = true;
		memcpy(&device->prop, &prop, sizeof(hipDeviceProp_t));
		strncpy(device->product_name, dev_name, sizeof(device->product_name) - 1);
		device->product_name[sizeof(device->product_name) - 1] = '\0';
		list.push_back(std::move(device));
	}

	mm_log_debug("[GPU] Created device list with {} {} devices", actual_count, baseline_name);

	if (should_free_devices) free(devices_to_use);
	return list;
}

int mm_gpu_visible_device_count(void)
{
	int device_count = 0;
	if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0)
		return 0;
	return device_count;
}
