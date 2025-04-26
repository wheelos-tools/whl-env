#!/usr/bin/env python

# Copyright 2025 WheelOS All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import platform
import json
from typing import Dict, List, Any, Optional

# Industry best practice is to use official bindings like pynvml when available.
# It's more robust and efficient than parsing command-line output.
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName, nvmlDeviceGetPciInfo, nvmlDeviceGetVbiosVersion,
        nvmlDeviceGetDriverVersion, nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature,
        nvmlDeviceGetPowerUsage, nvmlDeviceGetFanSpeed, nvmlDeviceGetUtilizationRates,
        NVMLError, NVML_TEMPERATURE_GPU
    )
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    # Define a dummy NVMLError for the except block below
    class NVMLError(Exception): pass

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NvidiaGPU:
    """
    Represents a single NVIDIA GPU, providing access to its static specs
    and dynamic state via the NVML library.
    """
    def __init__(self, handle):
        if not PYNVML_AVAILABLE:
            raise ImportError("pynvml library is required for NvidiaGPU class.")
        self._handle = handle

    def get_specs(self) -> Dict[str, Any]:
        """Returns the static specifications of the GPU."""
        pci_info = nvmlDeviceGetPciInfo(self._handle)
        mem_info = nvmlDeviceGetMemoryInfo(self._handle)

        return {
            "vendor": "NVIDIA",
            "model": nvmlDeviceGetName(self._handle),
            "bus_id": pci_info.busId,
            "vbios_version": nvmlDeviceGetVbiosVersion(self._handle),
            "driver_version": nvmlDeviceGetDriverVersion(),
            "total_memory_mib": mem_info.total // (1024**2),
            "source": "pynvml"
        }

    def get_state(self) -> Dict[str, Any]:
        """Returns the current dynamic state of the GPU."""
        try:
            mem_info = nvmlDeviceGetMemoryInfo(self._handle)
            utilization = nvmlDeviceGetUtilizationRates(self._handle)

            return {
                "temperature_celsius": nvmlDeviceGetTemperature(self._handle, NVML_TEMPERATURE_GPU),
                "power_usage_watts": nvmlDeviceGetPowerUsage(self._handle) / 1000.0, # From milliwatts
                "fan_speed_percent": nvmlDeviceGetFanSpeed(self._handle),
                "utilization_percent": {
                    "gpu_core": utilization.gpu,
                    "memory_io": utilization.memory
                },
                "memory_usage": {
                    "used_mib": mem_info.used // (1024**2),
                    "free_mib": mem_info.free // (1024**2)
                }
            }
        except NVMLError as e:
            # Some metrics might not be supported on all cards (e.g., laptops)
            logging.warning(f"Could not retrieve full state for GPU. Error: {e}")
            return {"error": str(e)}

class GPUManager:
    """
    Manages detection and data retrieval for all GPUs in the system.
    Currently focuses on NVIDIA, but is extensible for other vendors.
    """
    def __init__(self):
        self.gpus: List[NvidiaGPU] = []
        self.errors: List[str] = []
        self._discover_gpus()

    def _discover_gpus(self):
        """Discovers GPUs using the best available method."""
        # --- 1. Attempt NVIDIA GPU discovery via pynvml ---
        if PYNVML_AVAILABLE:
            try:
                logging.info("Initializing NVML to discover NVIDIA GPUs...")
                nvmlInit()
                device_count = nvmlDeviceGetCount()
                if device_count > 0:
                    logging.info(f"Found {device_count} NVIDIA GPU(s).")
                    for i in range(device_count):
                        handle = nvmlDeviceGetHandleByIndex(i)
                        self.gpus.append(NvidiaGPU(handle))
                else:
                    logging.info("NVML initialized, but no NVIDIA GPUs were found.")
            except NVMLError as e:
                msg = f"NVML Error during discovery: {e}. Is an NVIDIA driver installed?"
                logging.error(msg)
                self.errors.append(msg)
        else:
            msg = "pynvml library not found. NVIDIA GPU monitoring is disabled."
            logging.warning(msg)
            # Here you could fallback to the original `lspci` and `nvidia-smi` command-line parsing
            # for basic detection if pynvml is not installed. For this example, we keep it clean.
            self.errors.append(msg)

        # --- 2. Future: Add discovery for AMD, Intel GPUs ---
        # e.g., using rocm-smi for AMD or other tools for Intel Arc.

    def get_all_gpu_info(self) -> Dict[str, Any]:
        """
        Retrieves a combined report of specs and current state for all detected GPUs.
        """
        if not self.gpus and not self.errors:
             return {"info": "No supported GPUs detected."}

        gpu_reports = []
        for gpu in self.gpus:
            report = {
                "specs": gpu.get_specs(),
                "state": gpu.get_state()
            }
            gpu_reports.append(report)

        result = {"gpu_list": gpu_reports}
        if self.errors:
            result["errors"] = self.errors

        return result

    def __del__(self):
        """Ensure NVML is shut down cleanly when the manager is destroyed."""
        if PYNVML_AVAILABLE:
            try:
                nvmlShutdown()
                logging.info("NVML shut down successfully.")
            except NVMLError as e:
                logging.error(f"Error shutting down NVML: {e}")


def main():
    """Main function to demonstrate the GPUManager."""
    print("Initializing GPU Manager...")
    manager = GPUManager()

    print("\n--- GPU Information Report (JSON Output) ---")
    full_info = manager.get_all_gpu_info()
    print(json.dumps(full_info, indent=4))

    # Example of processing the detailed results
    if full_info.get('gpu_list'):
        print("\n--- Summary ---")
        for i, gpu_report in enumerate(full_info['gpu_list']):
            specs = gpu_report.get('specs', {})
            state = gpu_report.get('state', {})
            print(f"GPU #{i}: {specs.get('model', 'N/A')}")
            print(f"  Bus ID: {specs.get('bus_id', 'N/A')}")
            print(f"  Driver: {specs.get('driver_version', 'N/A')}")
            print(f"  Memory: {specs.get('total_memory_mib', 'N/A')} MiB")

            if "error" not in state:
                print(f"  Live State:")
                print(f"    - Temp: {state.get('temperature_celsius', 'N/A')}°C")
                print(f"    - Power: {state.get('power_usage_watts', 'N/A')} W")
                print(f"    - Core Usage: {state.get('utilization_percent', {}).get('gpu_core', 'N/A')}%")
            else:
                print(f"  Live State: Error retrieving state - {state.get('error')}")

            print("-" * 20)

    if full_info.get('errors'):
        print("\n--- Errors ---")
        for error in full_info['errors']:
            print(f"- {error}")

    if full_info.get('info'):
        print(f"\n--- Info ---\n{full_info['info']}")


if __name__ == "__main__":
    main()
