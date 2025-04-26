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
import json
from typing import Dict, Any, Optional

# psutil is the industry-standard, cross-platform library for this task.
try:
    import psutil
except ImportError:
    print("Error: The 'psutil' library is required. Please install it with 'pip install psutil'")
    exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Memory:
    """
    A class to retrieve comprehensive memory information, focusing on usage,
    performance metrics, and health status, without requiring special privileges.

    This implementation relies exclusively on cross-platform libraries and standard
    OS interfaces, making it robust and secure for general-purpose monitoring.
    """
    def __init__(self):
        """Initializes the Memory monitor."""
        # You can add one-time initializations here if needed.
        logging.info("Memory monitor initialized. Ready to get usage data.")

    def get_usage_and_health(self) -> Dict[str, Any]:
        """
        Retrieves system-wide memory usage statistics using psutil.
        This includes physical memory (RAM) and swap space, along with a health assessment.

        Returns:
            A dictionary containing detailed usage and health metrics.
        """
        try:
            # psutil's virtual_memory() returns a named tuple with bytes as units.
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # 'available' is what applications can use without causing swapping.
            # (total - available) is a more accurate metric for "used" by applications.
            app_used_memory = vm.total - vm.available

            usage_info = {
                "physical_memory": {
                    "total_gb": round(vm.total / (1024**3), 2),
                    "available_gb": round(vm.available / (1024**3), 2),
                    "used_gb": round(app_used_memory / (1024**3), 2),
                    "percent_used": round((app_used_memory / vm.total) * 100, 2),
                    # Provide details on how memory is allocated internally
                    # Use getattr for robustness if some fields are not present on all systems/versions
                    "internal_usage_gb": {
                        "buffers": round(getattr(vm, 'buffers', 0) / (1024**3), 2),
                        "cached": round(getattr(vm, 'cached', 0) / (1024**3), 2),
                        "shared": round(getattr(vm, 'shared', 0) / (1024**3), 2),
                    }
                },
                "swap_memory": {
                    "total_gb": round(swap.total / (1024**3), 2),
                    "used_gb": round(swap.used / (1024**3), 2),
                    "percent_used": swap.percent,
                },
                "health": {
                    "status": self._assess_health(vm, swap), # vm and swap are psutil.svmem and psutil.sswap NamedTuple instances
                    "notes": "Health status is based on memory pressure. High usage of swap or very low available memory indicates potential performance issues."
                },
                "physical_specs": {
                    "info": "Retrieving physical memory details (type, speed, manufacturer) requires tools like 'dmidecode' and administrative privileges, which are not used in this function."
                }
            }
            return usage_info
        except Exception as e:
            logging.error(f"Failed to get memory usage via psutil: {e}")
            return {"error": f"Failed to get memory usage: {e}"}

    @staticmethod
    def _assess_health(vm: Any, swap: Any) -> str: # <--- CHANGED HERE
        """
        Provides a simple health assessment based on memory pressure.
        This logic can be customized based on specific workload requirements.
        """
        # Calculate available memory percentage, which is a key health indicator.
        available_percent = (vm.available / vm.total) * 100

        if available_percent < 5 or swap.percent > 50:
            return "CRITICAL"
        elif available_percent < 15 or swap.percent > 25:
            return "WARNING"
        return "OK"


def main():
    """Main function to demonstrate the Memory class."""
    print("Initializing Memory monitor (no sudo required)...")
    memory_monitor = Memory()

    print("\n--- Memory Usage and Health Report ---")
    report = memory_monitor.get_usage_and_health()
    print(json.dumps(report, indent=4))

    # Example of processing the detailed results
    if "error" not in report:
        print("\n--- Summary ---")
        phys_mem = report.get("physical_memory", {})
        health = report.get("health", {})

        print(f"Physical Memory: {phys_mem.get('used_gb', 'N/A')} GB Used / {phys_mem.get('total_gb', 'N/A')} GB Total ({phys_mem.get('percent_used', 'N/A')}%)")
        print(f"Available for Apps: {phys_mem.get('available_gb', 'N/A')} GB")
        print(f"Health Status: {health.get('status', 'N/A')}")

        swap_mem = report.get("swap_memory", {})
        if swap_mem.get("total_gb", 0) > 0:
            print(f"Swap Usage: {swap_mem.get('used_gb', 'N/A')} GB Used / {swap_mem.get('total_gb', 'N/A')} GB Total ({swap_mem.get('percent_used', 'N/A')}%)")

if __name__ == "__main__":
    main()
