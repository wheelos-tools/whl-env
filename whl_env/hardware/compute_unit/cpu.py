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
import time
import json
from typing import Dict, Any, List, Optional, NamedTuple

try:
    import psutil
except ImportError:
    print("Error: The 'psutil' library is required. Please install it using 'pip install psutil'")
    exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CPUTimes(NamedTuple):
    """Represents CPU time statistics."""
    user: float
    system: float
    idle: float
    interrupt: Optional[float]
    dpc: Optional[float]

class CPU:
    """
    A class to represent and retrieve comprehensive CPU information,
    including specifications, real-time frequency, temperature, and load.

    This implementation prefers `psutil` for its reliability and cross-platform
    compatibility, falling back to manual parsing for specific details if needed.
    """
    def __init__(self):
        self._physical_cores: int = psutil.cpu_count(logical=False)
        self._logical_cores: int = psutil.cpu_count(logical=True)
        # Note: psutil does not directly expose socket count. We'll rely on
        # parsing /proc/cpuinfo for that specific, static detail.
        self._sockets: int = self._get_socket_count()
        self._model_name: str = self._get_cpu_model_name()

    @staticmethod
    def _get_cpu_model_name() -> str:
        """
        Retrieves the CPU model name, trying various sources for robustness.
        This is a static piece of information, so it's fine to get it once.
        """
        # In Linux, /proc/cpuinfo is the most reliable source for the model name.
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(':')[1].strip()
            except Exception as e:
                logging.warning(f"Could not read model name from /proc/cpuinfo: {e}")

        # Fallback for other systems or if /proc/cpuinfo fails
        # A more comprehensive solution might use `platform.processor()` but it can be less descriptive.
        return "Unknown"

    @staticmethod
    def _get_socket_count() -> int:
        """
        Parses /proc/cpuinfo to determine the number of physical sockets.
        This is one detail not easily available in psutil.
        """
        if platform.system() != "Linux":
            logging.info("Socket count detection is only supported on Linux.")
            return 1 # Assume 1 socket on non-Linux systems for basic functionality.

        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()

            physical_ids = set()
            for line in content.splitlines():
                if line.startswith("physical id"):
                    physical_id = line.split(':')[1].strip()
                    physical_ids.add(physical_id)

            # If `physical id` is not found, assume a single-socket system.
            return len(physical_ids) if physical_ids else 1
        except FileNotFoundError:
            logging.error("/proc/cpuinfo not found. Cannot determine socket count.")
            return 1 # Fallback
        except Exception as e:
            logging.error(f"Error parsing /proc/cpuinfo for socket count: {e}")
            return 1 # Fallback

    def get_specs(self) -> Dict[str, Any]:
        """
        Returns the static specifications of the CPU.
        """
        return {
            "model_name": self._model_name,
            "architecture": platform.machine(),
            "physical_cores": self._physical_cores,
            "logical_cores": self._logical_cores,
            "sockets": self._sockets,
            "physical_cores_per_socket": self._physical_cores // self._sockets if self._sockets > 0 else 0
        }

    @staticmethod
    def get_frequency() -> Dict[str, Optional[float]]:
        """
        Gets current, min, and max CPU frequency in MHz.

        Returns:
            A dictionary with 'current', 'min', and 'max' frequency.
            Values can be None if the information is not available.
        """
        try:
            freq = psutil.cpu_freq()
            return {
                "current_mhz": freq.current,
                "min_mhz": freq.min,
                "max_mhz": freq.max
            }
        except (AttributeError, NotImplementedError):
            logging.warning("CPU frequency information not available on this system via psutil.")
            return {"current_mhz": None, "min_mhz": None, "max_mhz": None}

    @staticmethod
    def get_temperature() -> Dict[str, Any]:
        """
        Gets CPU temperatures from available sensors.

        Returns:
            A dictionary where keys are sensor labels and values are a list of
            current, high, and critical temperatures in Celsius.
            Returns a dictionary with an error note if not supported.
        """
        if not hasattr(psutil, "sensors_temperatures"):
            return {"error": "Temperature sensing not supported on this platform by psutil."}

        temps = psutil.sensors_temperatures()
        if not temps:
            return {"error": "No temperature sensors found."}

        # The key for CPU temps varies, e.g., 'coretemp' on Intel, 'k10temp' on AMD.
        # We can look for common ones or return all available. Let's find the CPU-related ones.
        cpu_temps = {}
        # Common keys for CPU temperature sensors in Linux
        cpu_sensor_keys = ['coretemp', 'k10temp', 'zenpower', 'cpu_thermal']

        for name, measurements in temps.items():
            if any(key in name.lower() for key in cpu_sensor_keys):
                # shwtemp(label='', current=23.0, high=82.0, critical=82.0)
                cpu_temps[name] = [
                    {
                        "label": m.label or f"Core {i}",
                        "current_celsius": m.current,
                        "high_celsius": m.high,
                        "critical_celsius": m.critical
                    } for i, m in enumerate(measurements)
                ]

        if not cpu_temps:
             return {"error": "Could not identify a specific CPU temperature sensor.", "all_sensors": temps}

        return cpu_temps

    @staticmethod
    def get_load() -> Dict[str, Any]:
        """
        Calculates overall and per-core CPU load percentage over a short interval.
        Also retrieves system load average (on UNIX-like systems).

        Returns:
            A dictionary containing total usage, per-core usage, and load averages.
        """
        # The interval is crucial for an accurate reading.
        # A non-blocking call (interval=None) compares against the last call time.
        # A blocking call (e.g., interval=1) is simpler and often preferred for scripts.
        overall_percent = psutil.cpu_percent(interval=1, percpu=False)
        per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True) # Non-blocking after the first

        load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (None, None, None)

        return {
            "overall_percent": overall_percent,
            "per_cpu_percent": per_cpu_percent,
            "load_average": {
                "1_min": load_avg[0],
                "5_min": load_avg[1],
                "15_min": load_avg[2]
            }
        }

def main():
    """Main function to demonstrate the CPU class."""
    print("Initializing CPU monitor...")
    cpu = CPU()

    print("\n--- 1. CPU Specifications ---")
    specs = cpu.get_specs()
    print(json.dumps(specs, indent=4))

    print("\n--- 2. CPU Frequency (Real-time) ---")
    frequency = cpu.get_frequency()
    print(json.dumps(frequency, indent=4))

    print("\n--- 3. CPU Temperature ---")
    # Note: Requires root/admin privileges on some systems.
    # May return an error if not supported or no sensors are found.
    temperature = cpu.get_temperature()
    print(json.dumps(temperature, indent=4))

    print("\n--- 4. CPU Load (measuring over 1 second) ---")
    load = cpu.get_load()
    print(json.dumps(load, indent=4))

if __name__ == "__main__":
    main()
