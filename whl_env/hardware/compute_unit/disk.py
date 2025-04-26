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

"""
Storage Self-Check Module for Autonomous Driving Systems.

This module provides a comprehensive check of storage devices (HDDs/SSDs)
to ensure their health, performance, and reliability, which are critical for
safe and robust autonomous vehicle operations.

It assesses:
- Device connectivity and identification (lsblk).
- Filesystem available space (df).
- Real-time I/O performance (throughput and latency via psutil).
- Hardware health and estimated lifetime (S.M.A.R.T. via smartctl).

The final output is a structured report and a clear, actionable overall
health status for the entire storage subsystem.
"""

import logging
import re
import time
import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# --- Dependency Check ---
try:
    import psutil
except ImportError:
    logging.critical("CRITICAL: The 'psutil' library is required. System health monitoring cannot proceed.")
    # In a real system, this would trigger a major fault.
    raise

# --- Module Configuration ---
# Configure basic logging for standalone execution. In a larger system,
# this would be handled by a central logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)

# Constants
FSTAB_PATH = '/etc/fstab'
SMARTCTL_TIMEOUT = 20  # Increased timeout for potentially slow S.M.A.R.T. commands
COMMAND_TIMEOUT = 10   # General command timeout

# --- Data Models (Dataclasses for Clarity and Type Safety) ---

@dataclass
class FilesystemUsage:
    """Represents usage data for a mounted filesystem."""
    device: str
    mount_point: str
    total_bytes: int
    used_bytes: int
    available_bytes: int
    use_percentage: float

@dataclass
class FstabEntry:
    """Represents a single entry from /etc/fstab."""
    device: str
    mount_point: str
    fstype: str
    options: str
    dump: str
    pass_no: str

@dataclass
class IOStats:
    """Represents real-time I/O statistics for a device."""
    device_name: str
    read_bytes_per_sec: float
    write_bytes_per_sec: float
    read_ops_per_sec: float
    write_ops_per_sec: float
    read_latency_ms_per_op: float
    write_latency_ms_per_op: float

@dataclass
class SmartInfo:
    """Represents S.M.A.R.T. health information for a disk."""
    health_status: str  # PASSED, FAILED, WARNING
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    lifetime: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None
    serial_number: Optional[str] = None
    temperature_celsius: Optional[int] = None
    critical_issues: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class Partition:
    """Represents a disk partition."""
    name: str
    full_path: str
    type: str
    size_bytes: int
    size_gb: float
    mount_point: Optional[str] = None
    filesystem_type: Optional[str] = None
    usage: Optional[FilesystemUsage] = None
    fstab_entry: Optional[FstabEntry] = None
    io_stats: Optional[IOStats] = None
    note: Optional[str] = None

@dataclass
class Disk:
    """Represents a physical storage disk."""
    name: str
    full_path: str
    type: str
    size_bytes: int
    size_gb: float
    model: str
    vendor: str
    is_rotational: bool
    partitions: List[Partition] = field(default_factory=list)
    io_stats: Optional[IOStats] = None
    smart_info: Optional[SmartInfo] = None
    mount_point: Optional[str] = None  # For disks mounted directly

@dataclass
class StorageHealthReport:
    """The final, consolidated report."""
    overall_status: str  # HEALTHY, WARNING, CRITICAL
    disks: List[Disk]
    other_devices: List[Partition] # For loop devices, LVMs, etc.
    summary: Dict[str, Any]
    errors: List[str]

# --- Main Class for Storage Self-Check ---

class StorageManager:
    """
    Performs a comprehensive self-check of the vehicle's storage subsystem.
    """
    def __init__(self):
        self._last_disk_io_counters: Dict[str, psutil._common.sdiskio] = {}
        self._last_disk_io_timestamp: float = time.monotonic()
        # Initialize with first sample to enable rate calculation on the first real check.
        self.get_disk_io_stats()
        logging.info("StorageManager initialized and first I/O sample taken.")

    def _run_command(self, cmd: List[str], timeout: int, allow_sudo: bool = False) -> Tuple[str, str, int]:
        """A robust, unified command execution wrapper."""
        if cmd[0] == 'sudo' and not allow_sudo:
            err_msg = f"Command with 'sudo' is not permitted: {' '.join(cmd)}"
            logging.error(err_msg)
            return "", err_msg, -1

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
            return process.stdout.strip(), process.stderr.strip(), process.returncode
        except FileNotFoundError:
            return "", f"Command not found: '{cmd[0]}'", 127
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s: {' '.join(cmd)}", -2
        except Exception as e:
            return "", f"Unexpected error running command: {e}", -3

    def get_block_devices(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Retrieves block device info using 'lsblk'."""
        devices, errors = [], []
        cmd = ['lsblk', '-b', '-P', '-o', 'NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,MODEL,VENDOR,ROTA']
        output, error, code = self._run_command(cmd, COMMAND_TIMEOUT)

        if code == 0 and output:
            for line in output.strip().split('\n'):
                dev_raw = {k: v for k, v in re.findall(r'(\w+)="([^"]*)"', line)}
                if not dev_raw: continue

                try:
                    size_bytes = int(dev_raw.get('SIZE', 0))
                    devices.append({
                        'name': dev_raw.get('NAME', 'N/A'),
                        'full_path': f"/dev/{dev_raw.get('NAME', '')}",
                        'type': dev_raw.get('TYPE', 'unknown'),
                        'size_bytes': size_bytes,
                        'size_gb': round(size_bytes / (1024**3), 2),
                        'mount_point': dev_raw.get('MOUNTPOINT', '').strip() or None,
                        'filesystem_type': dev_raw.get('FSTYPE', '').strip() or None,
                        'model': dev_raw.get('MODEL', 'N/A').strip(),
                        'vendor': dev_raw.get('VENDOR', 'N/A').strip(),
                        'is_rotational': dev_raw.get('ROTA') == '1',
                    })
                except (ValueError, TypeError) as e:
                    errors.append(f"Failed to parse lsblk line '{line}': {e}")
        elif code != 0:
            errors.append(f"lsblk command failed (code {code}): {error}")

        return devices, errors

    def get_filesystem_usage(self) -> Tuple[List[FilesystemUsage], List[str]]:
        """Retrieves filesystem usage using 'df'."""
        filesystems, errors = [], []
        # Using psutil.disk_partitions() and psutil.disk_usage() is more robust and portable
        try:
            partitions = psutil.disk_partitions()
            for part in partitions:
                if 'loop' in part.device: continue # Often not relevant for autoware
                usage = psutil.disk_usage(part.mountpoint)
                filesystems.append(FilesystemUsage(
                    device=part.device,
                    mount_point=part.mountpoint,
                    total_bytes=usage.total,
                    used_bytes=usage.used,
                    available_bytes=usage.free,
                    use_percentage=usage.percent
                ))
        except Exception as e:
            errors.append(f"Failed to get filesystem usage via psutil: {e}")
        return filesystems, errors

    def get_disk_io_stats(self) -> Dict[str, IOStats]:
        """Calculates real-time disk I/O statistics using psutil."""
        io_stats_map = {}
        try:
            current_counters = psutil.disk_io_counters(perdisk=True)
            current_timestamp = time.monotonic()
            time_delta = current_timestamp - self._last_disk_io_timestamp

            if time_delta > 0:
                for name, stats in current_counters.items():
                    if name in self._last_disk_io_counters:
                        last_stats = self._last_disk_io_counters[name]
                        read_bytes_delta = stats.read_bytes - last_stats.read_bytes
                        write_bytes_delta = stats.write_bytes - last_stats.write_bytes
                        read_count_delta = stats.read_count - last_stats.read_count
                        write_count_delta = stats.write_count - last_stats.write_count
                        read_time_delta = stats.read_time - last_stats.read_time
                        write_time_delta = stats.write_time - last_stats.write_time

                        io_stats_map[name] = IOStats(
                            device_name=name,
                            read_bytes_per_sec=read_bytes_delta / time_delta,
                            write_bytes_per_sec=write_bytes_delta / time_delta,
                            read_ops_per_sec=read_count_delta / time_delta,
                            write_ops_per_sec=write_count_delta / time_delta,
                            read_latency_ms_per_op=read_time_delta / read_count_delta if read_count_delta > 0 else 0,
                            write_latency_ms_per_op=write_time_delta / write_count_delta if write_count_delta > 0 else 0,
                        )

            self._last_disk_io_counters = current_counters
            self._last_disk_io_timestamp = current_timestamp
        except Exception as e:
            logging.error(f"Failed to get disk I/O stats: {e}")
        return io_stats_map

    def get_smart_info(self, device_path: str) -> SmartInfo:
        """Retrieves and parses S.M.A.R.T. data using 'smartctl'."""
        cmd = ['sudo', 'smartctl', '-a', '-j', device_path]
        output, error, code = self._run_command(cmd, SMARTCTL_TIMEOUT, allow_sudo=True)

        if code != 0 or not output:
            return SmartInfo(health_status="UNKNOWN", error=f"smartctl failed (code {code}): {error}")

        try:
            data = json.loads(output)
            passed = data.get('smart_status', {}).get('passed', False)
            health_status = "PASSED" if passed else "FAILED"

            # Parse key lifetime and identity attributes
            lifetime = {
                'power_on_hours': data.get('power_on_time', {}).get('hours'),
                'power_cycle_count': data.get('power_cycle_count'),
                'data_units_written': data.get('nvme_smart_health_information_log', {}).get('data_units_written'),
                'percentage_used': data.get('nvme_smart_health_information_log', {}).get('percentage_used'),
            }
            # Remove None values for cleaner output
            lifetime = {k: v for k, v in lifetime.items() if v is not None}

            # Parse attributes and identify critical issues
            attributes = data.get('ata_smart_attributes', {}).get('table', [])
            critical_issues = []
            critical_attr_names = {
                'Reallocated_Sector_Ct', 'Current_Pending_Sector_Ct',
                'Offline_Uncorrectable', 'Reported_Uncorrect'
            }
            for attr in attributes:
                if attr.get('name') in critical_attr_names and attr.get('raw', {}).get('value', 0) > 0:
                    critical_issues.append(f"{attr['name']} is {attr['raw']['value']}")

            # For NVMe, check critical warnings from the log
            nvme_log = data.get('nvme_smart_health_information_log', {})
            if nvme_log.get('critical_warning', 0) > 0:
                 critical_issues.append(f"NVMe Critical Warning Flags: {nvme_log['critical_warning']}")

            if critical_issues and health_status == "PASSED":
                health_status = "WARNING"

            return SmartInfo(
                health_status=health_status,
                attributes=attributes,
                lifetime=lifetime,
                model_name=data.get('model_name'),
                serial_number=data.get('serial_number'),
                temperature_celsius=data.get('temperature', {}).get('current'),
                critical_issues=critical_issues
            )
        except json.JSONDecodeError:
            return SmartInfo(health_status="UNKNOWN", error="Failed to parse smartctl JSON output.")
        except Exception as e:
            return SmartInfo(health_status="UNKNOWN", error=f"Unexpected error parsing SMART data: {e}")

    @staticmethod
    def _get_parent_disk_name(partition_name: str) -> Optional[str]:
        """Deduces parent disk name from a partition name (e.g., sda1 -> sda)."""
        match = re.match(r'^(nvme\d+n\d+)p\d+', partition_name) or \
                re.match(r'^([a-z]+)\d+', partition_name)
        return match.group(1) if match else None

    def run_self_check(self) -> StorageHealthReport:
        """
        Executes the full storage self-check and returns a comprehensive report.
        """
        logging.info("Starting storage self-check...")
        all_errors = []

        # 1. Gather all data
        block_devices_raw, errors = self.get_block_devices()
        all_errors.extend(errors)

        fs_usage, errors = self.get_filesystem_usage()
        all_errors.extend(errors)

        io_stats = self.get_disk_io_stats()

        # 2. Build device hierarchy
        disks: List[Disk] = []
        other_devices: List[Partition] = []
        partitions_map: Dict[str, List[Partition]] = {}

        # Create Disk objects and map for their partitions
        for dev_raw in block_devices_raw:
            if dev_raw['type'] == 'disk':
                disk = Disk(**{k: v for k, v in dev_raw.items() if k in Disk.__annotations__})
                disks.append(disk)
                partitions_map[disk.name] = disk.partitions

        # Create Partition objects and assign them
        for dev_raw in block_devices_raw:
            if dev_raw['type'] == 'part':
                part = Partition(**{k: v for k, v in dev_raw.items() if k in Partition.__annotations__})
                parent_name = self._get_parent_disk_name(part.name)
                if parent_name and parent_name in partitions_map:
                    partitions_map[parent_name].append(part)
                else:
                    part.note = "Parent disk not recognized."
                    other_devices.append(part)
            elif dev_raw['type'] not in ['disk', 'part']:
                # Handle loop, lvm, etc.
                dev = Partition(**{k: v for k, v in dev_raw.items() if k in Partition.__annotations__})
                other_devices.append(dev)

        # 3. Augment with collected data
        all_devs = disks + other_devices
        usage_lookup = {fs.mount_point: fs for fs in fs_usage}

        for dev in all_devs:
            # Augment partitions if the device is a disk
            if isinstance(dev, Disk):
                for part in dev.partitions:
                    part.io_stats = io_stats.get(part.name)
                    if part.mount_point and part.mount_point in usage_lookup:
                        part.usage = usage_lookup[part.mount_point]
                # Augment the disk itself
                dev.io_stats = io_stats.get(dev.name)
                dev.smart_info = self.get_smart_info(dev.full_path)
            # Augment other top-level devices (partitions, LVMs)
            elif isinstance(dev, Partition):
                dev.io_stats = io_stats.get(dev.name)
                if dev.mount_point and dev.mount_point in usage_lookup:
                    dev.usage = usage_lookup[dev.mount_point]

        # 4. Assess overall health
        overall_status = "HEALTHY"
        summary = {"critical_issues": [], "warning_issues": []}

        for disk in disks:
            # Critical: S.M.A.R.T. failure or critical errors
            if disk.smart_info:
                if disk.smart_info.health_status == "FAILED":
                    overall_status = "CRITICAL"
                    summary["critical_issues"].append(f"Disk {disk.name}: S.M.A.R.T. check failed.")
                elif disk.smart_info.health_status == "WARNING":
                    if overall_status != "CRITICAL": overall_status = "WARNING"
                    summary["warning_issues"].append(f"Disk {disk.name}: S.M.A.R.T. warnings: {disk.smart_info.critical_issues}")
                if disk.smart_info.error:
                     if overall_status != "CRITICAL": overall_status = "WARNING"
                     summary["warning_issues"].append(f"Disk {disk.name}: Could not retrieve S.M.A.R.T. data.")

            # Critical: Key filesystems (e.g., root, data log) over 95% full
            for part in disk.partitions:
                if part.usage and part.usage.use_percentage > 95:
                    overall_status = "CRITICAL"
                    summary["critical_issues"].append(f"Filesystem at {part.mount_point} is critically full ({part.usage.use_percentage}%).")
                elif part.usage and part.usage.use_percentage > 85:
                    if overall_status != "CRITICAL": overall_status = "WARNING"
                    summary["warning_issues"].append(f"Filesystem at {part.mount_point} is nearly full ({part.usage.use_percentage}%).")

        logging.info(f"Storage self-check completed. Overall status: {overall_status}")

        return StorageHealthReport(
            overall_status=overall_status,
            disks=disks,
            other_devices=other_devices,
            summary=summary,
            errors=all_errors
        )

# --- Main execution block for standalone testing ---
if __name__ == "__main__":
    print("="*60)
    print("  AUTONOMOUS DRIVING STORAGE SUBSYSTEM SELF-CHECK")
    print("="*60)

    storage_manager = StorageManager()

    # Allow a short period for I/O stats to accumulate a baseline
    print("\n[INFO] Waiting for 2 seconds to gather initial I/O statistics...")
    time.sleep(2)

    # Run the comprehensive check
    print("[INFO] Performing comprehensive storage self-check...")
    report = storage_manager.run_self_check()

    # Convert report to a dictionary for JSON serialization
    # asdict is a helper from dataclasses to convert nested dataclasses to dicts
    report_dict = asdict(report)

    print("\n--- JSON Report ---")
    print(json.dumps(report_dict, indent=2))

    print("\n--- Human-Readable Summary ---")
    print(f"Overall Storage Health: {report.overall_status}")

    if report.summary['critical_issues']:
        print("\nCRITICAL ISSUES DETECTED:")
        for issue in report.summary['critical_issues']:
            print(f"  - [CRITICAL] {issue}")

    if report.summary['warning_issues']:
        print("\nWARNINGS DETECTED:")
        for issue in report.summary['warning_issues']:
            print(f"  - [WARNING] {issue}")

    if report.errors:
        print("\nERRORS DURING SELF-CHECK:")
        for err in report.errors:
            print(f"  - {err}")

    print("\n--- Detailed Device Status ---")
    for disk in report.disks:
        print(f"\n[DISK] {disk.name} ({disk.model}) - Size: {disk.size_gb} GB")
        if disk.smart_info:
            print(f"  - S.M.A.R.T. Health: {disk.smart_info.health_status}")
            print(f"  - Temperature: {disk.smart_info.temperature_celsius}°C")
            print(f"  - Power On Hours: {disk.smart_info.lifetime.get('power_on_hours', 'N/A')}")
            if disk.smart_info.lifetime.get('percentage_used') is not None:
                print(f"  - SSD Lifetime Used: {disk.smart_info.lifetime['percentage_used']}%")
        if disk.io_stats:
            print(f"  - I/O (r/w Bps): {disk.io_stats.read_bytes_per_sec:.0f} / {disk.io_stats.write_bytes_per_sec:.0f}")

        for part in disk.partitions:
            mount_info = f"at {part.mount_point}" if part.mount_point else "Not mounted"
            print(f"  - [PART] {part.name} ({part.size_gb} GB, {part.filesystem_type}) - {mount_info}")
            if part.usage:
                print(f"    - Usage: {part.usage.use_percentage}% full ({part.usage.available_bytes / (1024**3):.2f} GB free)")

    print("\n" + "="*60)
    print("  SELF-CHECK COMPLETE")
    print("="*60)
