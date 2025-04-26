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
AD System Profiler: A Pythonic tool to collect comprehensive system information
for Autonomous Driving platform baselining and diagnostics.
"""

import os
import sys
import platform
import json
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Callable, Optional, Tuple

# --- Data Structures (using dataclasses for clarity and type safety) ---

@dataclass
class HashedFile:
    """Represents a file with its path and SHA256 hash."""
    path: str
    hash_sha256: Optional[str]

@dataclass
class KernelModule:
    """Represents a loaded Linux kernel module."""
    name: str
    size: int
    instances: int
    dependencies: List[str]

@dataclass
class MountPoint:
    """Represents a filesystem mount point."""
    device: str
    path: str
    type: str
    options: List[str]

@dataclass
class PciDevice:
    """Represents a PCI device identified by lspci."""
    bus_id: str
    description: str

# --- Core Utilities ---

def _run_command(command: List[str]) -> Optional[str]:
    """A robust wrapper for running external commands."""
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        if process.returncode != 0:
            sys.stderr.write(
                f"⚠️  Warning: Command `{' '.join(command)}` "
                f"exited with code {process.returncode}: {process.stderr.strip()}\n"
            )
            return None
        return process.stdout.strip()
    except FileNotFoundError:
        sys.stderr.write(f"⚠️  Warning: Command not found: `{command[0]}`\n")
        return None
    except Exception as e:
        sys.stderr.write(f"❌ Error running command `{' '.join(command)}`: {e}\n")
        return None

def _get_file_hash(path: Path) -> Optional[str]:
    """Computes the SHA256 hash of a file, returning None on error."""
    if not path.is_file():
        return None
    try:
        hasher = hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, PermissionError) as e:
        sys.stderr.write(f"⚠️  Warning: Could not read file `{path}`: {e}\n")
        return "permission_denied_or_io_error"

def _collect_hashed_files(directories: List[str], patterns: List[str]) -> List[HashedFile]:
    """Helper to find files by patterns and return HashedFile objects."""
    files = []
    processed_paths = set()
    for dir_str in directories:
        directory = Path(dir_str)
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for file_path in directory.glob(f"**/{pattern}"):
                if file_path.is_file() and file_path not in processed_paths:
                    files.append(HashedFile(
                        path=str(file_path),
                        hash_sha256=_get_file_hash(file_path)
                    ))
                    processed_paths.add(file_path)
    # Sort for deterministic output
    files.sort(key=lambda x: x.path)
    return files

# --- Probe Functions (Each collects a specific piece of system data) ---

def probe_os_info() -> Dict[str, Any]:
    """Probes for Operating System, kernel, and hardware platform details."""
    try:
        cmdline = Path("/proc/cmdline").read_text(encoding='utf-8').strip()
    except (IOError, PermissionError):
        cmdline = "Not Accessible"

    gcc_output = _run_command(["gcc", "--version"])

    return {
        "node_name": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "kernel_release": platform.release(),
        "kernel_version": platform.version(),
        "kernel_cmdline": cmdline,
        "python_version": platform.python_version(),
        "gcc_version": gcc_output.splitlines()[0] if gcc_output else "Not Found",
    }

def probe_environment_variables() -> Dict[str, str]:
    """Probes for all environment variables."""
    # Note: Can contain sensitive information. The consumer of the data is
    # responsible for filtering/sanitizing if needed.
    return dict(os.environ)

def probe_installed_packages(tool: str = "dpkg") -> List[str]:
    """Probes for installed packages using dpkg or rpm."""
    if tool == "dpkg":
        output = _run_command(["dpkg-query", "-W", "-f=${Package}=${Version}\n"])
    elif tool == "rpm":
        output = _run_command(["rpm", "-qa", "--queryformat", "%{NAME}=%{VERSION}-%{RELEASE}\n"])
    else:
        sys.stderr.write(f"⚠️  Warning: Unsupported package tool `{tool}`\n")
        return []
    return sorted(output.splitlines()) if output else []

def probe_kernel_modules() -> List[KernelModule]:
    """Probes for loaded kernel modules from /proc/modules."""
    modules_path = Path("/proc/modules")
    if not modules_path.exists():
        return []

    modules = []
    try:
        content = modules_path.read_text(encoding='utf-8')
        for line in content.strip().splitlines():
            parts = line.split()
            modules.append(KernelModule(
                name=parts[0],
                size=int(parts[1]),
                instances=int(parts[2]),
                dependencies=parts[3].split(',') if parts[3] != '-' else []
            ))
        modules.sort(key=lambda m: m.name)
        return modules
    except (IOError, PermissionError) as e:
        sys.stderr.write(f"⚠️  Warning: Could not read `{modules_path}`: {e}\n")
        return []

def probe_pci_devices() -> List[PciDevice]:
    """Probes for PCI devices, essential for GPUs, NICs, etc."""
    output = _run_command(["lspci"])
    if not output:
        return []

    devices = []
    for line in output.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            devices.append(PciDevice(bus_id=parts[0], description=parts[1].strip()))
    return devices

def probe_network_config() -> Dict[str, Any]:
    """Probes for detailed network configuration using the 'ip' command."""
    interfaces = _run_command(["ip", "-j", "addr"])
    routes = _run_command(["ip", "-j", "route"])
    sockets = _run_command(["ss", "-tulnp"]) # TCP, UDP, Listening, Numeric, Processes

    return {
        "interfaces": json.loads(interfaces) if interfaces else "Failed to get interface data",
        "routing_table": json.loads(routes) if routes else "Failed to get routing data",
        "listening_sockets": sockets.splitlines() if sockets else "Failed to get socket data"
    }

def probe_mount_points() -> List[MountPoint]:
    """Probes for all active mount points from /proc/mounts."""
    mounts_path = Path("/proc/mounts")
    if not mounts_path.exists():
        return []

    mounts = []
    try:
        content = mounts_path.read_text(encoding='utf-8')
        for line in content.strip().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                mounts.append(MountPoint(
                    device=parts[0],
                    path=parts[1],
                    type=parts[2],
                    options=parts[3].split(',')
                ))
        mounts.sort(key=lambda m: m.path)
        return mounts
    except (IOError, PermissionError) as e:
        sys.stderr.write(f"⚠️  Warning: Could not read `{mounts_path}`: {e}\n")
        return []

def probe_udev_rules() -> List[HashedFile]:
    """Probes for udev rule files."""
    dirs = ['/etc/udev/rules.d/', '/lib/udev/rules.d/', '/run/udev/rules.d/']
    return _collect_hashed_files(dirs, ["*.rules"])

def probe_systemd_config() -> List[HashedFile]:
    """Probes for systemd unit and configuration files."""
    dirs = [
        '/etc/systemd/system/', '/run/systemd/system/', '/lib/systemd/system/',
        '/etc/systemd/user/', '/run/systemd/user/', '/lib/systemd/user/'
    ]
    patterns = ['*.service', '*.socket', '*.target', '*.mount', '*.timer']
    return _collect_hashed_files(dirs, patterns)

def probe_security_files() -> Dict[str, Any]:
    """Probes for key security-related configuration files."""
    files_to_check = [
        '/etc/ssh/sshd_config',
        '/etc/sysctl.conf',
    ]
    hashed_files = [HashedFile(path=f, hash_sha256=_get_file_hash(Path(f))) for f in files_to_check]

    # Also collect all files from common drop-in directories
    hashed_files.extend(_collect_hashed_files(['/etc/sysctl.d/'], ['*.conf']))
    hashed_files.extend(_collect_hashed_files(['/etc/ssh/sshd_config.d/'], ['*.conf']))

    # Sort for deterministic output
    hashed_files.sort(key=lambda x: x.path)
    return {"config_files": hashed_files}

# --- Profiler Engine ---

PROBE_REGISTRY: Dict[str, Callable[[], Any]] = {
    "os_info": probe_os_info,
    "environment": probe_environment_variables,
    "installed_packages": probe_installed_packages,
    "kernel_modules": probe_kernel_modules,
    "pci_devices": probe_pci_devices,
    "network_config": probe_network_config,
    "mount_points": probe_mount_points,
    "udev_rules": probe_udev_rules,
    "systemd_config": probe_systemd_config,
    "security_files": probe_security_files,
}

def run_profiler() -> Dict[str, Any]:
    """Executes all registered probes and collects their data."""
    system_profile = {}
    print("--- 🚀 Starting AD System Profiler ---")
    for name, probe_func in PROBE_REGISTRY.items():
        print(f"🔬 Probing {name}...")
        try:
            system_profile[name] = probe_func()
        except Exception as e:
            sys.stderr.write(f"❌ Unhandled error in probe `{name}`: {e}\n")
            system_profile[name] = {"error": str(e)}
    print("--- ✅ Profiling Complete ---")
    return system_profile

def custom_json_encoder(obj: Any) -> Any:
    """Custom JSON encoder to handle dataclasses."""
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def main():
    """Main entry point of the script."""
    profile_data = run_profiler()

    # Generate a descriptive filename
    hostname = platform.node().replace('.', '_')
    timestamp = platform.python_version() # A simple timestamp for this example
    output_filename = f"system_profile_{hostname}_{timestamp}.json"

    print(f"\n💾 Saving profile to `{output_filename}`...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, default=custom_json_encoder, ensure_ascii=False)
        print(f"   Successfully wrote profile.")
    except Exception as e:
        sys.stderr.write(f"❌ Error writing to file `{output_filename}`: {e}\n")

    # Optional: Print a summary to stdout
    print("\n--- 📝 Profile Summary ---")
    print(f"OS Platform: {profile_data.get('os_info', {}).get('platform', 'N/A')}")
    print(f"Kernel Release: {profile_data.get('os_info', {}).get('kernel_release', 'N/A')}")
    print(f"Found {len(profile_data.get('installed_packages', []))} installed packages.")
    print(f"Found {len(profile_data.get('kernel_modules', []))} loaded kernel modules.")
    print(f"Found {len(profile_data.get('pci_devices', []))} PCI devices.")
    print("-------------------------\n")
    print(f"The full, detailed profile has been saved to '{output_filename}'.")
    print("This file can be used as a 'golden baseline' for system verification.")

if __name__ == "__main__":
    main()