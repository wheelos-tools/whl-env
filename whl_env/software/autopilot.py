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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2025 WheelOS AD Systems. All Rights Reserved.
# (License header)

"""
AD System Profiler: A Pythonic tool to collect comprehensive static and
runtime information for Autonomous Driving platform baselining and diagnostics,
based on industry best practices for high-reliability systems.
"""

import os
import sys
import platform
import json
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Callable, Optional

# --- Core Utilities ---

def _run_command(command: List[str]) -> Optional[str]:
    """A robust wrapper for running external commands."""
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False,
            encoding='utf-8', errors='ignore'
        )
        if process.returncode != 0:
            sys.stderr.write(
                f"⚠️  Warning: Command `{' '.join(command)}` exited with code "
                f"{process.returncode}: {process.stderr.strip()}\n"
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
    """Computes the SHA256 hash of a file, returning a status on error."""
    if not path.is_file():
        return "not_a_file"
    try:
        hasher = hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, PermissionError):
        return "permission_denied_or_io_error"

# --- Dataclass Definitions for Structured Data ---

@dataclass
class Artifact:
    """Represents a data artifact (e.g., model, map, config file)."""
    path: str
    exists: bool
    type: str  # 'file' or 'directory'
    hash_sha256: Optional[str] = None
    error: Optional[str] = None

@dataclass
class DockerImage:
    """Represents essential details of a Docker image."""
    name: str
    digest: Optional[str]
    id: Optional[str]
    status: str # 'found' or 'not_found'

@dataclass
class RunningProcess:
    """Represents a running container or process."""
    id_or_pid: str
    name: str
    image_or_command: str
    status: str

# --- Probe Functions: One for each checklist item ---

def probe_ad_image_consistency(image_names: List[str]) -> List[DockerImage]:
    """
    Collects Docker image digests to verify AD/Toolchain consistency.
    The digest is the ground truth for image identity.
    """
    images_info = []
    for name in image_names:
        # Use --format to get the digest directly. This is the most reliable method.
        inspect_output = _run_command(["docker", "image", "inspect", name, "--format", "{{json .RepoDigests}},{{.Id}}"])
        if not inspect_output:
            images_info.append(DockerImage(name=name, digest=None, id=None, status="not_found"))
            continue

        try:
            digests_str, image_id = inspect_output.split(',', 1)
            digests = json.loads(digests_str)
            # Find the digest corresponding to the requested tag
            primary_digest = next((d for d in digests if d.startswith(name.split(':')[0])), digests[0] if digests else None)
            images_info.append(DockerImage(name=name, digest=primary_digest, id=image_id, status="found"))
        except (json.JSONDecodeError, IndexError) as e:
            sys.stderr.write(f"❌ Error parsing inspect output for {name}: {e}\n")
            images_info.append(DockerImage(name=name, digest=None, id=None, status="parse_error"))

    return images_info

def probe_artifact_integrity(artifact_paths: List[str]) -> List[Artifact]:
    """
    Collects checksums for configuration, model, calibration, and map files.
    """
    artifacts = []
    for path_str in artifact_paths:
        path = Path(path_str)
        if not path.exists():
            artifacts.append(Artifact(path=path_str, exists=False, type="unknown"))
            continue

        is_file = path.is_file()
        artifact_type = "file" if is_file else "directory"
        file_hash = _get_file_hash(path) if is_file else "is_directory"

        error = None
        if file_hash in ["permission_denied_or_io_error", "not_a_file"]:
            error = file_hash
            file_hash = None

        artifacts.append(Artifact(
            path=path_str,
            exists=True,
            type=artifact_type,
            hash_sha256=file_hash,
            error=error
        ))
    return artifacts

def probe_runtime_processes() -> Dict[str, List[RunningProcess]]:
    """
    Collects information on running Docker containers and key native processes.
    """
    # Docker containers
    containers = []
    docker_output = _run_command(["docker", "ps", "--format", "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}"])
    if docker_output:
        for line in docker_output.splitlines():
            parts = line.split('|')
            containers.append(RunningProcess(id_or_pid=parts[0], name=parts[1], image_or_command=parts[2], status=parts[3]))

    # Key native processes (example: a critical data logger)
    native_processes = []
    pgrep_output = _run_command(["pgrep", "-af", "critical_data_logger"]) # Search by full command line
    if pgrep_output:
        for line in pgrep_output.splitlines():
            pid, cmd = line.split(" ", 1)
            native_processes.append(RunningProcess(id_or_pid=pid, name="critical_data_logger", image_or_command=cmd, status="running"))

    return {"containers": containers, "native_processes": native_processes}

def probe_dds_topic_health(topics_to_check: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Collects runtime health of ROS/DDS topics, focusing on publishing frequency.
    Example `topics_to_check`: {"/perception/objects": {"min_hz": 8, "max_hz": 12}}
    """
    if not _run_command(["ros2", "daemon", "status"]):
         return {"status": "error", "message": "ROS 2 daemon not running or ros2 not found."}

    health_data = {}
    for topic, params in topics_to_check.items():
        # Using `--window 10` gives a more stable average rate over 10 messages
        hz_output = _run_command(["ros2", "topic", "hz", topic, "--window", "10"])

        result = {"status": "no_messages"}
        if hz_output:
            # Parse the summary line: "average rate: 9.951"
            for line in hz_output.splitlines():
                if "average rate" in line:
                    rate_str = line.split(":")[-1].strip()
                    try:
                        result = {"status": "publishing", "average_hz": float(rate_str)}
                    except ValueError:
                        result = {"status": "parse_error", "raw_output": rate_str}
                    break
        health_data[topic] = result

    return health_data

def probe_log_files(logs_to_scan: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Collects counts of critical keywords from specified log files.
    Example `logs_to_scan`: {"/var/log/syslog": ["error", "fail"], "/var/log/ad.log": ["FATAL"]}
    """
    scan_results = {}
    for log_file, keywords in logs_to_scan.items():
        path = Path(log_file)
        result = {"status": "", "keyword_counts": {k: 0 for k in keywords}}
        if not path.is_file():
            result["status"] = "not_found"
            scan_results[log_file] = result
            continue

        try:
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                # Reading last N lines is more efficient for large logs
                # For simplicity here, we read the whole file.
                content = f.read()
                for keyword in keywords:
                    result["keyword_counts"][keyword] = content.lower().count(keyword.lower())
            result["status"] = "scanned"
        except (IOError, PermissionError) as e:
            result["status"] = f"read_error: {e}"

        scan_results[log_file] = result
    return scan_results

# --- Main Profiler ---

def run_ad_system_profiler(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes all registered probes based on the provided configuration
    and returns a single JSON-serializable dictionary of system facts.
    """
    system_facts = {}
    print("--- 🚀 Starting AD System Profiler (Data Collection) ---")

    # A simple, registry-like execution pattern
    probes = {
        "ad_image_consistency": (probe_ad_image_consistency, config.get("critical_images", [])),
        "artifact_integrity": (probe_artifact_integrity, config.get("critical_artifacts", [])),
        "runtime_processes": (probe_runtime_processes, None),
        "dds_topic_health": (probe_dds_topic_health, config.get("topics_to_monitor", {})),
        "log_file_scan": (probe_log_files, config.get("logs_to_scan", {})),
    }

    for name, (probe_func, args) in probes.items():
        print(f"🔬 Collecting {name}...")
        try:
            if args is not None:
                system_facts[name] = probe_func(args)
            else:
                system_facts[name] = probe_func()
        except Exception as e:
            sys.stderr.write(f"❌ Unhandled error in probe `{name}`: {e}\n")
            system_facts[name] = {"error": str(e)}

    print("--- ✅ Profiling Complete ---")
    return system_facts

def custom_json_encoder(obj):
    """Custom JSON encoder for dataclasses."""
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

if __name__ == "__main__":
    # --- Configuration: Define what to check ---
    # In a real-world scenario, this configuration would be loaded from a YAML or JSON file.
    PROFILER_CONFIG = {
        "critical_images": [
            "autoware/autoware:latest", # Example AD stack image
            "nvidia/cuda:11.8.0-runtime-ubuntu22.04"  # Example base image
        ],
        "critical_artifacts": [
            "/opt/autoware/data/maps/sample-map.pcd",
            "/opt/autoware/data/models/yolox-s.onnx",
            "/opt/autoware/config/vehicle/vehicle_info.yaml",
            "/etc/some_critical_config.conf",
            "/this/file/does/not/exist.txt" # Example of a missing file
        ],
        "topics_to_monitor": {
            "/localization/kinematic_state": {},
            "/perception/object_recognition/tracking/objects": {},
            "/planning/scenario_planning/trajectory": {},
            "/sensing/lidar/top/pointcloud_raw": {},
        },
        "logs_to_scan": {
            "/var/log/syslog": ["error", "failure", "segfault"],
            "/var/log/autoware.log": ["FATAL", "Exception"]
        }
    }

    collected_facts = run_ad_system_profiler(PROFILER_CONFIG)

    output_filename = f"ad_system_facts_{platform.node()}.json"
    print(f"\n💾 Saving collected facts to `{output_filename}`...")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(collected_facts, f, indent=2, default=custom_json_encoder, ensure_ascii=False)

    print(f"   Successfully wrote facts.\n   This file is the 'ground truth' of the system's current state.")
