#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
An automated performance monitoring script for NVIDIA Jetson Orin.

This script starts the 'tegrastats' utility for a specified duration,
automatically stops it, parses the resulting log file, and then
calculates and prints a statistical report for CPU, GPU, and RAM usage
using only native Python libraries.
"""

import subprocess
import time
import os
import sys
import re
import argparse
import signal
from typing import List, Dict, Any, Optional

# --- Analysis logic (Step 2 & 4), rewritten with native Python ---

def calculate_stats(data: List[float]) -> Dict[str, Any]:
    """
    Calculates count, min, max, and average for a list of numbers.
    Returns a dictionary with the results.
    """
    if not data:
        return {'count': 0, 'min': 0, 'max': 0, 'avg': 0}

    count = len(data)
    total = sum(data)
    return {
        'count': count,
        'min': min(data),
        'max': max(data),
        'avg': total / count
    }

def analyze_and_report(log_file: str):
    """
    Parses the tegrastats log file, calculates, and prints a performance report.
    """
    records = []

    # Regular expressions to extract key information from each log line
    ram_regex = re.compile(r"RAM (\d+)/\d+MB")
    cpu_regex = re.compile(r"CPU \[([^\]]+)\]")
    gpu_regex = re.compile(r"GR3D_FREQ (\d+)%")

    print(f"\n[+] Analyzing log file: {log_file}")

    try:
        with open(log_file, 'r') as f:
            for line in f:
                ram_match = ram_regex.search(line)
                cpu_match = cpu_regex.search(line)
                gpu_match = gpu_regex.search(line)

                # Ensure all key metrics are found in the line before processing
                if not (ram_match and cpu_match and gpu_match):
                    continue

                # 1. Extract RAM usage in MB
                ram_used = int(ram_match.group(1))

                # 2. Extract and calculate average CPU usage across all cores
                cpu_usages_str = cpu_match.group(1).split(',')
                cpu_percentages = [int(u.split('%')[0]) for u in cpu_usages_str if u.split('%')[0].isdigit()]
                avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0

                # 3. Extract GPU usage percentage
                gpu_usage = int(gpu_match.group(1))

                records.append({
                    'RAM_Used_MB': ram_used,
                    'CPU_Avg_Usage_%': avg_cpu_usage,
                    'GPU_Usage_%': gpu_usage
                })
    except FileNotFoundError:
        print(f"[!] ERROR: Log file '{log_file}' not found.")
        return

    if not records:
        print("[!] No valid data found in the log file. Please check if tegrastats ran correctly.")
        return

    print(f"[*] Analysis complete. Found {len(records)} valid records.")

    # Extract data into separate lists for calculation
    ram_data = [r['RAM_Used_MB'] for r in records]
    cpu_data = [r['CPU_Avg_Usage_%'] for r in records]
    gpu_data = [r['GPU_Usage_%'] for r in records]

    # Calculate statistics for each metric
    ram_stats = calculate_stats(ram_data)
    cpu_stats = calculate_stats(cpu_data)
    gpu_stats = calculate_stats(gpu_data)

    # --- Print the final report ---
    print("\n==================== Performance Statistics Report ====================")
    header = f"{'Metric':<20} | {'Count':>10} | {'Average':>10} | {'Min':>10} | {'Max':>10}"
    print(header)
    print("-" * len(header))

    print(f"{'RAM_Used_MB':<20} | {ram_stats['count']:>10} | {ram_stats['avg']:>10.2f} | {ram_stats['min']:>10.2f} | {ram_stats['max']:>10.2f}")
    print(f"{'CPU_Avg_Usage_%':<20} | {cpu_stats['count']:>10} | {cpu_stats['avg']:>10.2f} | {cpu_stats['min']:>10.2f} | {cpu_stats['max']:>10.2f}")
    print(f"{'GPU_Usage_%':<20} | {gpu_stats['count']:>10} | {gpu_stats['avg']:>10.2f} | {gpu_stats['min']:>10.2f} | {gpu_stats['max']:>10.2f}")

    print("=====================================================================\n")


# --- Automation logic for execution (Step 1 & 3) ---

def main():
    """
    Main execution function for the script.
    """
    parser = argparse.ArgumentParser(
        description="Automated performance monitor and analyzer for NVIDIA Jetson.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "duration",
        type=int,
        help="The total duration to monitor in seconds."
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=1000,
        help="The data sampling interval in milliseconds. Default: 1000ms (1 second)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tegrastats_run.log",
        help="The path for the output log file. Default: tegrastats_run.log"
    )
    parser.add_argument(
        "--keep-log",
        action="store_true",
        help="Keep the log file after analysis. By default, it is deleted."
    )

    args = parser.parse_args()

    # Check if the 'tegrastats' command exists
    try:
        subprocess.check_call(['which', 'tegrastats'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("[!] ERROR: 'tegrastats' command not found. Please run this script on a Jetson device.")
        sys.exit(1)

    # Define the tegrastats command
    log_file = args.output
    command = ["tegrastats", "--interval", str(args.interval), "--logfile", log_file]

    tegrastats_process: Optional[subprocess.Popen] = None

    try:
        # Start the tegrastats process
        print(f"[*] Starting tegrastats. Monitoring will last for {args.duration} seconds...")
        print(f"[*] Data will be logged to: {log_file}")

        # Use Popen to run the process in the background
        tegrastats_process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)

        # Wait for the specified duration, providing user feedback
        for i in range(args.duration):
            time.sleep(1)
            # '\r' moves the cursor to the beginning of the line, 'end=""' prevents a new line
            print(f"\r[*] Monitoring... {args.duration - 1 - i} seconds remaining", end="")
        print("\n[*] Monitoring time finished!")

    except KeyboardInterrupt:
        print("\n[!] User interruption detected (Ctrl+C). Stopping and analyzing...")
    finally:
        if tegrastats_process and tegrastats_process.poll() is None:
            print("[*] Stopping the tegrastats process...")
            # Send SIGTERM for a graceful shutdown
            os.kill(tegrastats_process.pid, signal.SIGTERM)
            try:
                # Wait for the process to terminate
                tegrastats_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If it doesn't terminate gracefully, force kill it
                print("[!] Process did not terminate gracefully. Forcing shutdown.")
                os.kill(tegrastats_process.pid, signal.SIGKILL)
            print("[*] tegrastats has been stopped.")

    # Analyze the log file if it exists
    if not os.path.exists(log_file):
        print(f"[!] The log file {log_file} was not created. Cannot analyze.")
        return

    analyze_and_report(log_file)

    # Clean up the log file unless requested otherwise
    if not args.keep_log:
        try:
            os.remove(log_file)
            print(f"[*] Temporary log file '{log_file}' has been deleted.")
        except OSError as e:
            print(f"[!] Error deleting log file: {e}")

if __name__ == "__main__":
    main()
