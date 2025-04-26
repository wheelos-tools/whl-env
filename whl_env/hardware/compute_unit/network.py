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
import time
from typing import Dict, List, Any, Optional

try:
    import psutil
except ImportError:
    print("Error: The 'psutil' library is required. Please install it using 'pip install psutil'")
    exit(1)

# --- Placeholder for CAN bus specific library ---
# This would be an actual library like 'python-can'
try:
    # from can import interface # Example import from python-can
    CAN_LIB_AVAILABLE = False # Set to True if actual lib is imported
except ImportError:
    CAN_LIB_AVAILABLE = False
    logging.warning("Python-can library not found. CAN bus monitoring will be limited or unavailable.")

# --- Placeholder for Serial (RS232) specific library ---
# This would be an actual library like 'pyserial'
try:
    # import serial # Example import from pyserial
    SERIAL_LIB_AVAILABLE = False # Set to True if actual lib is imported
except ImportError:
    SERIAL_LIB_AVAILABLE = False
    logging.warning("PySerial library not found. RS232 monitoring will be limited or unavailable.")


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NetworkManager:
    """
    Manages retrieval of network interface information, including configuration
    and real-time statistics (flow rate, packet counts, errors, drops).
    Primarily uses psutil for robust and cross-platform data collection.
    """
    def __init__(self):
        # We store last counters globally for the class instance to enable rate calculation
        self._last_net_io_counters: Dict[str, psutil._common.snetio] = {}
        # Store the timestamp of the last counter read
        self._last_net_io_timestamp: float = time.monotonic()
        logging.info("NetworkManager initialized.")

    def get_interface_config(self) -> Dict[str, Any]:
        """
        Retrieves static configuration details for all network interfaces (primarily IP-based).

        Returns:
            A dictionary containing a list of interface configurations.
        """
        interfaces_config: List[Dict[str, Any]] = []
        try:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            for name, addr_list in addrs.items():
                iface_info: Dict[str, Any] = {
                    "name": name,
                    "addresses": [],
                    "mac_address": "N/A",
                    "state": "UNKNOWN",
                    "is_up": False,
                    "duplex": "N/A",
                    "speed_mbps": "N/A"
                }

                if name in stats:
                    stat = stats[name]
                    iface_info["is_up"] = stat.isup
                    iface_info["state"] = "UP" if stat.isup else "DOWN"
                    iface_info["duplex"] = str(stat.duplex).split('.')[-1]
                    iface_info["speed_mbps"] = stat.speed

                for addr in addr_list:
                    if addr.family == psutil.AF_LINK:
                        iface_info["mac_address"] = addr.address
                    else:
                        family_name = "IPv4" if addr.family == psutil.AF_INET else \
                                      "IPv6" if addr.family == psutil.AF_INET6 else \
                                      str(addr.family)

                        # Calculate CIDR for IPv4 if netmask is available
                        cidr = None
                        if addr.netmask and addr.family == psutil.AF_INET:
                            try:
                                # Convert netmask to binary and count '1's
                                netmask_parts = [int(x) for x in addr.netmask.split('.')]
                                cidr = sum(bin(x).count('1') for x in netmask_parts)
                                cidr = f"{addr.address}/{cidr}"
                            except ValueError:
                                pass # Malformed netmask
                        elif addr.family == psutil.AF_INET6 and addr.netmask:
                             # For IPv6, netmask is already in CIDR format or can be easily converted
                            cidr = f"{addr.address}/{addr.netmask.count('f')*4}" # Heuristic, not robust
                            # More robust IPv6 CIDR parsing needs ipaddress module or similar

                        iface_info["addresses"].append({
                            "family": family_name,
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast,
                            "cidr": cidr
                        })
                interfaces_config.append(iface_info)

            return {"interfaces": interfaces_config}

        except Exception as e:
            logging.error(f"Failed to get network interface configuration: {e}")
            return {"error": f"Failed to get network configuration: {e}"}

    def get_interface_stats(self) -> Dict[str, Any]:
        """
        Retrieves real-time network interface statistics (bytes/packets sent/received, errors, drops).
        Calculates rates (bytes/sec, packets/sec) using the time difference from the last call.

        Returns:
            A dictionary containing a list of interface statistics.
        """
        current_net_io_counters = psutil.net_io_counters(pernic=True)
        current_timestamp = time.monotonic()
        interfaces_stats: List[Dict[str, Any]] = []

        # Calculate time interval since last call
        time_delta = current_timestamp - self._last_net_io_timestamp

        if time_delta == 0:
            # This can happen if called too quickly in succession, or if time.monotonic() resolution is low.
            # In this case, we cannot calculate rates, so return 0 for rates.
            logging.warning("Time delta is zero. Cannot calculate rates for network stats.")

        for name, current_stats in current_net_io_counters.items():
            stats_entry: Dict[str, Any] = {
                "name": name,
                "bytes_sent_total": current_stats.bytes_sent,
                "bytes_recv_total": current_stats.bytes_recv,
                "packets_sent_total": current_stats.packets_sent,
                "packets_recv_total": current_stats.packets_recv,
                "errors_in_total": current_stats.errin,
                "errors_out_total": current_stats.errout,
                "drops_in_total": current_stats.dropin,
                "drops_out_total": current_stats.dropout,
                "bytes_sent_rate_bps": 0.0,
                "bytes_recv_rate_bps": 0.0,
                "packets_sent_rate_pps": 0.0,
                "packets_recv_rate_pps": 0.0,
                "errors_in_rate_pps": 0.0,
                "errors_out_rate_pps": 0.0,
                "drops_in_rate_pps": 0.0,
                "drops_out_rate_pps": 0.0,
                "health_status": "N/A (first sample or no time delta)"
            }

            if name in self._last_net_io_counters and time_delta > 0:
                last_stats = self._last_net_io_counters[name]

                bytes_sent_delta = current_stats.bytes_sent - last_stats.bytes_sent
                bytes_recv_delta = current_stats.bytes_recv - last_stats.bytes_recv
                packets_sent_delta = current_stats.packets_sent - last_stats.packets_sent
                packets_recv_delta = current_stats.packets_recv - last_stats.packets_recv

                stats_entry["bytes_sent_rate_bps"] = round(bytes_sent_delta / time_delta, 2)
                stats_entry["bytes_recv_rate_bps"] = round(bytes_recv_delta / time_delta, 2)
                stats_entry["packets_sent_rate_pps"] = round(packets_sent_delta / time_delta, 2)
                stats_entry["packets_recv_rate_pps"] = round(packets_recv_delta / time_delta, 2)

                errin_delta = current_stats.errin - last_stats.errin
                errout_delta = current_stats.errout - last_stats.errout
                dropin_delta = current_stats.dropin - last_stats.dropin
                dropout_delta = current_stats.dropout - last_stats.dropout

                stats_entry["errors_in_rate_pps"] = round(errin_delta / time_delta, 2)
                stats_entry["errors_out_rate_pps"] = round(errout_delta / time_delta, 2)
                stats_entry["drops_in_rate_pps"] = round(dropin_delta / time_delta, 2)
                stats_entry["drops_out_rate_pps"] = round(dropout_delta / time_delta, 2)

                stats_entry["health_status"] = self._assess_network_health(
                    name,
                    stats_entry["errors_in_rate_pps"],
                    stats_entry["errors_out_rate_pps"],
                    stats_entry["drops_in_rate_pps"],
                    stats_entry["drops_out_rate_pps"]
                )
            else:
                logging.debug(f"No previous stats for interface '{name}' or zero time_delta. Rates are 0.")

            interfaces_stats.append(stats_entry)

        # Update last counters and timestamp for the next call
        self._last_net_io_counters = current_net_io_counters
        self._last_net_io_timestamp = current_timestamp

        return {"interfaces_stats": interfaces_stats}

    @staticmethod
    def _assess_network_health(interface_name: str, errin_rate: float, errout_rate: float, dropin_rate: float, dropout_rate: float) -> str:
        """
        Assesses the health of a network interface based on error and drop rates.
        Thresholds are illustrative and should be tuned for specific environments.
        """
        # Define thresholds for warning and critical levels
        WARN_RATE_THRESHOLD = 0.1 # Example: 0.1 errors/drops per second
        CRIT_RATE_THRESHOLD = 0.5 # Example: 0.5 errors/drops per second

        if (errin_rate >= CRIT_RATE_THRESHOLD or errout_rate >= CRIT_RATE_THRESHOLD) or \
           (dropin_rate >= CRIT_RATE_THRESHOLD or dropout_rate >= CRIT_RATE_THRESHOLD):
            return "CRITICAL"
        elif (errin_rate >= WARN_RATE_THRESHOLD or errout_rate >= WARN_RATE_THRESHOLD) or \
             (dropin_rate >= WARN_RATE_THRESHOLD or dropout_rate >= WARN_RATE_THRESHOLD):
            return "WARNING"
        return "OK"

    def get_overall_network_health(self) -> str:
        """
        Provides an overall network health assessment based on all interfaces' current statistics.
        This method will call get_interface_stats to get the latest data.
        """
        # Call get_interface_stats. The time_delta calculation is now handled internally
        # by get_interface_stats based on its last call.
        current_stats_report = self.get_interface_stats()

        if "interfaces_stats" in current_stats_report:
            has_critical = False
            has_warning = False
            for iface in current_stats_report["interfaces_stats"]:
                status = iface.get("health_status", "N/A")
                if status == "CRITICAL":
                    has_critical = True
                    break # Critical overrides everything, no need to check further
                elif status == "WARNING":
                    has_warning = True

            if has_critical:
                return "OVERALL: CRITICAL - At least one network interface is severely impacted."
            elif has_warning:
                return "OVERALL: WARNING - Some network interfaces show minor issues."

        return "OVERALL: OK - Network appears healthy."


# --- Dedicated Classes for Non-IP Network Protocols ---

class CANbusMonitor:
    """
    Monitors CAN bus activity, including status, message counts, and error frames.
    Requires 'python-can' library and appropriate hardware interface (e.g., CAN adapter).
    """
    def __init__(self, channel: str = 'can0', bustype: str = 'socketcan'):
        self._channel = channel
        self._bustype = bustype
        self._bus_instance = None
        self._last_rx_count = 0
        self._last_tx_count = 0
        self._last_error_count = 0
        self._last_timestamp = time.monotonic()
        logging.info(f"CANbusMonitor initialized for {bustype}:{channel}.")

    def _connect_bus(self):
        """Attempts to establish connection to the CAN bus."""
        if not CAN_LIB_AVAILABLE:
            logging.error("CAN bus monitoring failed: 'python-can' library not installed.")
            raise ImportError("python-can library is required for CANbusMonitor.")

        try:
            # Example: self._bus_instance = can.interface.Bus(channel=self._channel, bustype=self._bustype)
            # In a real implementation, you'd try to connect here.
            # For now, simulate a successful connection for the example.
            logging.warning("Simulating CAN bus connection. Actual 'python-can' interaction is not implemented.")
            self._bus_instance = True # Placeholder for a successful bus object
        except Exception as e:
            logging.error(f"Failed to connect to CAN bus {self._channel} ({self._bustype}): {e}")
            self._bus_instance = None

    def get_status_and_stats(self) -> Dict[str, Any]:
        """
        Retrieves current CAN bus status and statistics (flow, errors, drops).
        """
        if self._bus_instance is None:
            try:
                self._connect_bus() # Try to connect on demand
                if self._bus_instance is None: # If connection failed
                    return {"name": f"CANbus({self._channel})", "status": "DISCONNECTED", "error": "Could not connect to CAN bus."}
            except ImportError as e:
                 return {"name": f"CANbus({self._channel})", "status": "UNAVAILABLE", "error": str(e)}

        current_timestamp = time.monotonic()
        time_delta = current_timestamp - self._last_timestamp

        # --- Simulate CAN data ---
        # In a real scenario, you'd read from self._bus_instance
        # Example: bus_stats = self._bus_instance.get_stats() (if API exists)
        # Or, listen to messages and count them.
        sim_rx_count = self._last_rx_count + 10 # Simulate messages
        sim_tx_count = self._last_tx_count + 5
        sim_error_count = self._last_error_count + (1 if current_timestamp % 5 < 1 else 0) # Simulate occasional error

        rx_delta = sim_rx_count - self._last_rx_count
        tx_delta = sim_tx_count - self._last_tx_count
        error_delta = sim_error_count - self._last_error_count

        msg_rx_rate = round(rx_delta / time_delta, 2) if time_delta > 0 else 0.0
        msg_tx_rate = round(tx_delta / time_delta, 2) if time_delta > 0 else 0.0
        error_rate = round(error_delta / time_delta, 2) if time_delta > 0 else 0.0

        status = {
            "name": f"CANbus({self._channel})",
            "status": "CONNECTED", # Or from bus_instance state
            "total_messages_received": sim_rx_count,
            "total_messages_sent": sim_tx_count,
            "total_error_frames": sim_error_count,
            "messages_received_rate_msg_per_sec": msg_rx_rate,
            "messages_sent_rate_msg_per_sec": msg_tx_rate,
            "error_frame_rate_err_per_sec": error_rate,
            "health_status": "OK" if error_rate < 0.1 else "WARNING" # Basic health
        }

        # Update last counts
        self._last_rx_count = sim_rx_count
        self._last_tx_count = sim_tx_count
        self._last_error_count = sim_error_count
        self._last_timestamp = current_timestamp

        return status

class SerialPortMonitor:
    """
    Monitors a Serial (RS232) port, including status, byte counts, and communication errors.
    Requires 'pyserial' library and a valid serial port.
    """
    def __init__(self, port: str = '/dev/ttyS0', baudrate: int = 9600):
        self._port = port
        self._baudrate = baudrate
        self._serial_instance = None
        self._last_rx_bytes = 0
        self._last_tx_bytes = 0
        self._last_error_count = 0 # Placeholder for parity/framing errors
        self._last_timestamp = time.monotonic()
        logging.info(f"SerialPortMonitor initialized for {port} @ {baudrate} baud.")

    def _connect_port(self):
        """Attempts to establish connection to the serial port."""
        if not SERIAL_LIB_AVAILABLE:
            logging.error("Serial port monitoring failed: 'pyserial' library not installed.")
            raise ImportError("pyserial library is required for SerialPortMonitor.")

        try:
            # Example: self._serial_instance = serial.Serial(self._port, self._baudrate, timeout=0.1)
            # For now, simulate a successful connection.
            logging.warning("Simulating serial port connection. Actual 'pyserial' interaction is not implemented.")
            self._serial_instance = True # Placeholder for a successful serial object
            # self._serial_instance.read() to clear buffers on start if needed
        except Exception as e:
            logging.error(f"Failed to open serial port {self._port}: {e}")
            self._serial_instance = None

    def get_status_and_stats(self) -> Dict[str, Any]:
        """
        Retrieves current serial port status and statistics (flow, errors).
        """
        if self._serial_instance is None:
            try:
                self._connect_port() # Try to connect on demand
                if self._serial_instance is None: # If connection failed
                    return {"name": f"SerialPort({self._port})", "status": "DISCONNECTED", "error": "Could not open serial port."}
            except ImportError as e:
                return {"name": f"SerialPort({self._port})", "status": "UNAVAILABLE", "error": str(e)}

        current_timestamp = time.monotonic()
        time_delta = current_timestamp - self._last_timestamp

        # --- Simulate Serial data ---
        # In a real scenario, you'd read from self._serial_instance.
        # This usually involves non-blocking reads and tracking bytes.
        sim_rx_bytes = self._last_rx_bytes + 20 # Simulate bytes
        sim_tx_bytes = self._last_tx_bytes + 10
        sim_error_count = self._last_error_count + (1 if current_timestamp % 7 < 1 else 0) # Simulate occasional error

        rx_delta = sim_rx_bytes - self._last_rx_bytes
        tx_delta = sim_tx_bytes - self._last_tx_bytes
        error_delta = sim_error_count - self._last_error_count

        bytes_rx_rate = round(rx_delta / time_delta, 2) if time_delta > 0 else 0.0
        bytes_tx_rate = round(tx_delta / time_delta, 2) if time_delta > 0 else 0.0
        error_rate = round(error_delta / time_delta, 2) if time_delta > 0 else 0.0

        status = {
            "name": f"SerialPort({self._port})",
            "status": "OPEN", # Or from serial_instance.is_open
            "total_bytes_received": sim_rx_bytes,
            "total_bytes_sent": sim_tx_bytes,
            "total_comm_errors": sim_error_count, # e.g., framing, parity errors
            "bytes_received_rate_Bps": bytes_rx_rate,
            "bytes_sent_rate_Bps": bytes_tx_rate,
            "comm_error_rate_err_per_sec": error_rate,
            "health_status": "OK" if error_rate < 0.1 else "WARNING" # Basic health
        }

        # Update last counts
        self._last_rx_bytes = sim_rx_bytes
        self._last_tx_bytes = sim_tx_bytes
        self._last_error_count = sim_error_count
        self._last_timestamp = current_timestamp

        return status


# Example of how to use the function (for testing purposes)
if __name__ == "__main__":
    print("Gathering network information...")
    net_manager = NetworkManager()

    # --- 1. Get Static IP Network Interface Configuration ---
    print("\n--- 1. IP Network Interface Configuration (Static) ---")
    config_info = net_manager.get_interface_config()
    print(json.dumps(config_info, indent=4))

    # --- 2. Get Real-time IP Network Interface Statistics ---
    print("\n--- 2. IP Network Interface Statistics (Live - First sample with rates=0) ---")
    # First call will populate _last_net_io_counters and _last_net_io_timestamp.
    # Rates will be 0 as there's no previous delta.
    stats_info_first = net_manager.get_interface_stats()
    print(json.dumps(stats_info_first, indent=4))

    print("\n--- 2a. IP Network Interface Statistics (Live - after 1 second, rates available) ---")
    # Wait for 1 second to ensure a meaningful time_delta for rate calculation
    time.sleep(1)
    stats_info_second = net_manager.get_interface_stats() # Now rates will be calculated
    print(json.dumps(stats_info_second, indent=4))

    # --- 3. Overall IP Network Health ---
    print("\n--- 3. Overall IP Network Health ---")
    overall_health = net_manager.get_overall_network_health()
    print(json.dumps({"overall_health": overall_health}, indent=4))

    # --- 4. Non-IP Network Monitoring ---
    print("\n--- 4. Non-IP Network Monitoring ---")
    print("\n--- CAN Bus Monitoring ---")
    can_monitor = CANbusMonitor()
    # Call get_status_and_stats multiple times with a delay to see rates change
    can_stats_first = can_monitor.get_status_and_stats()
    print(json.dumps(can_stats_first, indent=4))
    time.sleep(1)
    can_stats_second = can_monitor.get_status_and_stats()
    print(json.dumps(can_stats_second, indent=4))

    print("\n--- Serial Port (RS232) Monitoring ---")
    serial_monitor = SerialPortMonitor()
    serial_stats_first = serial_monitor.get_status_and_stats()
    print(json.dumps(serial_stats_first, indent=4))
    time.sleep(1)
    serial_stats_second = serial_monitor.get_status_and_stats()
    print(json.dumps(serial_stats_second, indent=4))

    # --- Summary Processing (updated to reflect new structure) ---
    print("\n--- Consolidated Summary ---")
    print("\nIP Network Interfaces:")
    if "interfaces" in config_info:
        for iface_config in config_info["interfaces"]:
            name = iface_config.get('name', 'N/A')

            # Find corresponding live stats from the second sample
            current_live_stats = next(
                (item for item in stats_info_second.get("interfaces_stats", []) if item["name"] == name),
                None
            )

            print(f"  Interface: {name}")
            print(f"    MAC: {iface_config.get('mac_address', 'N/A')}")
            print(f"    State: {'UP' if iface_config.get('is_up') else 'DOWN'}")

            if iface_config.get('addresses'):
                print("    IPs:")
                for addr in iface_config['addresses']:
                    print(f"      {addr.get('family')}: {addr.get('address')}/{addr.get('cidr') if addr.get('cidr') else addr.get('netmask')}")

            if current_live_stats:
                print(f"    Rates (Bps/pps): Tx {current_live_stats.get('bytes_sent_rate_bps', 0):.2f}/{current_live_stats.get('packets_sent_rate_pps', 0):.2f} | Rx {current_live_stats.get('bytes_recv_rate_bps', 0):.2f}/{current_live_stats.get('packets_recv_rate_pps', 0):.2f}")
                print(f"    Errors/Drops (pps): In {current_live_stats.get('errors_in_rate_pps', 0):.2f} / Out {current_live_stats.get('errors_out_rate_pps', 0):.2f} | Drop In {current_live_stats.get('drops_in_rate_pps', 0):.2f} / Drop Out {current_live_stats.get('drops_out_rate_pps', 0):.2f}")
                print(f"    Health: {current_live_stats.get('health_status', 'N/A')}")
            else:
                print("    Live stats not available or initial sample.")
            print("    " + "-" * 15)

    print("\nCAN Bus Status:")
    print(json.dumps(can_stats_second, indent=2)) # Use the second sample

    print("\nSerial Port Status:")
    print(json.dumps(serial_stats_second, indent=2)) # Use the second sample
