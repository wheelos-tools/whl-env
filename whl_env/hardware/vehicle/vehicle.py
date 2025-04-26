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


import time
import logging
from typing import Dict, Any, List, Optional

# Configure basic logging if not already configured
# In a real application, you would configure logging more elaborately
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class VehicleHealthChecker:
    """
    Monitors and processes vehicle chassis status information relevant to
    automated driving, focusing on received data about system components
    like SocketCAN interfaces, power, actuator readiness, and DTCs.

    The checks performed and thresholds used are guided by a configuration
    passed during initialization.

    Note: This class works with pre-formatted data received via its interface
    (`process_message`). It does NOT directly interact with vehicle hardware,
    CAN bus interfaces (/dev/...), or monitor raw topics/signals.
    The data publisher is responsible for gathering that raw information.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the VehicleHealthChecker with a configuration and default state.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration
                                     parameters for checks and thresholds.
                                     Example structure (will be validated):
                                     {
                                         "chassis_checks": {
                                             "socketcan": {
                                                 "require_all_up": True,
                                                 "max_interface_errors": 10, # Threshold for tx/rx errors or dropped
                                                 "monitored_interfaces": ["can0", "can1"] # Optional: list expected interfaces
                                             },
                                             "power": {
                                                 "require_stable": True,
                                                 "min_voltage_v": 12.0, # Optional threshold
                                                 "min_soc_percent": 20.0  # Optional threshold
                                             },
                                             "actuators": {
                                                 "require_brake_ready": True,
                                                 "require_throttle_ready": True,
                                                 "require_steer_ready": True
                                             },
                                             "dtc": {
                                                 "fail_on_critical_dtcs": True # Whether presence of any critical DTC fails AD readiness
                                             }
                                         }
                                     }
        """
        self._config = self._validate_config(
            config)  # Validate and store config
        # Timestamp when _chassis_data was last updated
        self._last_update_timestamp: Optional[float] = None

        # Internal state to store the latest processed data
        self._chassis_data: Dict[str, Any] = {
            "socketcan_interfaces": None,  # Store the list of interface dicts received
            "critical_dtcs": None,        # Store the list of critical AD-related DTCs received
            "battery_voltage_v": None,
            "battery_soc_percent": None,
            "power_stable": None,       # Boolean indicating power stability as reported
            # Boolean indicating brake drive-by-wire readiness as reported
            "brake_ready": None,
            # Boolean indicating throttle drive-by-wire readiness as reported
            "throttle_ready": None,
            "steer_ready": None         # Boolean indicating steer drive-by-wire readiness as reported
        }

        if self._config.get('initialization_error'):
            logging.error(
                f"VehicleHealthChecker initialized with config errors: {self._config['initialization_error']}")
        else:
            logging.info(
                "VehicleHealthChecker initialized successfully with config.")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to validate the structure and types of the input configuration.

        Args:
            config (Dict[str, Any]): The raw configuration dictionary.

        Returns:
            Dict[str, Any]: A validated and potentially defaulted configuration dictionary,
                            including an 'initialization_error' key if validation fails.
        """
        validated_config: Dict[str, Any] = {}
        errors: List[str] = []

        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary.")
            # Return a minimal structure indicating failure
            return {'initialization_error': errors}

        chassis_config = config.get("chassis_checks", {})
        if not isinstance(chassis_config, dict):
            errors.append("Config key 'chassis_checks' must be a dictionary.")
            chassis_config = {}  # Use empty dict to prevent further errors

        # Validate socketcan config
        socketcan_cfg = chassis_config.get("socketcan", {})
        if not isinstance(socketcan_cfg, dict):
            errors.append(
                "Config key 'chassis_checks.socketcan' must be a dictionary.")
            socketcan_cfg = {}  # Use empty dict

        validated_config["socketcan"] = {
            # Default values if keys are missing or wrong type
            "require_all_up": socketcan_cfg.get("require_all_up", True) if isinstance(socketcan_cfg.get("require_all_up"), bool) else True,
            "max_interface_errors": socketcan_cfg.get("max_interface_errors", 10) if isinstance(socketcan_cfg.get("max_interface_errors"), (int, float)) else 10,
            "monitored_interfaces": socketcan_cfg.get("monitored_interfaces", []) if isinstance(socketcan_cfg.get("monitored_interfaces"), list) else []
        }
        if not all(isinstance(name, str) for name in validated_config["socketcan"]["monitored_interfaces"]):
            errors.append(
                "Config key 'chassis_checks.socketcan.monitored_interfaces' must be a list of strings.")
            validated_config["socketcan"]["monitored_interfaces"] = []

        # Validate power config
        power_cfg = chassis_config.get("power", {})
        if not isinstance(power_cfg, dict):
            errors.append(
                "Config key 'chassis_checks.power' must be a dictionary.")
            power_cfg = {}  # Use empty dict

        validated_config["power"] = {
            "require_stable": power_cfg.get("require_stable", True) if isinstance(power_cfg.get("require_stable"), bool) else True,
            # Optional thresholds - allow None if missing/invalid
            "min_voltage_v": power_cfg.get("min_voltage_v") if isinstance(power_cfg.get("min_voltage_v"), (int, float, type(None))) else None,
            "min_soc_percent": power_cfg.get("min_soc_percent") if isinstance(power_cfg.get("min_soc_percent"), (int, float, type(None))) else None,
        }

        # Validate actuators config
        actuators_cfg = chassis_config.get("actuators", {})
        if not isinstance(actuators_cfg, dict):
            errors.append(
                "Config key 'chassis_checks.actuators' must be a dictionary.")
            actuators_cfg = {}  # Use empty dict

        validated_config["actuators"] = {
            "require_brake_ready": actuators_cfg.get("require_brake_ready", True) if isinstance(actuators_cfg.get("require_brake_ready"), bool) else True,
            "require_throttle_ready": actuators_cfg.get("require_throttle_ready", True) if isinstance(actuators_cfg.get("require_throttle_ready"), bool) else True,
            "require_steer_ready": actuators_cfg.get("require_steer_ready", True) if isinstance(actuators_cfg.get("require_steer_ready"), bool) else True
        }

        # Validate dtc config
        dtc_cfg = chassis_config.get("dtc", {})
        if not isinstance(dtc_cfg, dict):
            errors.append(
                "Config key 'chassis_checks.dtc' must be a dictionary.")
            dtc_cfg = {}  # Use empty dict

        validated_config["dtc"] = {
            "fail_on_critical_dtcs": dtc_cfg.get("fail_on_critical_dtcs", True) if isinstance(dtc_cfg.get("fail_on_critical_dtcs"), bool) else True,
        }

        # Aggregate all config checks under 'chassis_checks' key in final validated config
        final_validated_config: Dict[str, Any] = {
            "chassis_checks": validated_config}

        if errors:
            final_validated_config['initialization_error'] = errors
            logging.error(f"Configuration validation failed: {errors}")
        else:
            logging.info("Configuration validated successfully.")

        return final_validated_config

    def process_message(self, chassis_message: Dict[str, Any]) -> bool:
        """
        Receives, validates, and processes a new chassis data message.

        This is the primary interface for updating the monitor's state.
        It expects a dictionary containing keys for socketcan_interfaces,
        critical_dtcs, power, and actuator readiness.

        Args:
            chassis_message (Dict[str, Any]): A dictionary containing the latest
                                             chassis status data. Expected keys
                                             match the internal state keys.

        Returns:
            bool: True if the message was processed successfully (at least partially),
                  False otherwise (e.g., not a dict or critical structure invalid).
                  Warnings printed for missing/invalid individual fields.
        """
        if self._config.get('initialization_error'):
            logging.error(
                "Cannot process message due to configuration errors.")
            return False

        logging.debug("Attempting to process new chassis message...")

        # --- Input Validation (Industry Best Practice) ---
        if not isinstance(chassis_message, dict):
            logging.error(
                f"Error: Received message is not a dictionary. Type: {type(chassis_message)}")
            return False

        # Define expected keys and their basic types for incoming message payload
        # Allow None for many fields if the data publisher indicates value is unavailable
        expected_payload_structure = {
            "socketcan_interfaces": (list, type(None)),
            "critical_dtcs": (list, type(None)),
            "battery_voltage_v": (int, float, type(None)),
            "battery_soc_percent": (int, float, type(None)),
            "power_stable": (bool, type(None)),
            "brake_ready": (bool, type(None)),
            "throttle_ready": (bool, type(None)),
            "steer_ready": (bool, type(None))
        }

        validated_payload = {}
        # Flag for errors that prevent meaningful processing
        has_critical_validation_errors = False

        for key, expected_type in expected_payload_structure.items():
            if key not in chassis_message:
                # For incoming message, missing key is a warning, not necessarily fatal unless critical
                logging.warning(
                    f"Warning: Key '{key}' missing in chassis message. Using None.")
                validated_payload[key] = None  # Use None for missing keys
            else:
                value = chassis_message[key]
                if not isinstance(value, expected_type):
                    logging.warning(
                        f"Warning: Type mismatch for key '{key}'. Expected {expected_type}, got {type(value)}. Using None.")
                    validated_payload[key] = None  # Use None for type mismatch
                elif key == "socketcan_interfaces" and value is not None:
                    # Additional validation for the list of interfaces if provided
                    if not self._validate_socketcan_interfaces_payload(value):
                        logging.warning(
                            f"Warning: Invalid structure/types for 'socketcan_interfaces' payload. Using None.")
                        # Invalidate the whole list if internal validation fails
                        validated_payload[key] = None
                        # Note: If socketcan_interfaces is critical, set has_critical_validation_errors = True here
                    else:
                        # Use the validated list
                        validated_payload[key] = value
                elif key == "critical_dtcs" and value is not None:
                    # Additional validation for critical_dtcs list
                    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                        logging.warning(
                            f"Warning: Invalid structure/types for 'critical_dtcs' payload. Expected list of strings. Using None.")
                        validated_payload[key] = None
                    else:
                        validated_payload[key] = value
                else:
                    # Basic validation for other keys already done by isinstance check
                    validated_payload[key] = value

        # --- Update Internal State ---
        # Only update if initial validation passed enough for meaningful state
        if not has_critical_validation_errors:
            self._chassis_data.update(validated_payload)
            self._last_update_timestamp = time.time()
            logging.debug(
                "Chassis message processed (potentially with warnings).")
            return True
        else:
            logging.error(
                "Chassis message has critical validation errors, state not updated.")
            return False

    def _validate_socketcan_interfaces_payload(self, interfaces_list: List[Dict[str, Any]]) -> bool:
        """
        Internal helper to validate the structure and types of the
        socketcan_interfaces list received in the payload.
        Checks if it's a list and if each item is a dict with basic expected keys/types.
        """
        if not isinstance(interfaces_list, list):
            logging.warning(
                "_validate_socketcan_interfaces_payload: Input is not a list.")
            return False

        expected_interface_item_structure = {
            "name": str,
            "state": str,  # e.g., "UP", "DOWN", "UNKNOWN"
            # Add checks for expected error counters if needed in analysis, e.g.:
            "tx_errors": (int, type(None)),
            "rx_errors": (int, type(None)),
            "dropped": (int, type(None)),
            "state_flags": (str, type(None))  # e.g., "RUNNING", "NO-CARRIER"
        }
        # Keys that MUST be present in each interface dict
        required_keys = ["name", "state"]

        all_items_valid = True
        for i, interface in enumerate(interfaces_list):
            if not isinstance(interface, dict):
                logging.warning(
                    f"_validate_socketcan_interfaces_payload: Item {i} is not a dict: {interface}")
                all_items_valid = False
                continue  # Check next item

            # Check for presence of required keys
            if not all(key in interface for key in required_keys):
                logging.warning(
                    f"_validate_socketcan_interfaces_payload: Item {i} missing required key(s) {required_keys}: {interface}")
                all_items_valid = False
                # Continue to check types of keys that *are* present

            # Check types of *all* expected keys if they are present
            for key, expected_type in expected_interface_item_structure.items():
                if key in interface:
                    value = interface[key]
                    if not isinstance(value, expected_type):
                        logging.warning(
                            f"_validate_socketcan_interfaces_payload: Item {i} key '{key}' type mismatch. Expected {expected_type}, got {type(value)}: {interface}")
                        all_items_valid = False  # Type mismatch makes item invalid

        return all_items_valid  # Return True only if all items passed validation

    def _check_socketcan_overall_status(self) -> Dict[str, Any]:
        """
        Internal helper to analyze the SocketCAN interface data from the
        latest processed message and determine an overall status based on config.

        Status can be: OK, WARNING, ERROR, NO_DATA, NO_INTERFACE.

        Returns:
            Dict[str, Any]: Dictionary containing the overall status summary
                            and the list of interfaces based on processed data.
        """
        interfaces = self._chassis_data.get("socketcan_interfaces")
        config = self._config.get("chassis_checks", {}).get("socketcan", {})

        if interfaces is None:
            # Data for interfaces was not provided in the message or failed payload validation
            return {"overall_status": "NO_DATA", "message": "SocketCAN interface data not available."}

        if not interfaces:
            # List was provided but is empty - assume no CAN interfaces reported by publisher
            # This might be an ERROR depending on config/expectations, but NO_INTERFACE is specific.
            monitored_interfaces = config.get("monitored_interfaces", [])
            if monitored_interfaces:
                # Config expects interfaces, but none were reported
                return {"overall_status": "ERROR", "message": f"No SocketCAN interfaces reported, but config monitors: {monitored_interfaces}"}
            else:
                # Config doesn't specify monitored interfaces, reporting none is just informational
                return {"overall_status": "NO_INTERFACE", "message": "No SocketCAN interfaces reported."}

        all_up = True
        any_errors_exceeding_threshold = False
        any_monitored_interface_missing = False
        checked_interfaces_details: List[Dict[str, Any]] = []  # Use type hint

        require_all_up = config.get("require_all_up", True)
        max_interface_errors = config.get("max_interface_errors", 10)
        monitored_interfaces_cfg = config.get("monitored_interfaces", [])
        reported_interface_names = {iface.get(
            "name") for iface in interfaces if isinstance(iface, dict) and iface.get("name")}

        if monitored_interfaces_cfg:
            for req_iface in monitored_interfaces_cfg:
                if req_iface not in reported_interface_names:
                    any_monitored_interface_missing = True
                    logging.warning(
                        f"Monitored interface '{req_iface}' not found in reported data.")

        for interface in interfaces:
            # Assuming _validate_socketcan_interfaces_payload ensures basic structure if interfaces is not None
            interface_name = interface.get("name", "Unnamed Interface")
            interface_state = interface.get("state", "UNKNOWN")
            # Assume 0 if not reported in payload or failed validation
            tx_errors = interface.get("tx_errors", 0) if isinstance(
                interface.get("tx_errors"), int) else 0
            rx_errors = interface.get("rx_errors", 0) if isinstance(
                interface.get("rx_errors"), int) else 0
            dropped = interface.get("dropped", 0) if isinstance(
                interface.get("dropped"), int) else 0
            state_flags = interface.get("state_flags", "") if isinstance(
                interface.get("state_flags"), str) else ""

            interface_overall_ok = True  # Status for this specific interface
            messages: List[str] = []  # Messages for this specific interface

            # Check state
            if interface_state != "UP":
                all_up = False
                interface_overall_ok = False
                messages.append(f"State is {interface_state}")

            # Check errors against threshold
            total_errors = tx_errors + rx_errors + dropped
            if total_errors > max_interface_errors:
                any_errors_exceeding_threshold = True
                interface_overall_ok = False
                messages.append(
                    f"Total errors ({total_errors}) exceeds threshold ({max_interface_errors})")

            # Check if this interface is one of the explicitly monitored ones and its state is critical
            # If no specific interfaces monitored, consider all reported interfaces as relevant
            is_monitored = interface_name in monitored_interfaces_cfg if monitored_interfaces_cfg else True
            # Define critical state for an interface
            is_critical_state = (
                interface_state != "UP" or total_errors > max_interface_errors)

            checked_interfaces_details.append({
                "name": interface_name,
                "state": interface_state,
                "tx_errors": tx_errors,
                "rx_errors": rx_errors,
                "dropped": dropped,
                "state_flags": state_flags,
                "total_errors": total_errors,
                "overall_ok": interface_overall_ok,
                "messages": messages
            })

        overall_status = "OK"
        message = "All reported SocketCAN interfaces are OK."
        errors: List[str] = []

        if any_monitored_interface_missing:
            overall_status = "ERROR"
            message = "One or more configured monitored interfaces were not reported."
            errors.append(message)  # Aggregate this error

        # Evaluate overall status based on checks and config requirements
        elif require_all_up and not all_up:
            overall_status = "ERROR"
            message = "Config requires all interfaces UP, but one or more are not."
            errors.append(message)
        elif any_errors_exceeding_threshold:
            overall_status = "WARNING"  # Or ERROR depending on severity policy
            message = "One or more interfaces report errors exceeding the threshold."
            errors.append(message)
        # If not all required are UP, or errors exceed threshold, it's already ERROR/WARNING.
        # If all required are UP and no errors exceed threshold, it's OK (unless missing monitored interfaces).

        return {
            "overall_status": overall_status,
            "message": message,
            "interfaces": checked_interfaces_details,
            "errors": errors  # Include specific status-level errors here
        }

    def _check_ad_readiness(self) -> Dict[str, Any]:
        """
        Internal helper to evaluate if the chassis status indicates readiness
        for Automated Driving based on current data and configuration.

        Returns:
            Dict[str, Any]: Dictionary containing the readiness status (True/False)
                            and the reasons for failure if not ready.
        """
        readiness_info: Dict[str, Any] = {
            "is_ready": True,  # Assume ready unless a check fails
            "reasons_not_ready": []  # List of reasons if not ready
        }
        config = self._config.get("chassis_checks", {})

        if self._last_update_timestamp is None:
            readiness_info["is_ready"] = False
            readiness_info["reasons_not_ready"].append(
                "No chassis data received yet.")
            return readiness_info

        # --- Check SocketCAN Status ---
        socketcan_report = self._check_socketcan_overall_status()
        if socketcan_report.get("overall_status") != "OK":
            readiness_info["is_ready"] = False
            readiness_info["reasons_not_ready"].append(
                f"SocketCAN status is not OK ({socketcan_report.get('overall_status')}).")
            readiness_info["reasons_not_ready"].extend([f"SocketCAN Issue: {msg}" for msg in socketcan_report.get(
                "errors", [])])  # Include specific SocketCAN errors

        # --- Check DTCs ---
        critical_dtcs = self._chassis_data.get("critical_dtcs")
        fail_on_critical_dtcs = config.get(
            "dtc", {}).get("fail_on_critical_dtcs", True)

        if fail_on_critical_dtcs and critical_dtcs is not None and len(critical_dtcs) > 0:
            readiness_info["is_ready"] = False
            readiness_info["reasons_not_ready"].append(
                f"Active critical DTCs detected: {critical_dtcs}.")

        # --- Check Power Status ---
        power_cfg = config.get("power", {})
        power_stable_reported = self._chassis_data.get("power_stable")
        min_voltage_v = power_cfg.get("min_voltage_v")
        min_soc_percent = power_cfg.get("min_soc_percent")
        battery_voltage_v = self._chassis_data.get("battery_voltage_v")
        battery_soc_percent = self._chassis_data.get("battery_soc_percent")

        if power_cfg.get("require_stable", True) and power_stable_reported is not True:
            readiness_info["is_ready"] = False
            # Be specific if power_stable was explicitly False vs None (data missing)
            reason = "Power is not reported as stable." if power_stable_reported is False else "Power stability status not available."
            readiness_info["reasons_not_ready"].append(reason)

        # Check voltage against threshold if configured and data is available
        if min_voltage_v is not None and battery_voltage_v is not None:
            if battery_voltage_v < min_voltage_v:
                readiness_info["is_ready"] = False
                readiness_info["reasons_not_ready"].append(
                    f"Battery voltage ({battery_voltage_v}V) is below minimum required ({min_voltage_v}V).")
        elif min_voltage_v is not None:  # Threshold set but data is None
            readiness_info["is_ready"] = False
            readiness_info["reasons_not_ready"].append(
                f"Battery voltage data not available, but minimum required voltage ({min_voltage_v}V) is configured.")

        # Check SOC against threshold if configured and data is available
        if min_soc_percent is not None and battery_soc_percent is not None:
            if battery_soc_percent < min_soc_percent:
                readiness_info["is_ready"] = False
                readiness_info["reasons_not_ready"].append(
                    f"Battery SOC ({battery_soc_percent}%) is below minimum required ({min_soc_percent}%).")
        elif min_soc_percent is not None:  # Threshold set but data is None
            readiness_info["is_ready"] = False
            readiness_info["reasons_not_ready"].append(
                f"Battery SOC data not available, but minimum required SOC ({min_soc_percent}%) is configured.")

        # --- Check Actuator Readiness ---
        actuators_cfg = config.get("actuators", {})
        brake_ready = self._chassis_data.get("brake_ready")
        throttle_ready = self._chassis_data.get("throttle_ready")
        steer_ready = self._chassis_data.get("steer_ready")

        if actuators_cfg.get("require_brake_ready", True) and brake_ready is not True:
            readiness_info["is_ready"] = False
            reason = "Brake not reported as ready." if brake_ready is False else "Brake readiness status not available."
            readiness_info["reasons_not_ready"].append(reason)

        if actuators_cfg.get("require_throttle_ready", True) and throttle_ready is not True:
            readiness_info["is_ready"] = False
            reason = "Throttle not reported as ready." if throttle_ready is False else "Throttle readiness status not available."
            readiness_info["reasons_not_ready"].append(reason)

        if actuators_cfg.get("require_steer_ready", True) and steer_ready is not True:
            readiness_info["is_ready"] = False
            reason = "Steer not reported as ready." if steer_ready is False else "Steer readiness status not available."
            readiness_info["reasons_not_ready"].append(reason)

        return readiness_info

    def get_full_status_report(self) -> Dict[str, Any]:
        """
        Compiles and returns a comprehensive report of the latest processed
        chassis status, including SocketCAN details and AD readiness.

        Returns:
            Dict[str, Any]: A dictionary containing all available chassis
                            status information, analysis results, and the
                            overall AD readiness state.
        """
        if self._config.get('initialization_error'):
            return {
                "report_status": "ERROR",
                "message": "Report cannot be generated due to configuration errors.",
                "errors": self._config['initialization_error']
            }

        if self._last_update_timestamp is None:
            return {"report_status": "No chassis data received yet."}

        # Get status details using internal helpers
        socketcan_report = self._check_socketcan_overall_status()
        # Get the detailed readiness info
        ad_readiness_report = self._check_ad_readiness()

        # Determine overall report status (e.g., OK, WARNING, ERROR)
        # This is distinct from AD readiness, and indicates the health of the data and checks themselves.
        report_status = "OK"
        report_message = "Chassis status report generated successfully."
        report_errors: List[str] = []

        # Aggregate errors from sub-checks for the report's error list
        report_errors.extend(socketcan_report.get("errors", []))
        # Note: _check_ad_readiness returns reasons_not_ready, not errors for the report itself.
        # Decision: Include reasons_not_ready in the report structure, not the report_errors list.

        # If AD readiness is False, the report overall might be considered WARNING or ERROR
        if not ad_readiness_report["is_ready"]:
            report_status = "WARNING"  # Or "ERROR" depending on policy
            report_message = "Chassis systems not ready for Automated Driving."

        report = {
            "report_status": report_status,
            "message": report_message,
            "last_update_timestamp": self._last_update_timestamp,
            # Add ISO format timestamp
            "last_update_time_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self._last_update_timestamp)),

            # Include analysis results
            "socketcan_status": socketcan_report,
            "ad_readiness": ad_readiness_report,  # Include the detailed readiness report

            # Include the raw processed data for reference
            "latest_chassis_data": self._chassis_data,

            # Include errors encountered during processing or sub-checks (if any)
            # Use report_errors key to distinguish from ad_readiness reasons
            "report_errors": report_errors
        }

        # Add a timestamp to the report generation itself
        report["report_generated_timestamp"] = time.time()
        report["report_generated_time_iso"] = time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime(report["report_generated_timestamp"]))

        return report


# --- Example Usage ---
if __name__ == "__main__":
    # Define a sample configuration
    SAMPLE_CONFIG = {
        "chassis_checks": {
            "socketcan": {
                "require_all_up": True,
                "max_interface_errors": 50,  # Allow up to 50 errors/dropped before warning/failure
                # We expect data for these two interfaces
                "monitored_interfaces": ["can0", "can1"]
            },
            "power": {
                "require_stable": True,
                "min_voltage_v": 12.5,  # Minimum required battery voltage for AD
                "min_soc_percent": 30.0  # Minimum required SOC for AD
            },
            "actuators": {
                "require_brake_ready": True,
                "require_throttle_ready": True,
                "require_steer_ready": True
            },
            "dtc": {
                "fail_on_critical_dtcs": True  # If True, any critical DTC fails AD readiness
            }
        }
    }

    # Define a configuration with less strict requirements
    LESS_STRICT_CONFIG = {
        "chassis_checks": {
            "socketcan": {
                "require_all_up": False,  # Don't require ALL UP, just check errors
                "max_interface_errors": 100,  # Higher threshold
                "monitored_interfaces": []  # Don't check for specific interface names
            },
            "power": {
                "require_stable": False,  # Don't require power_stable boolean be True
                "min_voltage_v": 11.0,  # Lower voltage threshold
                "min_soc_percent": None  # No minimum SOC requirement
            },
            "actuators": {
                "require_brake_ready": True,
                "require_throttle_ready": False,  # Throttle not required
                "require_steer_ready": True
            },
            "dtc": {
                "fail_on_critical_dtcs": False  # Ignore critical DTCs for AD readiness
            }
        }
    }

    print("--- Initializing with SAMPLE_CONFIG ---")
    monitor = VehicleHealthChecker(SAMPLE_CONFIG)

    # --- Simulate receiving messages ---

    # Message 1: All OK state with SocketCAN data
    print("\n--- Simulating Message 1 (OK State) ---")
    ok_message = {
        "socketcan_interfaces": [
            {"name": "can0", "state": "UP", "tx_errors": 5,
                "rx_errors": 2, "dropped": 0, "state_flags": "RUNNING"},  # Errors below threshold
            {"name": "can1", "state": "UP", "tx_errors": 0,
                "rx_errors": 0, "dropped": 0, "state_flags": "RUNNING"}
        ],
        "critical_dtcs": [],
        "battery_voltage_v": 13.8,
        "battery_soc_percent": 95.0,
        "power_stable": True,
        "brake_ready": True,
        "throttle_ready": True,
        "steer_ready": True
    }
    monitor.process_message(ok_message)

    # Get and print the full status report
    print("\n--- Current Status Report after Message 1 ---")
    import json
    print(json.dumps(monitor.get_full_status_report(), indent=4))

    # Message 2: Error state (CAN errors above threshold, low power, actuator not ready, DTCs, missing CAN interface)
    print("\n--- Simulating Message 2 (Error State) ---")
    error_message = {
        "socketcan_interfaces": [
            {"name": "can0", "state": "UP", "tx_errors": 60, "rx_errors": 10,
                "dropped": 5, "state_flags": "RUNNING"},  # Errors exceed threshold
            # can1 is missing from reported interfaces
            {"name": "can2", "state": "UP", "tx_errors": 0, "rx_errors": 0,
                "dropped": 0, "state_flags": "RUNNING"}  # An extra interface not in config
        ],
        "critical_dtcs": ["DTC B1000 - CAN Bus Off", "DTC P0500 - Speed Sensor A Malfunction"],
        "battery_voltage_v": 11.5,  # Below min_voltage_v (12.5)
        "battery_soc_percent": 25.0,  # Below min_soc_percent (30.0)
        "power_stable": False,  # Not stable
        "brake_ready": True,
        "throttle_ready": False,   # Not ready
        "steer_ready": True
    }
    monitor.process_message(error_message)

    # Get and print the full status report
    print("\n--- Current Status Report after Message 2 ---")
    print(json.dumps(monitor.get_full_status_report(), indent=4))

    # Message 3: Invalid message type
    print("\n--- Simulating Message 3 (Invalid Type) ---")
    invalid_message = "This is not a dictionary"
    monitor.process_message(invalid_message)

    # Status report should reflect the state before the invalid message (Message 2)
    print("\n--- Current Status Report after Invalid Message ---")
    print(json.dumps(monitor.get_full_status_report(), indent=4))

    # Message 4: Missing/Wrong Type for SocketCAN interfaces Payload
    print("\n--- Simulating Message 4 (Bad SocketCAN Data Payload) ---")
    bad_message_payload = {
        "socketcan_interfaces": "should_be_a_list",  # Wrong type for the key itself
        "critical_dtcs": [],
        "battery_voltage_v": 13.0,
        "battery_soc_percent": 70,
        "power_stable": True,
        "brake_ready": True,
        "throttle_ready": True,
        "steer_ready": True,
    }
    monitor.process_message(bad_message_payload)

    # Status report - SocketCAN status should show NO_DATA due to payload validation failure
    print("\n--- Current Status Report after Bad SocketCAN Data Payload ---")
    print(json.dumps(monitor.get_full_status_report(), indent=4))

    print("\n--- Simulating Message 4b (Bad SocketCAN Interface Item Payload) ---")
    bad_message_payload_item = {
        "socketcan_interfaces": [
            {"name": "can0", "state": "UP"},  # OK
            {"name": "can1", "state": 5}  # State has wrong type
        ],
        "critical_dtcs": [],
        "battery_voltage_v": 13.0,
        "battery_soc_percent": 70,
        "power_stable": True,
        "brake_ready": True,
        "throttle_ready": True,
        "steer_ready": True,
    }
    monitor.process_message(bad_message_payload_item)

    # Status report - SocketCAN status should show NO_DATA due to payload item validation failure
    print("\n--- Current Status Report after Bad SocketCAN Interface Item Payload ---")
    print(json.dumps(monitor.get_full_status_report(), indent=4))

    # Message 5: Missing optional keys in payload
    print("\n--- Simulating Message 5 (Missing Optional Keys Payload) ---")
    missing_keys_message = {
        "socketcan_interfaces": [
            {"name": "can0", "state": "UP"}  # Minimal info for the list item
        ],
        "critical_dtcs": [],
        # battery_voltage_v, battery_soc_percent, power_stable are missing
        # actuator readiness keys are missing
    }
    monitor.process_message(missing_keys_message)

    # Status report - missing keys should result in None in latest_chassis_data
    # AD readiness checks using 'is not True' will fail for missing boolean keys
    # Threshold checks for missing voltage/SOC will fail if thresholds are configured
    print("\n--- Current Status Report after Missing Optional Keys Payload ---")
    print(json.dumps(monitor.get_full_status_report(), indent=4))

    print("\n--- Initializing with LESS_STRICT_CONFIG ---")
    monitor_less_strict = VehicleHealthChecker(LESS_STRICT_CONFIG)

    # Simulate Message 2 (Error State from before) with the less strict config
    print("\n--- Simulating Message 2 (Error State) with LESS_STRICT_CONFIG ---")
    # Use the same error_message as before
    monitor_less_strict.process_message(error_message)

    # Get and print the full status report - some checks might pass now
    print("\n--- Current Status Report after Message 2 with LESS_STRICT_CONFIG ---")
    print(json.dumps(monitor_less_strict.get_full_status_report(), indent=4))
