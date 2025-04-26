import os
import time
import json
import logging
import socket
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Callable

# Try importing numpy and OpenCV, handle if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logging.warning("Numpy not found. Data stream quality calculations will be limited.")
    NUMPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logging.warning("OpenCV not found. Camera image analysis checks will be skipped.")
    OPENCV_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Models ---
# 定义健康状态的枚举，增加可读性
class HealthStatus:
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SKIPPED = "SKIPPED"
    NOT_CONFIGURED = "NOT_CONFIGURED" # 明确表示未配置相关检查

@dataclass
class HealthReport:
    """
    Represents the result of a single health check or a category of checks.
    """
    status: str = HealthStatus.HEALTHY # Use predefined status strings
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensorReport:
    """
    Aggregates the health reports for a single sensor across all check categories.
    """
    name: str
    type: str
    overall_status: str
    connectivity: HealthReport
    data_stream: HealthReport
    internal_health: HealthReport
    # errors: List[str] = field(default_factory=list) # Removed, errors are now within HealthReport messages

# --- Middleware Abstraction ---
class MiddlewareInterface(ABC):
    """
    Abstract Base Class for middleware interaction.
    Defines methods to retrieve sensor data or information from the middleware.
    """
    @abstractmethod
    def get_topic_info(self, topic: str, duration_sec: float) -> List[Dict[str, Any]]:
        """
        Subscribes to a topic for a duration and collects message metadata/content.
        Returns a list of dicts, each with:
        {'timestamp_ros', 'timestamp_sensor', 'size_bytes', 'seq_num', 'message'}
        - 'message' field should contain the parsed sensor message object/dict.
        """
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """Checks if the middleware client is currently connected and ready."""
        raise NotImplementedError

    @abstractmethod
    def connect(self, **kwargs):
        """Connects to the middleware."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        """Disconnects from the middleware."""
        raise NotImplementedError


class PlaceholderMiddleware(MiddlewareInterface):
    """
    A more advanced placeholder that simulates realistic data streams for testing.
    This class is for demonstration and testing purposes.
    """
    def __init__(self, simulation_settings: Optional[Dict[str, Any]] = None):
        logging.info("Initializing PlaceholderMiddleware for simulation.")
        self.seq_counters: Dict[str, int] = {}
        self.is_connected_flag: bool = False
        self.simulation_settings = simulation_settings or {
            'lidar': {'rate': 10.0, 'base_latency': 0.03, 'jitter': 0.005, 'size': 120000},
            'camera': {'rate': 30.0, 'base_latency': 0.04, 'jitter': 0.01, 'size': 2073600}, # 1920x1080
            'gps': {'rate': 10.0, 'base_latency': 0.01, 'jitter': 0.002, 'size': 500},
            'packet_loss_prob': 0.01, # Global packet loss probability
        }

    def connect(self, **kwargs):
        logging.info("PlaceholderMiddleware: Simulating connection...")
        time.sleep(0.1) # Simulate connection time
        self.is_connected_flag = True
        logging.info("PlaceholderMiddleware: Simulated connection successful.")

    def disconnect(self):
        logging.info("PlaceholderMiddleware: Simulating disconnection...")
        self.is_connected_flag = False
        logging.info("PlaceholderMiddleware: Simulated disconnected.")

    def is_connected(self) -> bool:
        return self.is_connected_flag

    def get_topic_info(self, topic: str, duration_sec: float) -> List[Dict[str, Any]]:
        if not self.is_connected_flag:
            logging.warning(f"PlaceholderMiddleware not connected. Cannot get info for topic '{topic}'.")
            return []

        self.seq_counters.setdefault(topic, 0)
        infos = []

        # Determine simulation parameters based on topic keywords
        params = {}
        if 'lidar' in topic: params = self.simulation_settings.get('lidar', {})
        elif 'camera' in topic: params = self.simulation_settings.get('camera', {})
        elif 'gps' in topic: params = self.simulation_settings.get('gps', {})

        rate = params.get('rate', 0.0)
        base_latency = params.get('base_latency', 0.0)
        jitter = params.get('jitter', 0.0)
        size = params.get('size', 0)
        packet_loss_prob = self.simulation_settings.get('packet_loss_prob', 0.0)

        if rate == 0:
            logging.debug(f"No simulation settings found for topic: {topic}")
            return []

        start_time = time.time()
        # Ensure at least one message is simulated if duration > 0 and rate > 0
        num_messages_to_simulate = max(1, int(rate * duration_sec)) if duration_sec > 0 else 0

        for i in range(num_messages_to_simulate):
            # Simulate occasional packet loss
            if NUMPY_AVAILABLE and np.random.rand() < packet_loss_prob:
                self.seq_counters[topic] += 1 # Sequence number still increments for lost packet
                continue

            timestamp_ros = start_time + (i / rate) + (np.random.normal(0, jitter) if NUMPY_AVAILABLE else 0)
            timestamp_sensor = timestamp_ros - base_latency

            # Simulate specific message content for internal checks
            message = {}
            if 'gps' in topic:
                # Example GPS status from ROS NavSatFix, 2=3D fix, 3=3D differential fix
                message = {'status': np.random.choice([0,1,2,3]) if NUMPY_AVAILABLE else 2,
                           'satellites': np.random.randint(4, 20) if NUMPY_AVAILABLE else 12,
                           'hdop': round(np.random.uniform(0.8, 3.0), 2) if NUMPY_AVAILABLE else 1.1}
            elif 'camera' in topic and OPENCV_AVAILABLE:
                # Create a sample image (e.g., noisy gray)
                # For realistic simulation, ensure the size matches expected image dimensions
                sim_width, sim_height = 1920, 1080 # Example
                message = {'data': np.random.randint(0, 256, (sim_height, sim_width), dtype=np.uint8)}

            infos.append({
                'timestamp_ros': timestamp_ros,
                'timestamp_sensor': timestamp_sensor,
                'size_bytes': int(size * (1 + (np.random.uniform(-0.05, 0.05) if NUMPY_AVAILABLE else 0))),
                'seq_num': self.seq_counters[topic],
                'message': message
            })
            self.seq_counters[topic] += 1
        return infos


# --- Health Check Logic ---
class CheckExecutor:
    """
    Contains static methods for various sensor health checks.
    Each method takes specific parameters and returns a HealthReport.
    """

    # --- Connectivity Checks ---
    @staticmethod
    def device_file(path: str) -> HealthReport:
        """Checks if a device file exists and is accessible."""
        logging.debug(f"Checking device file: {path}")
        if not path:
            return HealthReport(HealthStatus.CRITICAL, "Device file path not provided.")
        if os.path.exists(path):
            # Optional: Add os.stat(path) checks for permissions or device type
            # try:
            #     stat_info = os.stat(path)
            #     if os.stat.S_ISCHR(stat_info.st_mode) or os.stat.S_ISBLK(stat_info.st_mode):
            #         return HealthReport(HealthStatus.HEALTHY, f"Device file exists and is accessible: {path}")
            #     else:
            #         return HealthReport(HealthStatus.WARNING, f"Path exists but is not a device file: {path}")
            # except Exception as e:
            #     return HealthReport(HealthStatus.WARNING, f"Device file exists but could not get status ({path}): {e}")
            return HealthReport(HealthStatus.HEALTHY, f"Device file '{path}' found.")
        return HealthReport(HealthStatus.CRITICAL, f"Device file '{path}' not found.")

    @staticmethod
    def network_port(host: str, port: int, timeout: float = 1.0) -> HealthReport:
        """Checks if a network port is open and reachable."""
        logging.debug(f"Checking network port: {host}:{port}")
        if not host or not isinstance(port, int):
            return HealthReport(HealthStatus.CRITICAL, "Network host or port not provided/invalid.")
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return HealthReport(HealthStatus.HEALTHY, f"Network port {host}:{port} is open.")
        except ConnectionRefusedError:
            return HealthReport(HealthStatus.CRITICAL, f"Network port refused connection: {host}:{port}")
        except socket.timeout:
            return HealthReport(HealthStatus.CRITICAL, f"Network port connection timed out: {host}:{port}")
        except socket.gaierror:
            return HealthReport(HealthStatus.CRITICAL, f"Network address resolution error (unknown host): {host}")
        except Exception as e:
            return HealthReport(HealthStatus.CRITICAL, f"Error checking network port {host}:{port}: {e}")

    # --- Data Stream Checks ---
    @staticmethod
    def data_stream_quality(middleware: MiddlewareInterface, topic: str, duration_sec: float, **thresholds) -> HealthReport:
        """
        Analyzes data stream quality based on rate, latency, jitter, packet loss, and message size.
        Requires a middleware interface to retrieve topic info.
        """
        if not middleware.is_connected():
            return HealthReport(HealthStatus.SKIPPED, "Middleware not connected, cannot check data stream.")
        if not topic:
            return HealthReport(HealthStatus.CRITICAL, "Topic name not provided for data stream check.")
        if not NUMPY_AVAILABLE:
             return HealthReport(HealthStatus.SKIPPED, "Numpy not available, advanced data stream quality checks skipped.")

        logging.debug(f"Collecting data for topic '{topic}' for {duration_sec} seconds...")
        infos = middleware.get_topic_info(topic, duration_sec)
        count = len(infos)

        if count == 0:
            return HealthReport(HealthStatus.CRITICAL, f"No messages received on '{topic}' in {duration_sec}s.", {'topic': topic})

        # --- Sub-check calculations ---
        actual_rate = count / duration_sec
        latencies = [info['timestamp_ros'] - info['timestamp_sensor'] for info in infos if 'timestamp_sensor' in info]
        avg_latency = np.mean(latencies) if latencies else 0

        # Jitter calculation: standard deviation of time differences between consecutive messages
        if count > 1:
            time_diffs = [infos[i]['timestamp_ros'] - infos[i-1]['timestamp_ros'] for i in range(1, count)]
            jitter = np.std(time_diffs) if time_diffs else 0
        else:
            jitter = 0

        # Packet loss calculation based on sequence numbers
        seq_nums = sorted([info['seq_num'] for info in infos])
        packet_loss_pct = 0.0
        if len(seq_nums) > 1:
            # Check for gaps in sequence numbers
            expected_total_packets = seq_nums[-1] - seq_nums[0] + 1
            actual_received_unique_packets = len(set(seq_nums)) # Use set to handle potential duplicate seq_nums
            if expected_total_packets > 0:
                packet_loss_pct = ((expected_total_packets - actual_received_unique_packets) / expected_total_packets) * 100
        elif len(seq_nums) == 1:
            # Only one packet received, cannot determine loss without more context
            packet_loss_pct = 0.0 # Assume no loss for single packet

        avg_size = np.mean([info['size_bytes'] for info in infos]) if infos else 0

        # --- Evaluation ---
        details = {
            'topic': topic,
            'actual_rate_hz': round(actual_rate, 2),
            'avg_latency_ms': round(avg_latency * 1000, 2),
            'jitter_ms': round(jitter * 1000, 2),
            'packet_loss_pct': round(packet_loss_pct, 2),
            'avg_size_kb': round(avg_size / 1024, 2)
        }
        issues = []
        overall_status = HealthStatus.HEALTHY

        # Rate check
        expected_rate_hz = thresholds.get('expected_rate_hz')
        rate_tolerance_pct = thresholds.get('rate_tolerance_pct', 10.0) # Default 10%
        if expected_rate_hz is not None:
            lower_bound = expected_rate_hz * (1 - rate_tolerance_pct / 100.0)
            upper_bound = expected_rate_hz * (1 + rate_tolerance_pct / 100.0)
            if not (lower_bound <= actual_rate <= upper_bound):
                issues.append(f"Rate {details['actual_rate_hz']}Hz outside expected range [{round(lower_bound, 2)}, {round(upper_bound, 2)}]")
                overall_status = HealthStatus.WARNING

        # Latency check
        max_latency_ms = thresholds.get('max_latency_ms')
        if max_latency_ms is not None and details['avg_latency_ms'] > max_latency_ms:
            issues.append(f"Avg latency {details['avg_latency_ms']}ms > {max_latency_ms}ms")
            overall_status = HealthStatus.WARNING

        # Jitter check
        max_jitter_ms = thresholds.get('max_jitter_ms')
        if max_jitter_ms is not None and details['jitter_ms'] > max_jitter_ms:
            issues.append(f"Jitter {details['jitter_ms']}ms > {max_jitter_ms}ms")
            overall_status = HealthStatus.WARNING

        # Loss check
        max_loss_pct = thresholds.get('max_loss_pct')
        if max_loss_pct is not None and details['packet_loss_pct'] > max_loss_pct:
            issues.append(f"Packet loss {details['packet_loss_pct']}% > {max_loss_pct}%")
            # Packet loss is often critical, override WARNING
            overall_status = HealthStatus.CRITICAL

        # Size check (useful for sanity check, e.g., camera delivering zero-byte images)
        min_size_kb = thresholds.get('min_size_kb')
        if min_size_kb is not None and details['avg_size_kb'] < min_size_kb:
            issues.append(f"Avg size {details['avg_size_kb']}KB < {min_size_kb}KB")
            if overall_status != HealthStatus.CRITICAL: # Don't downgrade from CRITICAL
                overall_status = HealthStatus.WARNING

        if not issues:
            return HealthReport(HealthStatus.HEALTHY, "Data stream quality is good.", details)
        return HealthReport(overall_status, ", ".join(issues), details)

    # --- Internal Health Checks ---
    @staticmethod
    def gps_fix_quality(middleware: MiddlewareInterface, topic: str, duration_sec: float = 1.0, **params) -> HealthReport:
        """
        Checks GPS fix quality based on satellites, HDOP, and fix status.
        Assumes the 'message' in topic_info contains 'status', 'satellites', 'hdop' keys.
        """
        if not middleware.is_connected():
            return HealthReport(HealthStatus.SKIPPED, "Middleware not connected, cannot check GPS fix quality.")
        if not topic:
            return HealthReport(HealthStatus.CRITICAL, "Topic name not provided for GPS check.")

        # Just need one recent message for GPS status
        infos = middleware.get_topic_info(topic, duration_sec=duration_sec)
        if not infos:
            return HealthReport(HealthStatus.CRITICAL, f"No GPS message received on topic '{topic}' in {duration_sec}s.")

        # Use the latest message
        msg = infos[-1]['message'] # Assuming latest message is best

        sats = msg.get('satellites')
        hdop = msg.get('hdop')
        fix_status = msg.get('status')
        details = {'topic': topic, 'satellites': sats, 'hdop': hdop, 'fix_status': fix_status}

        issues = []
        overall_status = HealthStatus.HEALTHY

        min_sats = params.get('min_sats')
        if min_sats is not None and (sats is None or sats < min_sats):
            issues.append(f"Satellites ({sats}) < required {min_sats}")
            overall_status = HealthStatus.WARNING

        max_hdop = params.get('max_hdop')
        if max_hdop is not None and (hdop is None or hdop > max_hdop):
            issues.append(f"HDOP ({hdop}) > allowed {max_hdop}")
            overall_status = HealthStatus.WARNING

        required_fix_status = params.get('required_fix_status') # e.g., [2, 3] for 3D fix
        if required_fix_status is not None and (fix_status is None or fix_status not in required_fix_status):
            issues.append(f"Fix status ({fix_status}) not in required {required_fix_status}")
            # Wrong fix type is often critical for AD, override WARNING
            overall_status = HealthStatus.CRITICAL

        if issues:
            return HealthReport(overall_status, ", ".join(issues), details)
        return HealthReport(HealthStatus.HEALTHY, "GPS fix quality is good.", details)

    @staticmethod
    def camera_image_analysis(middleware: MiddlewareInterface, topic: str, duration_sec: float = 1.0, **params) -> HealthReport:
        """
        Performs basic image quality analysis (blur, brightness) on a camera stream.
        Requires OpenCV and assumes 'message' in topic_info contains image 'data' (numpy array).
        """
        if not OPENCV_AVAILABLE:
            return HealthReport(HealthStatus.SKIPPED, "OpenCV not installed, camera image analysis skipped.")
        if not middleware.is_connected():
            return HealthReport(HealthStatus.SKIPPED, "Middleware not connected, cannot check camera image quality.")
        if not topic:
            return HealthReport(HealthStatus.CRITICAL, "Topic name not provided for camera image analysis.")

        infos = middleware.get_topic_info(topic, duration_sec=duration_sec)
        if not infos:
            return HealthReport(HealthStatus.CRITICAL, f"No image message received on topic '{topic}' in {duration_sec}s.")

        img_data = infos[-1]['message'].get('data') # Use the latest image
        if not isinstance(img_data, np.ndarray):
            return HealthReport(HealthStatus.CRITICAL, "Image data in message is not a valid numpy array.")
        if img_data.ndim not in [2, 3]: # Grayscale or Color
             return HealthReport(HealthStatus.CRITICAL, f"Invalid image dimensions: {img_data.shape}. Expected 2D or 3D.")

        # Ensure image is grayscale for blur detection if it's color
        if img_data.ndim == 3:
            gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img_data

        # 1. Blur detection (Laplacian variance)
        # Low variance indicates low contrast, often blurry image
        blur_score = cv2.Laplacian(gray_img, cv2.CV_64F).var()

        # 2. Brightness check (mean pixel value)
        brightness = gray_img.mean()

        details = {'topic': topic, 'blur_score': round(blur_score, 2), 'brightness': round(brightness, 2)}
        issues = []
        overall_status = HealthStatus.HEALTHY

        min_blur_score = params.get('min_blur_score')
        if min_blur_score is not None and blur_score < min_blur_score:
            issues.append(f"Image may be blurry/occluded (score: {details['blur_score']} < {min_blur_score})")
            overall_status = HealthStatus.WARNING

        brightness_range = params.get('brightness_range') # e.g., [50, 200] for 0-255 scale
        if brightness_range is not None and (brightness < brightness_range[0] or brightness > brightness_range[1]):
            issues.append(f"Image is too dark/bright (brightness: {details['brightness']} outside {brightness_range})")
            overall_status = HealthStatus.WARNING

        if issues:
            return HealthReport(overall_status, ", ".join(issues), details)
        return HealthReport(HealthStatus.HEALTHY, "Image quality appears normal.", details)


# --- Main Health Monitor Class (Refactored) ---
class SensorHealthMonitor:
    """
    Orchestrates the execution of sensor health checks based on a configuration.
    """
    def __init__(self, config: Dict[str, Any], middleware: MiddlewareInterface):
        if not isinstance(config, dict) or 'sensors' not in config or not isinstance(config['sensors'], list):
            logging.error("Invalid configuration format. Expected a dict with a 'sensors' list.")
            self.config = {'sensors': []}
            self.initialization_error = "Invalid configuration format."
        else:
            self.config = config
            self.initialization_error = None

        self.middleware = middleware

        # Define a registry of available check functions
        self.check_registry: Dict[str, Callable[..., HealthReport]] = {
            'device_file': CheckExecutor.device_file,
            'network_port': CheckExecutor.network_port,
            'data_stream_quality': CheckExecutor.data_stream_quality,
            'gps_fix_quality': CheckExecutor.gps_fix_quality,
            'camera_image_analysis': CheckExecutor.camera_image_analysis,
        }
        logging.info("SensorHealthMonitor initialized with check registry.")
        if not NUMPY_AVAILABLE:
            logging.warning("Numpy is not available. Some advanced checks will be skipped.")
        if not OPENCV_AVAILABLE:
            logging.warning("OpenCV is not available. Camera image analysis will be skipped.")


    def _execute_check(self, check_type: str, check_params: Dict[str, Any], sensor_name: str) -> HealthReport:
        """
        Executes a specific health check using the registry.
        Handles errors during check execution.
        """
        if check_type not in self.check_registry:
            logging.error(f"Sensor '{sensor_name}': Unknown check type '{check_type}'. Skipping.")
            return HealthReport(HealthStatus.CRITICAL, f"Unknown check type '{check_type}'.")

        # Prepare parameters, injecting middleware where needed
        # Create a copy to avoid modifying the original config dict
        params_for_check = check_params.copy()
        if self.check_registry[check_type] in [CheckExecutor.data_stream_quality,
                                               CheckExecutor.gps_fix_quality,
                                               CheckExecutor.camera_image_analysis]:
            params_for_check['middleware'] = self.middleware

        try:
            # Use **kwargs to pass dictionary as named arguments
            return self.check_registry[check_type](**params_for_check)
        except TypeError as te:
            # Catch argument errors if config doesn't match function signature
            logging.error(f"Sensor '{sensor_name}' check '{check_type}': Parameter mismatch. Error: {te}")
            return HealthReport(HealthStatus.CRITICAL, f"Configuration error for check '{check_type}': {te}")
        except Exception as e:
            logging.error(f"Sensor '{sensor_name}' check '{check_type}': Exception during execution: {e}", exc_info=True)
            return HealthReport(HealthStatus.CRITICAL, f"Exception during check '{check_type}': {e}")


    def _run_check_category(self,
                            sensor_cfg_category: Optional[List[Dict[str, Any]]],
                            sensor_name: str,
                            category_name: str
                           ) -> HealthReport:
        """
        Runs all checks within a specific category (e.g., connectivity, data_stream).
        Aggregates results into a single HealthReport for the category.
        """
        if not sensor_cfg_category:
            return HealthReport(HealthStatus.NOT_CONFIGURED, f"No {category_name} checks configured.")

        if not isinstance(sensor_cfg_category, list):
            logging.error(f"Sensor '{sensor_name}': Invalid format for '{category_name}' checks. Expected a list.")
            return HealthReport(HealthStatus.CRITICAL, f"Invalid configuration for {category_name} checks.")

        individual_reports: List[HealthReport] = []
        overall_status_for_category = HealthStatus.HEALTHY
        category_messages: List[str] = []

        for check_cfg in sensor_cfg_category:
            check_type = check_cfg.get('type')
            check_params = check_cfg.get('params', {})

            if not check_type:
                individual_reports.append(HealthReport(HealthStatus.CRITICAL, "Check type not specified in config."))
                overall_status_for_category = HealthStatus.CRITICAL
                category_messages.append("Missing check type.")
                continue

            report = self._execute_check(check_type, check_params, sensor_name)
            individual_reports.append(report)
            category_messages.append(f"[{report.status}] {report.message}")

            # Aggregate status: CRITICAL > WARNING > HEALTHY > SKIPPED/NOT_CONFIGURED
            if report.status == HealthStatus.CRITICAL:
                overall_status_for_category = HealthStatus.CRITICAL
            elif report.status == HealthStatus.WARNING and overall_status_for_category != HealthStatus.CRITICAL:
                overall_status_for_category = HealthStatus.WARNING
            # SKIPPED status does not downgrade overall_status if other checks are HEALTHY
            # NOT_CONFIGURED only applies if there are no checks at all

        return HealthReport(
            overall_status_for_category,
            " | ".join(category_messages) if category_messages else f"All {category_name} checks passed.",
            {'individual_reports': [asdict(r) for r in individual_reports]}
        )


    def run_all_checks(self) -> List[SensorReport]:
        """
        Executes all configured health checks for all sensors.
        Returns a list of SensorReport objects.
        """
        if self.initialization_error:
            logging.error(f"Health Monitor not initialized properly: {self.initialization_error}")
            return [SensorReport(
                name="System", type="N/A", overall_status=HealthStatus.CRITICAL,
                connectivity=HealthReport(HealthStatus.CRITICAL, f"Initialization error: {self.initialization_error}"),
                data_stream=HealthReport(HealthStatus.SKIPPED, "Not applicable due to initialization error."),
                internal_health=HealthReport(HealthStatus.SKIPPED, "Not applicable due to initialization error.")
            )]

        logging.info("Starting all sensor health checks...")
        full_report: List[SensorReport] = []

        # Connect to middleware once before running data/internal checks
        if not self.middleware.is_connected():
            try:
                self.middleware.connect() # Pass any necessary connection params here, e.g., self.config.get('middleware_settings', {})
            except Exception as e:
                logging.error(f"Failed to connect to middleware: {e}. Data stream and internal health checks will be skipped.", exc_info=True)
                # If middleware connection fails, all data/internal health checks will be SKIPPED
                # We still proceed to run connectivity checks

        for sensor_cfg in self.config.get('sensors', []):
            name = sensor_cfg.get('name', 'UnknownSensor')
            sensor_type = sensor_cfg.get('type', 'UnknownType')
            logging.info(f"--- Checking sensor: {name} ({sensor_type}) ---")

            # Run checks for each category
            # We use .get() for robustness if a category is missing in config
            conn_report = self._run_check_category(sensor_cfg.get('connectivity'), name, 'connectivity')
            data_report = self._run_check_category(sensor_cfg.get('data_stream'), name, 'data_stream')
            health_report = self._run_check_category(sensor_cfg.get('internal_health'), name, 'internal_health')

            # Determine overall status for this sensor
            sensor_overall_status = HealthStatus.HEALTHY
            for report in [conn_report, data_report, health_report]:
                if report.status == HealthStatus.CRITICAL:
                    sensor_overall_status = HealthStatus.CRITICAL
                    break # Critical failure, no need to check further for this sensor
                if report.status == HealthStatus.WARNING and sensor_overall_status != HealthStatus.CRITICAL:
                    sensor_overall_status = HealthStatus.WARNING
                # SKIPPED or NOT_CONFIGURED status does not downgrade HEALTHY or WARNING

            full_report.append(SensorReport(name, sensor_type, sensor_overall_status,
                                            conn_report, data_report, internal_health=health_report))
            logging.info(f"--- Sensor {name} finished with overall status: {sensor_overall_status} ---")

        # Disconnect from middleware after all checks are done
        if self.middleware.is_connected():
            try:
                self.middleware.disconnect()
            except Exception as e:
                logging.error(f"Failed to disconnect from middleware: {e}", exc_info=True)


        logging.info("All sensor health checks completed.")
        return full_report


# --- Example Usage ---
if __name__ == "__main__":
    # Example sensor configuration
    # Note: 'connectivity', 'data_stream', 'internal_health' are lists of checks
    # Each check has a 'type' and 'params' dictionary
    SAMPLE_CONFIG = {
        "sensors": [
            {
                "name": "FrontLiDAR",
                "type": "LiDAR",
                "connectivity": [
                    {"type": "device_file", "params": {"path": "/dev/ttyUSB_LIDAR_A"}}, # This path should exist for HEALTHY
                    # {"type": "device_file", "params": {"path": "/dev/nonexistent_lidar"}}, # Uncomment to test CRITICAL connectivity
                ],
                "data_stream": [
                    {
                        "type": "data_stream_quality",
                        "params": {
                            "topic": "/sensing/lidar/front/points_raw", "duration_sec": 2,
                            "expected_rate_hz": 10.0, "rate_tolerance_pct": 10, # Allow 9-11Hz
                            "max_latency_ms": 60, "max_jitter_ms": 10,
                            "max_loss_pct": 0.5, # Test with 1% sim loss
                            "min_size_kb": 90 # Simulated is ~117KB
                        }
                    }
                ],
                "internal_health": [
                    # Example: a hypothetical LiDAR specific internal temp check
                    # {"type": "lidar_temp_check", "params": {"topic": "/sensing/lidar/front/temperature", "max_temp_c": 70}}
                ]
            },
            {
                "name": "FrontCamera",
                "type": "Camera",
                "connectivity": [
                    {"type": "device_file", "params": {"path": "/dev/video0"}}, # Or your camera's actual device
                    # {"type": "network_port", "params": {"host": "192.168.1.100", "port": 8080, "timeout": 0.5}}, # Example for IP camera
                ],
                "data_stream": [
                    {
                        "type": "data_stream_quality",
                        "params": {
                            "topic": "/sensing/camera/front/image_raw", "duration_sec": 1,
                            "expected_rate_hz": 30.0, "rate_tolerance_pct": 5,
                            "max_latency_ms": 70,
                            "max_loss_pct": 0.1 # Very strict loss requirement
                        }
                    }
                ],
                "internal_health": [
                    {
                        "type": "camera_image_analysis",
                        "params": {
                            "topic": "/sensing/camera/front/image_raw",
                            "min_blur_score": 50.0, # Threshold for blurriness (lower = blurrier)
                            "brightness_range": [30, 220] # Normal brightness range for 8-bit image
                        }
                    }
                ]
            },
            {
                "name": "GPS_Receiver",
                "type": "GPS/GNSS",
                "connectivity": [
                    {"type": "device_file", "params": {"path": "/dev/ttyACM0"}},
                ],
                "data_stream": [
                    {
                        "type": "data_stream_quality",
                        "params": {
                            "topic": "/sensing/gnss/fix", "duration_sec": 3,
                            "expected_rate_hz": 10.0, "rate_tolerance_pct": 20,
                            "max_loss_pct": 1.0
                        }
                    }
                ],
                "internal_health": [
                    {
                        "type": "gps_fix_quality",
                        "params": {
                            "topic": "/sensing/gnss/fix", # Use same topic, assuming it contains fix info
                            "min_sats": 6, # Minimum satellites for a good fix
                            "max_hdop": 1.8, # Maximum HDOP (lower is better)
                            "required_fix_status": [2, 3] # 2=3D fix, 3=3D differential fix
                        }
                    }
                ]
            },
            {
                "name": "SensorWithNoConfiguredChecks",
                "type": "Dummy",
                # This sensor has no checks configured, should show NOT_CONFIGURED status for all categories
            },
            {
                "name": "SensorWithInvalidConfig",
                "type": "Faulty",
                "connectivity": "not_a_list", # Invalid configuration for connectivity
            }
        ]
    }

    # Create a dummy device file for testing `device_file` check
    # In a real system, these paths would exist naturally
    dummy_device_path_lidar = "/dev/ttyUSB_LIDAR_A"
    dummy_device_path_camera = "/dev/video0"
    dummy_device_path_gps = "/dev/ttyACM0"

    # Create dummy files if they don't exist, for testing purposes
    for p in [dummy_device_path_lidar, dummy_device_path_camera, dummy_device_path_gps]:
        if not os.path.exists(p):
            try:
                # Use mknod to create a character device file for more realism,
                # but it requires root permissions. For simple testing, touch is enough.
                # os.mknod(p, stat.S_IFCHR | 0o666, os.makedev(1, 0)) # Major/minor numbers example
                with open(p, 'a'):
                    os.utime(p, None) # Simply touch the file
                logging.info(f"Created dummy device file for testing: {p}")
            except Exception as e:
                logging.warning(f"Could not create dummy device file '{p}': {e}. Connectivity check might fail.")


    # Initialize the placeholder middleware
    # You could pass simulation settings here to make specific topics fail or warn
    middleware_client = PlaceholderMiddleware(simulation_settings={
        'lidar': {'rate': 10.0, 'base_latency': 0.03, 'jitter': 0.005, 'size': 120000},
        'camera': {'rate': 30.0, 'base_latency': 0.04, 'jitter': 0.01, 'size': 2073600},
        'gps': {'rate': 10.0, 'base_latency': 0.01, 'jitter': 0.002, 'size': 500},
        'packet_loss_prob': 0.02, # Simulate 2% global packet loss to trigger warnings/criticals
    })


    # Initialize the sensor health monitor with the configuration and middleware
    monitor = SensorHealthMonitor(SAMPLE_CONFIG, middleware_client)

    # Run all checks
    final_report_list = monitor.run_all_checks()

    # Convert dataclass report to dictionary for JSON serialization
    report_dict_serializable = [asdict(r) for r in final_report_list]

    print("\n" + "="*80 + "\n" + " AUTONOMOUS DRIVING SENSOR HEALTH MONITOR REPORT ".center(80) + "\n" + "="*80)
    print(json.dumps(report_dict_serializable, indent=2))

    print("\n" + "="*80 + "\n" + " REPORT SUMMARY ".center(80) + "\n" + "="*80)
    overall_system_status = HealthStatus.HEALTHY
    for sensor_report in final_report_list:
        print(f"Sensor: {sensor_report.name} ({sensor_report.type})")
        print(f"  Overall Status: {sensor_report.overall_status}")
        if sensor_report.overall_status == HealthStatus.CRITICAL:
            overall_system_status = HealthStatus.CRITICAL
        elif sensor_report.overall_status == HealthStatus.WARNING and overall_system_status != HealthStatus.CRITICAL:
            overall_system_status = HealthStatus.WARNING

        # Print detailed status for each category if not HEALTHY or NOT_CONFIGURED
        for category_name, category_report in [
            ("Connectivity", sensor_report.connectivity),
            ("Data Stream", sensor_report.data_stream),
            ("Internal Health", sensor_report.internal_health)
        ]:
            if category_report.status not in [HealthStatus.HEALTHY, HealthStatus.NOT_CONFIGURED]:
                print(f"  {category_name} Status: {category_report.status}")
                print(f"    Message: {category_report.message}")
                if category_report.details and 'individual_reports' in category_report.details:
                    print("    Individual Checks:")
                    for ind_report in category_report.details['individual_reports']:
                        print(f"      - [{ind_report['status']}] {ind_report['message']}")
                        if ind_report['details']:
                             print(f"          Details: {json.dumps(ind_report['details'], indent=10)}")
                elif category_report.details: # For category level details
                    print(f"    Category Details: {json.dumps(category_report.details, indent=6)}")
            elif category_report.status == HealthStatus.NOT_CONFIGURED:
                 print(f"  {category_name} Status: {category_report.status} - {category_report.message}")
            else: # Healthy
                print(f"  {category_name} Status: {category_report.status}")

        print("-" * 40) # Separator for sensors

    print(f"\nSYSTEM HEALTH: {overall_system_status}")

    # Clean up dummy files
    for p in [dummy_device_path_lidar, dummy_device_path_camera, dummy_device_path_gps]:
        if os.path.exists(p):
            try:
                os.remove(p)
                logging.info(f"Removed dummy device file: {p}")
            except Exception as e:
                logging.warning(f"Could not remove dummy device file '{p}': {e}.")