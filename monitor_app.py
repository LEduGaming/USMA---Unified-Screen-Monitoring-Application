#!/usr/bin/env python3
"""
USMA (Unified Screen Monitoring Application) - v.0.3.10

A single, GUI-driven application that combines a professional-grade region 
configuration tool, real-time screen monitoring, visual overlay, and clear 
image logging.

v.0.3.10 Changes:
- Added physical scaling to regions: Users can now define the real-world units
  (e.g., Hz, g/N) corresponding to the pixel dimensions of a monitoring region.
- Enhanced data logging with physical units: .mat and .unv files now store
  data with proper frequency (Hz) and amplitude axes instead of pixel values.
- Updated .unv headers to be fully compliant with industry standards (UFF Type
  58), including configurable node/DOF info for Testlab compatibility.
- Expanded the Configuration Tool to include input fields for all new scaling
  and metadata parameters.

"""

import cv2
import numpy as np
import pyautogui
import time
import threading
import json
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List
from PIL import Image, ImageTk
from scipy.fft import rfft, rfftfreq
import scipy.io
import matplotlib
# Use the 'Agg' backend for non-interactive plotting in a thread.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Import sounddevice with fallback ---
try:
    import sounddevice as sd
    SOUND_DEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUND_DEVICE_AVAILABLE = False
    print(f"Warning: sounddevice library not found or audio device error: {e}. "
          "Audio feedback will be disabled. Install with: pip install sounddevice")


# --- 1. SETUP: DIRECTORY AND LOGGING CONFIGURATION ---
def setup_environment():
    """Create necessary directories for logs, configs, and image logs."""
    for folder in ['logs', 'configs', 'image_logs', 'signal_logs']:
        if not os.path.exists(folder):
            os.makedirs(folder)

setup_environment()

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
file_handler = logging.FileHandler('logs/monitor_app.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# --- 2. DATA CLASSES: CORE DATA STRUCTURES ---
@dataclass
class ImageLogOptions:
    """Stores user preferences for image log content."""
    include_screenshot: bool = False
    include_color_filter: bool = False
    include_signal_plot: bool = False
    include_fft_plot: bool = False

@dataclass
class DataLogOptions:
    """Stores user preferences for data file logging."""
    log_mat: bool = False
    log_unv: bool = False

@dataclass
class MonitoringRegion:
    """Defines a region of interest (ROI) on the screen with physical scaling."""
    name: str
    x: int
    y: int
    width: int
    height: int
    roi_type: str
    enabled: bool = field(default=True)
    # Physical Scaling Parameters
    x_axis_min: float = field(default=0.0)
    x_axis_max: float = field(default=1024.0)
    y_axis_min: float = field(default=0.0)
    y_axis_max: float = field(default=1.0)
    y_axis_unit: str = field(default="g/N")
    # UNV Header Metadata
    resp_node: int = field(default=1)
    resp_dof: int = field(default=3)
    ref_node: int = field(default=1)
    ref_dof: int = field(default=3)

@dataclass
class WaveAnalysisResult:
    """Stores the complete results of a wave pattern analysis."""
    is_high_frequency: bool
    energy_ratio: float
    high_freq_energy: float
    signal_vector: np.ndarray # Amplitude in pixels
    fft_freqs: np.ndarray
    fft_mags: np.ndarray
    roi_image: np.ndarray
    color_mask: np.ndarray

@dataclass
class AppConfig:
    """Stores global application settings, including dynamic color thresholds."""
    regions: Dict[str, MonitoringRegion] = field(default_factory=dict)
    hsv_lower: List[int] = field(default_factory=lambda: [0, 0, 0])
    hsv_upper: List[int] = field(default_factory=lambda: [179, 255, 240])
    screenshot_interval: float = 0.25  # Default to 4 Hz
    fft_cutoff_frequency: float = 0.09
    fft_energy_ratio_threshold: float = 0.013


# --- 3. CORE LOGIC: THE SCREEN MONITOR ENGINE ---
class ScreenMonitor:
    """Handles the core task of capturing and analyzing the screen."""
    def __init__(self, config_path, update_callback=None):
        self.running = False
        self.thread = None
        self.config_path = config_path
        self.app_config = self._load_config(self.config_path)
        self.update_callback = update_callback
        self.frame_count = 0
        self.verbose_logging_enabled = True
        self.image_logging_enabled = True
        self.image_log_options = ImageLogOptions()
        self.data_log_options = DataLogOptions()
        self.last_logged_ratio: Optional[float] = None
        self.last_logged_energy: Optional[float] = None
        self.audio_feedback_enabled = False
        self.audio_stream = None
        self.audio_phase = 0
        self.audio_frequency = 400
        self.audio_lock = threading.Lock()
        self.sample_rate = 44100

    def start(self, verbose_logging=True, image_logging=True, image_log_options=None, data_log_options=None):
        if not self.app_config.regions:
            logger.error("Cannot start monitoring: No regions loaded.")
            messagebox.showerror("Error", "Cannot start monitoring. Please load a valid configuration file.")
            return False
        
        self.verbose_logging_enabled = verbose_logging
        self.image_logging_enabled = image_logging
        self.image_log_options = image_log_options if image_log_options else ImageLogOptions()
        self.data_log_options = data_log_options if data_log_options else DataLogOptions()
        self.frame_count = 0 
        self.last_logged_ratio = None
        self.last_logged_energy = None
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"Screen monitoring thread started for USMA v.0.3.10")
        return True

    def stop(self):
        self.running = False
        self._stop_audio_feedback()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        logger.info("Screen monitoring stopped.")

    def update_config(self, new_config_path):
        self.config_path = new_config_path
        self.app_config = self._load_config(new_config_path)
        logger.info(f"Configuration updated to {new_config_path}")

    def set_audio_feedback(self, enabled: bool):
        self.audio_feedback_enabled = enabled
        if not enabled:
            self._stop_audio_feedback()

    def _load_config(self, path: str) -> AppConfig:
        try:
            with open(path, 'r') as f: config_data = json.load(f)
            config = AppConfig()
            metadata = config_data.get('_metadata', {})
            config.hsv_lower = metadata.get('hsv_lower', config.hsv_lower)
            config.hsv_upper = metadata.get('hsv_upper', config.hsv_upper)
            config.screenshot_interval = metadata.get('screenshot_interval', config.screenshot_interval)
            config.fft_cutoff_frequency = metadata.get('fft_cutoff_frequency', config.fft_cutoff_frequency)
            config.fft_energy_ratio_threshold = metadata.get('fft_energy_ratio_threshold', config.fft_energy_ratio_threshold)
            
            # Get all possible field names from the dataclass to load robustly
            region_fields = MonitoringRegion.__annotations__.keys()

            for name, data in config_data.items():
                if not name.startswith('_') and isinstance(data, dict):
                    # Filter data to only include keys that exist in the MonitoringRegion dataclass
                    filtered_data = {k: v for k, v in data.items() if k in region_fields}
                    if 'name' in filtered_data: # Ensure essential key is present
                        config.regions[name] = MonitoringRegion(**filtered_data)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return AppConfig()

    def _audio_callback(self, outdata, frames, time, status):
        try:
            if status: 
                logger.warning(f"Audio stream status: {status}")
            t = (self.audio_phase + np.arange(frames)) / self.sample_rate
            t = t.reshape(-1, 1)
            amplitude = np.iinfo(np.int16).max * 0.3
            outdata[:] = amplitude * np.sin(2 * np.pi * self.audio_frequency * t)
            self.audio_phase += frames
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            outdata.fill(0)  # Output silence on error

    def _start_audio_feedback(self):
        if not SOUND_DEVICE_AVAILABLE or self.audio_stream is not None: return
        try:
            self.audio_phase = 0
            self.audio_stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback, dtype='int16')
            self.audio_stream.start()
            logger.info("Continuous audio feedback started.")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.audio_stream = None

    def _stop_audio_feedback(self):
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop(); self.audio_stream.close()
                logger.info("Continuous audio feedback stopped.")
            except Exception as e: logger.error(f"Error stopping audio stream: {e}")
            finally: self.audio_stream = None
            
    def _monitoring_loop(self):
        while self.running:
            start_time = time.time()
            try:
                screenshot = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                results, overall_is_hf, avg_ratio, avg_hf_energy = self._process_frame(image, self.frame_count)
                
                if self.update_callback and overall_is_hf is not None:
                    self.update_callback(overall_is_hf, avg_ratio, avg_hf_energy)
                
                if self.audio_feedback_enabled:
                    with self.audio_lock:
                        if overall_is_hf and self.audio_stream is None:
                            self._start_audio_feedback()
                        elif not overall_is_hf and self.audio_stream is not None:
                            self._stop_audio_feedback()

                if self.frame_count % 10 == 0: self._log_results(results, self.frame_count)
                self.frame_count += 1
                
                elapsed_time = time.time() - start_time
                sleep_duration = self.app_config.screenshot_interval - elapsed_time
                if sleep_duration > 0: time.sleep(sleep_duration)
            except Exception as e: logger.error(f"Error in monitoring loop: {e}"); time.sleep(1)

    def _process_frame(self, image: np.ndarray, frame_count: int) -> (Dict, Optional[bool], Optional[float], Optional[float]):
        results = {'waves': {}}
        active_regions = {}
        for name, region in self.app_config.regions.items():
            if region.enabled and region.roi_type == 'wave':
                roi = image[region.y:region.y+region.height, region.x:region.x+region.width]
                analysis_result = self._analyze_wave_pattern(roi, name)
                if analysis_result:
                    results['waves'][name] = analysis_result
                    active_regions[name] = region

        if not results['waves']:
            return results, None, None, None

        classifications = [res.is_high_frequency for res in results['waves'].values()]
        overall_is_hf = sum(classifications) > len(classifications) / 2
        avg_energy_ratio = np.mean([res.energy_ratio for res in results['waves'].values()])
        avg_high_freq_energy = np.mean([res.high_freq_energy for res in results['waves'].values()])

        # --- LOG DE-CLUTTERING ---
        has_changed = (avg_energy_ratio is not None and 
                       (self.last_logged_ratio is None or 
                        not np.isclose(avg_energy_ratio, self.last_logged_ratio, atol=1e-9) or
                        not np.isclose(avg_high_freq_energy, self.last_logged_energy, atol=1e-9)))

        if has_changed:
            logger.info(f"Frame {frame_count}: New event detected. Logging enabled for this frame.")
            self.last_logged_ratio = avg_energy_ratio
            self.last_logged_energy = avg_high_freq_energy

            for name, result in results['waves'].items():
                region = active_regions[name]
                if self.image_logging_enabled:
                    self._create_visual_logs(result, region, frame_count)
                if self.data_log_options.log_mat:
                    self._save_mat_log(result, region, frame_count)
                if self.data_log_options.log_unv:
                    self._save_unv_log(result, region, frame_count)
        
        return results, overall_is_hf, avg_energy_ratio, avg_high_freq_energy

    def _validate_signal_quality(self, color_mask: np.ndarray) -> bool:
        height, width = color_mask.shape
        if height == 0 or width == 0:
            return False
        total_pixels = height * width
        if total_pixels == 0: return False
        signal_pixels = np.count_nonzero(color_mask)
        coverage_ratio = signal_pixels / total_pixels
        if not (0.0005 < coverage_ratio < 0.4):
            if self.verbose_logging_enabled: logger.debug(f"Skip: Signal coverage {coverage_ratio:.3e} out of range.")
            return False
        cols_with_signal = np.count_nonzero(np.sum(color_mask, axis=0) > 0)
        continuity_ratio = cols_with_signal / width
        if continuity_ratio < 0.15:
            if self.verbose_logging_enabled: logger.debug(f"Skip: Signal continuity {continuity_ratio:.3e} too low.")
            return False
        return True

    def _analyze_wave_pattern(self, roi: np.ndarray, region_name: str) -> Optional[WaveAnalysisResult]:
        if roi.size == 0: return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.app_config.hsv_lower), np.array(self.app_config.hsv_upper))
        if not self._validate_signal_quality(mask): return None

        y_coords, x_coords = np.nonzero(mask)
        width = mask.shape[1]
        if len(x_coords) == 0: signal_vector = np.full(width, roi.shape[0] / 2)
        else:
            unique_x, anchor_y = np.unique(x_coords, return_inverse=True)
            sum_y = np.bincount(anchor_y, weights=y_coords)
            count_y = np.bincount(anchor_y)
            anchor_y = sum_y / count_y
            if len(unique_x) < 2:
                signal_vector = np.full(width, anchor_y[0] if anchor_y.size > 0 else roi.shape[0] / 2)
            else: signal_vector = np.interp(np.arange(width), unique_x, anchor_y)

        signal_vector = roi.shape[0] - signal_vector
        if signal_vector.size < 2: return None
        
        N = len(signal_vector)
        detrended_signal = signal_vector - np.mean(signal_vector)
        yf, xf = rfft(detrended_signal), rfftfreq(N, 1) 
        fft_mags = np.abs(yf)

        total_energy = np.sum(fft_mags**2)
        high_freq_energy, energy_ratio, is_hf = 0, 0, False
        if total_energy > 1e-9:
            cutoff_indices = np.where(xf >= self.app_config.fft_cutoff_frequency)[0]
            if cutoff_indices.size > 0:
                high_freq_energy = np.sum(fft_mags[cutoff_indices[0]:]**2)
                energy_ratio = high_freq_energy / total_energy
            is_hf = energy_ratio > self.app_config.fft_energy_ratio_threshold
        return WaveAnalysisResult(is_hf, energy_ratio, high_freq_energy, signal_vector, xf, fft_mags, roi.copy(), mask.copy())

    def _create_visual_logs(self, result: WaveAnalysisResult, region: MonitoringRegion, frame_count: int):
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            base_filename = f"image_logs/{ts}_frame{frame_count}_{region.name}"

            if self.image_log_options.include_screenshot:
                cv2.imwrite(f"{base_filename}_01_ROI.jpg", result.roi_image)

            if self.image_log_options.include_color_filter:
                cv2.imwrite(f"{base_filename}_02_Mask.jpg", result.color_mask)

            if self.image_log_options.include_signal_plot:
                num_points = len(result.signal_vector)
                freq_axis = np.linspace(region.x_axis_min, region.x_axis_max, num_points)
                amp_axis = region.y_axis_min + (result.signal_vector / region.height) * (region.y_axis_max - region.y_axis_min)

                fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
                fig.patch.set_facecolor('#1E1E1E')
                ax.plot(freq_axis, amp_axis, color='cyan')
                ax.set_title(f'Reconstructed Signal - {region.name}', color='white')
                ax.set_xlabel('Frequency (Hz)', color='white')
                ax.set_ylabel(f'Amplitude ({region.y_axis_unit})', color='white')
                ax.set_facecolor('#2E2E2E')
                ax.tick_params(axis='both', colors='white')
                ax.grid(True, linestyle='--', alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{base_filename}_03_Signal.png", facecolor=fig.get_facecolor())
                plt.close(fig)

            if self.image_log_options.include_fft_plot:
                fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
                fig.patch.set_facecolor('#1E1E1E')
                ax.plot(result.fft_freqs, result.fft_mags, color='magenta')
                ax.set_title(f'FFT Magnitude Spectrum - {region.name}', color='white')
                ax.axvline(x=self.app_config.fft_cutoff_frequency, color='yellow', linestyle='--', linewidth=1, label=f'Cutoff: {self.app_config.fft_cutoff_frequency:.2f}')
                ax.set_xlim(left=0, right=0.5)
                ax.set_xlabel('Normalized Frequency', color='white')
                ax.set_ylabel('Magnitude (A.U.)', color='white')
                ax.set_facecolor('#2E2E2E')
                ax.tick_params(axis='both', colors='white')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(labelcolor='white')
                fig.tight_layout()
                fig.savefig(f"{base_filename}_04_FFT.png", facecolor=fig.get_facecolor())
                plt.close(fig)
                
        except Exception as e:
            logger.error(f"Failed to create visual logs for {region.name}: {e}")
    
    def _save_mat_log(self, result: WaveAnalysisResult, region: MonitoringRegion, frame_count: int):
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"signal_logs/{ts}_frame{frame_count}_{region.name}.mat"
            
            num_points = len(result.signal_vector)
            frequency_hz = np.linspace(region.x_axis_min, region.x_axis_max, num_points)
            amplitude_scaled = region.y_axis_min + (result.signal_vector / region.height) * (region.y_axis_max - region.y_axis_min)

            mat_data = {
                'frequency_hz': frequency_hz,
                'amplitude': amplitude_scaled,
                'amplitude_units': region.y_axis_unit,
                'info_region_name': region.name,
                'info_hf_ratio': result.energy_ratio,
                'raw_amplitude_pixels': result.signal_vector
            }
            scipy.io.savemat(filename, mat_data)
        except Exception as e:
            logger.error(f"Failed to save .mat file for {region.name}: {e}")

    def _save_unv_log(self, result: WaveAnalysisResult, region: MonitoringRegion, frame_count: int):
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"signal_logs/{ts}_frame{frame_count}_{region.name}.unv"
            num_points = len(result.signal_vector)
            if num_points < 2: return

            start_freq = region.x_axis_min
            freq_step = (region.x_axis_max - region.x_axis_min) / (num_points - 1)
            amplitude_scaled = region.y_axis_min + (result.signal_vector / region.height) * (region.y_axis_max - region.y_axis_min)

            with open(filename, 'w') as f:
                f.write("    -1\n")
                f.write("    58\n")
                
                # Line 6: Function ID. Type 4=FRF. C=Cartesian coord.
                f.write(f"         4         0         0         0 C{region.resp_node:>10}{region.resp_dof:>4} C{region.ref_node:>10}{region.ref_dof:>4}\n")
                
                # Line 7: Data Characteristics. Type=2 (Real), num_points, Spacing=1 (Even), start_x, step_x
                f.write(f"         2{num_points:>10}         1{start_freq:15.5E}{freq_step:15.5E}{0.0:15.5E}\n")

                # Lines 8-11: Labels
                f.write("        18         0         0         0 X-axis               Hz              \n")
                f.write(f"        12         0         0         0 Y-axis               {region.y_axis_unit:<16}\n")
                f.write("        13         0         0         0 Z-axis               NONE            \n")
                f.write("         0         0         0         0 NONE                 NONE            \n")

                # Data points: value and 0.0 for the imaginary part
                for val in amplitude_scaled:
                    f.write(f"  {val:13.6E}  {0.0:13.6E}\n")
                
                f.write("    -1\n")
        except Exception as e:
            logger.error(f"Failed to save .unv file for {region.name}: {e}")

    def _log_results(self, results: Dict, frame_count: int):
        if not self.verbose_logging_enabled: return
        for name, res in results.get('waves', {}).items():
            logger.debug(f"Frame {frame_count}: {name} -> {'HF' if res.is_high_frequency else 'LF'} (HF Ratio: {res.energy_ratio:.3e})")

# --- 4. VISUALIZATION: REGION OVERLAY ---
class RegionOverlay(tk.Toplevel):
    def __init__(self, parent, config_path):
        super().__init__(parent)
        self.config_path = config_path
        self.attributes("-transparentcolor", "white", "-topmost", True); self.overrideredirect(True)
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self._draw_overlay()
    def _draw_overlay(self):
        try:
            with open(self.config_path, 'r') as f: data = json.load(f)
        except Exception as e: logger.error(f"Overlay Error: {e}"); self.destroy(); return
        canvas = tk.Canvas(self, bg="white", highlightthickness=0); canvas.pack(fill=tk.BOTH, expand=True)
        colors = {"wave": "#3498db", "status": "#2ecc71", "text": "#f39c12"}
        for name, region_data in data.items():
            if not name.startswith('_') and region_data.get('enabled', True):
                x,y,w,h = region_data['x'], region_data['y'], region_data['width'], region_data['height']
                color = colors.get(region_data.get('roi_type', 'wave'), "#95a5a6")
                canvas.create_rectangle(x-5, y-5, x+w+5, y+h+5, outline=color, width=2)
                canvas.create_text(x-5, y-5, text=name, anchor="sw", font=("Arial", 10, "bold"), fill=color)
        canvas.create_text(self.winfo_screenwidth()-10, self.winfo_screenheight()-10, text=f"Config: {os.path.basename(self.config_path)}", anchor="se", fill="#333")

# --- 5. INTERACTIVE HSV THRESHOLDER ---
class HSVThresholderWindow(tk.Toplevel):
    def __init__(self, parent, screenshot_cv2, regions, current_config, callback):
        super().__init__(parent); self.title("Interactive HSV Thresholder"); self.geometry("1000x700")
        self.full_screenshot, self.regions, self.callback = screenshot_cv2, regions, callback
        self.active_roi_cv2, self.hsv_image, self.photo = None, None, None
        top_frame = ttk.Frame(self, padding=5); top_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top_frame, text="Select Region:").pack(side=tk.LEFT, padx=5)
        self.region_selector = ttk.Combobox(top_frame, state="readonly", values=[r.name for r in regions.values() if r.roi_type == 'wave'])
        self.region_selector.pack(side=tk.LEFT, padx=5); self.region_selector.bind("<<ComboboxSelected>>", self._on_region_select)
        self.img_label = ttk.Label(self); self.img_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        controls_frame = ttk.Frame(self, padding=10); controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.hsv_vars = {k: tk.IntVar() for k in ['HMin','SMin','VMin','HMax','SMax','VMax']}
        def create_slider(parent, text, var, from_, to):
            frame = ttk.Frame(parent)
            tk.Scale(frame, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL, length=200, command=self._update_mask, showvalue=0).pack(side=tk.LEFT, expand=True, fill=tk.X)
            ttk.Label(frame, textvariable=var, width=4).pack(side=tk.LEFT); frame.pack(fill=tk.X, pady=2); ttk.Label(parent, text=text).pack(anchor='w')
        create_slider(controls_frame, "Hue Min", self.hsv_vars['HMin'], 0, 179); create_slider(controls_frame, "Hue Max", self.hsv_vars['HMax'], 0, 179)
        create_slider(controls_frame, "Sat Min", self.hsv_vars['SMin'], 0, 255); create_slider(controls_frame, "Sat Max", self.hsv_vars['SMax'], 0, 255)
        create_slider(controls_frame, "Val Min", self.hsv_vars['VMin'], 0, 255); create_slider(controls_frame, "Val Max", self.hsv_vars['VMax'], 0, 255)
        self.hsv_vars['HMin'].set(current_config.hsv_lower[0]); self.hsv_vars['SMin'].set(current_config.hsv_lower[1]); self.hsv_vars['VMin'].set(current_config.hsv_lower[2])
        self.hsv_vars['HMax'].set(current_config.hsv_upper[0]); self.hsv_vars['SMax'].set(current_config.hsv_upper[1]); self.hsv_vars['VMax'].set(current_config.hsv_upper[2])
        ttk.Button(controls_frame, text="Apply & Close", command=self._apply_and_close).pack(pady=20, side=tk.BOTTOM)
        self.grab_set(); self.lift()
        if self.region_selector['values']: self.region_selector.current(0); self._on_region_select(None)
    def _on_region_select(self, _=None):
        region = self.regions.get(self.region_selector.get());
        if not region: return
        self.active_roi_cv2 = self.full_screenshot[region.y:region.y+region.height, region.x:region.x+region.width]
        self.hsv_image = cv2.cvtColor(self.active_roi_cv2, cv2.COLOR_BGR2HSV); self._update_mask()
    def _update_mask(self, _=None):
        if self.hsv_image is None: return
        lower = np.array([v.get() for v in (self.hsv_vars['HMin'],self.hsv_vars['SMin'],self.hsv_vars['VMin'])])
        upper = np.array([v.get() for v in (self.hsv_vars['HMax'],self.hsv_vars['SMax'],self.hsv_vars['VMax'])])
        mask = cv2.inRange(self.hsv_image, lower, upper)
        output = cv2.bitwise_and(self.active_roi_cv2, self.active_roi_cv2, mask=mask)
        img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        w, h = img_pil.size; max_w, max_h = self.img_label.winfo_width(), self.img_label.winfo_height()
        if max_w < 50 or max_h < 50: max_w, max_h = 700, 600
        if w > max_w or h > max_h: scale = min(max_w/w, max_h/h); img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img_pil); self.img_label.config(image=self.photo)
    def _apply_and_close(self): self.callback({k: v.get() for k, v in self.hsv_vars.items()}); self.destroy()

# --- 6. CONFIGURATION TOOL ---
class ConfigToolWindow(tk.Toplevel):
    def __init__(self, parent, main_root):
        super().__init__(parent); self.title("Advanced Region & Color Configuration Tool")
        self.main_root = main_root
        self.app_config = AppConfig()
        self.screenshot = None
        self.photo = None
        self.scale = 1.0
        self.drawing, self.start_x, self.start_y, self.selected_region_name = False, 0, 0, None
        self.resize_timer = None
        self.x_offset, self.y_offset = 0, 0
        self.state('zoomed') 
        self._setup_gui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(200, self._take_screenshot) 

    def _on_closing(self):
        self.main_root.deiconify()
        self.destroy()

    def _setup_gui(self):
        toolbar = ttk.Frame(self, padding=5); toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load Config", command=self._load_config).pack(side=tk.LEFT, padx=2)
        
        main_frame = ttk.Frame(self, padding=5); main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Screenshot Preview"); canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, bg="black"); self.canvas.pack(fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(main_frame, width=450); right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5); right_panel.pack_propagate(False)
        
        # --- Right Panel Content ---
        ttk.Button(ttk.LabelFrame(right_panel, text="Capture"), text="Take Screenshot", command=self._take_screenshot).pack(pady=5, padx=5, fill=tk.X)
        
        list_frame = ttk.LabelFrame(right_panel, text="Defined Regions"); list_frame.pack(fill=tk.X, pady=5)
        self.region_listbox = tk.Listbox(list_frame, height=6); self.region_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.region_listbox.yview); list_scroll.pack(side=tk.RIGHT, fill=tk.Y); self.region_listbox.config(yscrollcommand=list_scroll.set)
        
        # --- Editor Frame ---
        editor_frame = ttk.LabelFrame(right_panel, text="Region Editor"); editor_frame.pack(fill=tk.X, pady=5)
        
        self.editor_vars = {
            'name': tk.StringVar(),'x': tk.IntVar(),'y': tk.IntVar(),'width': tk.IntVar(),'height': tk.IntVar(),
            'roi_type': tk.StringVar(),'enabled': tk.BooleanVar(), 'x_axis_min': tk.DoubleVar(), 'x_axis_max': tk.DoubleVar(),
            'y_axis_min': tk.DoubleVar(), 'y_axis_max': tk.DoubleVar(), 'y_axis_unit': tk.StringVar(),
            'resp_node': tk.IntVar(), 'resp_dof': tk.IntVar(), 'ref_node': tk.IntVar(), 'ref_dof': tk.IntVar()
        }

        # Basic properties
        f1 = ttk.Frame(editor_frame); f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="Name:", width=12).pack(side=tk.LEFT, padx=5); ttk.Entry(f1, textvariable=self.editor_vars['name']).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(f1, text="Type:").pack(side=tk.LEFT, padx=5); ttk.Combobox(f1, textvariable=self.editor_vars['roi_type'], values=['wave', 'status', 'text'], state='readonly', width=8).pack(side=tk.LEFT, padx=5)
        
        # Physical Scaling
        f_scale = ttk.LabelFrame(editor_frame, text="Physical Axis Scaling"); f_scale.pack(fill=tk.X, pady=5, padx=5)
        g = ttk.Frame(f_scale); g.pack(fill=tk.X); ttk.Label(g, text="X-Axis Min (Hz):").grid(row=0, column=0, sticky=tk.W); ttk.Entry(g, textvariable=self.editor_vars['x_axis_min'], width=10).grid(row=0, column=1, padx=5); ttk.Label(g, text="X-Axis Max (Hz):").grid(row=0, column=2, sticky=tk.W, padx=5); ttk.Entry(g, textvariable=self.editor_vars['x_axis_max'], width=10).grid(row=0, column=3, padx=5)
        g2 = ttk.Frame(f_scale); g2.pack(fill=tk.X); ttk.Label(g2, text="Y-Axis Min:").grid(row=0, column=0, sticky=tk.W); ttk.Entry(g2, textvariable=self.editor_vars['y_axis_min'], width=10).grid(row=0, column=1, padx=5); ttk.Label(g2, text="Y-Axis Max:").grid(row=0, column=2, sticky=tk.W, padx=5); ttk.Entry(g2, textvariable=self.editor_vars['y_axis_max'], width=10).grid(row=0, column=3, padx=5)
        
        # UNV Header Metadata
        f_unv = ttk.LabelFrame(editor_frame, text="UNV/.mat Metadata"); f_unv.pack(fill=tk.X, pady=5, padx=5)
        g3 = ttk.Frame(f_unv); g3.pack(fill=tk.X); ttk.Label(g3, text="Y-Axis Unit:").grid(row=0, column=0); ttk.Entry(g3, textvariable=self.editor_vars['y_axis_unit'], width=10).grid(row=0, column=1, padx=5);
        g4 = ttk.Frame(f_unv); g4.pack(fill=tk.X); ttk.Label(g4, text="Resp Node/DOF:").grid(row=0, column=0); ttk.Entry(g4, textvariable=self.editor_vars['resp_node'], width=6).grid(row=0, column=1); ttk.Entry(g4, textvariable=self.editor_vars['resp_dof'], width=6).grid(row=0, column=2, padx=5); ttk.Label(g4, text="Ref Node/DOF:").grid(row=0, column=3); ttk.Entry(g4, textvariable=self.editor_vars['ref_node'], width=6).grid(row=0, column=4); ttk.Entry(g4, textvariable=self.editor_vars['ref_dof'], width=6).grid(row=0, column=5, padx=5)

        # Buttons and Checkbox
        f_buttons = ttk.Frame(editor_frame); f_buttons.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(f_buttons, text="Enabled", variable=self.editor_vars['enabled']).pack(side=tk.LEFT, padx=10)
        ttk.Button(f_buttons, text="Update Region", command=self._update_region_from_editor).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(f_buttons, text="Delete Region", command=self._delete_selected_region).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # Bottom general params
        params_frame = ttk.LabelFrame(right_panel, text="Global FFT Analysis Parameters"); params_frame.pack(fill=tk.X, pady=5)
        self.param_vars = {'fft_cutoff_frequency': tk.DoubleVar(value=self.app_config.fft_cutoff_frequency), 'fft_energy_ratio_threshold': tk.DoubleVar(value=self.app_config.fft_energy_ratio_threshold)}
        g_fft = ttk.Frame(params_frame); g_fft.pack(fill=tk.X, pady=2)
        ttk.Label(g_fft, text="Cutoff Freq:").grid(row=0, column=0); ttk.Spinbox(g_fft, from_=0.0, to=0.5, increment=0.01, textvariable=self.param_vars['fft_cutoff_frequency'], width=8).grid(row=0, column=1, padx=5)
        ttk.Label(g_fft, text="Energy Ratio:").grid(row=0, column=2); ttk.Spinbox(g_fft, from_=0.0, to=1.0, increment=0.001, textvariable=self.param_vars['fft_energy_ratio_threshold'], width=8).grid(row=0, column=3, padx=5)
        ttk.Button(params_frame, text="Apply Global Parameters", command=self._apply_params).pack(fill=tk.X, pady=5)
        ttk.Button(ttk.LabelFrame(right_panel, text="Signal Color Calibration"), text="Calibrate Signal Color...", command=self._launch_hsv_thresholder).pack(pady=5, fill=tk.X)

        self.canvas.bind("<Button-1>", self._on_canvas_click); self.canvas.bind("<B1-Motion>", self._update_selection); self.canvas.bind("<ButtonRelease-1>", self._end_selection); self.region_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
    
    def _on_canvas_resize(self, event):
        if self.resize_timer: self.after_cancel(self.resize_timer)
        self.resize_timer = self.after(150, self._redraw_canvas_content)

    def _take_screenshot(self):
        self.withdraw()
        self.main_root.iconify()
        time.sleep(0.5)
        self.screenshot = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        self.deiconify()
        self.lift()
        self.focus_force()
        self._redraw_canvas_content()

    def _redraw_canvas_content(self):
        if self.screenshot is None: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return
        self.canvas.delete("all")
        img_h, img_w = self.screenshot.shape[:2]
        self.scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * self.scale), int(img_h * self.scale)
        img_resized = Image.fromarray(cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2RGB)).resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img_resized)
        self.x_offset, self.y_offset = (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2
        self.canvas.create_image(self.x_offset, self.y_offset, image=self.photo, anchor=tk.NW, tags="screenshot")
        self._redraw_regions_on_canvas()

    def _on_canvas_click(self, event): self.drawing, self.start_x, self.start_y = True, event.x, event.y; self.canvas.delete("selection_rect")
    def _update_selection(self, event):
        if self.drawing: self.canvas.delete("selection_rect"); self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", width=2, tags="selection_rect")
    def _end_selection(self, event):
        if not self.drawing: return
        self.drawing = False; 
        
        # Create a default region first
        name = f"region_{len(self.app_config.regions)+1}"
        x1_canvas, y1_canvas = min(self.start_x, event.x), min(self.start_y, event.y)
        x2_canvas, y2_canvas = max(self.start_x, event.x), max(self.start_y, event.y)
        x1, y1 = int((x1_canvas - self.x_offset) / self.scale), int((y1_canvas - self.y_offset) / self.scale)
        x2, y2 = int((x2_canvas - self.x_offset) / self.scale), int((y2_canvas - self.y_offset) / self.scale)
        
        new_region = MonitoringRegion(name=name, x=x1, y=y1, width=x2-x1, height=y2-y1, roi_type='wave')
        self.app_config.regions[name] = new_region
        
        self.canvas.delete("selection_rect"); self._update_ui_from_data()
        
        # Auto-select the newly created region
        self.region_listbox.selection_clear(0, tk.END)
        new_idx = sorted(self.app_config.regions.keys()).index(name)
        self.region_listbox.selection_set(new_idx)
        self.region_listbox.activate(new_idx)
        self._on_listbox_select(None)


    def _launch_hsv_thresholder(self):
        if self.screenshot is None: return messagebox.showwarning("Warning", "Please take a screenshot first.", parent=self)
        wave_regions = {n: r for n, r in self.app_config.regions.items() if r.roi_type == 'wave'}
        if not wave_regions: return messagebox.showwarning("Warning", "Define at least one 'wave' region first.", parent=self)
        HSVThresholderWindow(self, self.screenshot, wave_regions, self.app_config, self._apply_new_hsv)
    def _apply_new_hsv(self, hsv):
        self.app_config.hsv_lower=[hsv['HMin'],hsv['SMin'],hsv['VMin']]; self.app_config.hsv_upper=[hsv['HMax'],hsv['SMax'],hsv['VMax']]
    def _apply_params(self): self.app_config.fft_cutoff_frequency=self.param_vars['fft_cutoff_frequency'].get(); self.app_config.fft_energy_ratio_threshold=self.param_vars['fft_energy_ratio_threshold'].get(); messagebox.showinfo("Success", "Analysis parameters updated.", parent=self)
    def _update_ui_from_data(self):
        sel_name = self.selected_region_name; self.region_listbox.delete(0, tk.END)
        for i, name in enumerate(sorted(self.app_config.regions.keys())):
            disp = f"{name}" if self.app_config.regions[name].enabled else f"{name} (Disabled)"
            self.region_listbox.insert(tk.END, disp)
            if name == sel_name: self.region_listbox.selection_set(i)
        self.param_vars['fft_cutoff_frequency'].set(self.app_config.fft_cutoff_frequency); self.param_vars['fft_energy_ratio_threshold'].set(self.app_config.fft_energy_ratio_threshold); self._redraw_regions_on_canvas()
    def _redraw_regions_on_canvas(self):
        self.canvas.delete("region"); colors = {"wave":"lime","status":"cyan","text":"yellow"}
        if not hasattr(self, 'x_offset'): return 
        for name, r in self.app_config.regions.items():
            x1, y1 = r.x * self.scale + self.x_offset, r.y * self.scale + self.y_offset
            x2, y2 = (r.x + r.width) * self.scale + self.x_offset, (r.y + r.height) * self.scale + self.y_offset
            color = colors.get(r.roi_type,"white") if r.enabled else "gray"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=("region", name)); self.canvas.create_text(x1+5, y1+5, text=name, fill=color, anchor="nw", tags=("region", name))
    def _on_listbox_select(self, _):
        if not self.region_listbox.curselection(): 
            self.selected_region_name = None
            return
        self.selected_region_name = self.region_listbox.get(self.region_listbox.curselection()).replace(" (Disabled)", "")
        region_data = self.app_config.regions[self.selected_region_name]
        for key, var in self.editor_vars.items(): 
            if hasattr(region_data, key):
                var.set(getattr(region_data, key))

    def _update_region_from_editor(self):
        if not self.selected_region_name: return messagebox.showerror("Error", "No region selected.", parent=self)
        old_name, new_name = self.selected_region_name, self.editor_vars['name'].get()
        if new_name != old_name and new_name in self.app_config.regions: return messagebox.showerror("Error", "Region name must be unique.", parent=self)
        
        try:
            # Create a dictionary of the new values from the GUI variables
            new_data = {k: v.get() for k, v in self.editor_vars.items()}
            # The pixel dimensions are read-only and should not be updated from here
            current_region = self.app_config.regions[old_name]
            new_data['x'] = current_region.x
            new_data['y'] = current_region.y
            new_data['width'] = current_region.width
            new_data['height'] = current_region.height
            
            updated_region = MonitoringRegion(**new_data)

            del self.app_config.regions[old_name]
            self.app_config.regions[new_name] = updated_region
            self.selected_region_name = new_name
            self._update_ui_from_data()
        except tk.TclError as e:
            messagebox.showerror("Input Error", f"Invalid input value: {e}", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Could not update region: {e}", parent=self)


    def _delete_selected_region(self):
        if not self.selected_region_name: return messagebox.showerror("Error", "No region selected.", parent=self)
        if messagebox.askyesno("Confirm Delete", f"Delete '{self.selected_region_name}'?", parent=self):
            del self.app_config.regions[self.selected_region_name]
            self.selected_region_name = None
            # Clear editor fields
            for key, var in self.editor_vars.items():
                if isinstance(var, (tk.IntVar, tk.DoubleVar)): var.set(0)
                elif isinstance(var, tk.BooleanVar): var.set(False)
                else: var.set("")
            self._update_ui_from_data()
            
    def _save_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialdir="configs", parent=self)
        if not path: return
        try:
            self._apply_params()
            data = {n: asdict(r) for n, r in self.app_config.regions.items()}
            data['_metadata'] = {
                'hsv_lower': self.app_config.hsv_lower,
                'hsv_upper': self.app_config.hsv_upper,
                'screenshot_interval': self.app_config.screenshot_interval,
                'fft_cutoff_frequency': self.app_config.fft_cutoff_frequency,
                'fft_energy_ratio_threshold': self.app_config.fft_energy_ratio_threshold
            }
            with open(path, 'w') as f: json.dump(data, f, indent=2)
            messagebox.showinfo("Success", f"Saved to {os.path.basename(path)}", parent=self)
        except Exception as e: messagebox.showerror("Error", f"Failed to save: {e}", parent=self)
    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], initialdir="configs", parent=self)
        if not path: return
        try:
            self.app_config = ScreenMonitor(path).app_config; self._update_ui_from_data()
            if self.screenshot: self._redraw_regions_on_canvas()
            messagebox.showinfo("Success", f"Loaded {os.path.basename(path)}", parent=self)
        except Exception as e: messagebox.showerror("Error", f"Failed to load: {e}", parent=self)

# --- 7. MAIN GUI: THE CENTRAL CONTROL APPLICATION ---
class MonitorControlGUI:
    def __init__(self, root):
        self.root = root; self.root.title("USMA v.0.3.10"); self.root.geometry("850x450")
        self.config_path = tk.StringVar(value="configs/default_config.json")
        self.is_monitoring, self.is_overlay_on = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.verbose_logging_on = tk.BooleanVar(value=True)
        self.image_logging_on = tk.BooleanVar(value=False)
        self.log_opt_screenshot = tk.BooleanVar(value=False)
        self.log_opt_color_filter = tk.BooleanVar(value=False)
        self.log_opt_signal_plot = tk.BooleanVar(value=False)
        self.log_opt_fft_plot = tk.BooleanVar(value=False)
        self.audio_feedback_on = tk.BooleanVar(value=False)
        self.log_to_mat = tk.BooleanVar(value=False)
        self.log_to_unv = tk.BooleanVar(value=False)

        self.monitor = ScreenMonitor(self.config_path.get(), self.update_feedback_panel)
        initial_freq = 1.0/self.monitor.app_config.screenshot_interval if self.monitor.app_config.screenshot_interval>0 else 4.0
        self.sample_frequency = tk.DoubleVar(value=round(initial_freq, 2))
        
        self.overlay = None
        self._setup_main_gui(); self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_main_gui(self):
        frame = ttk.Frame(self.root, padding=10); frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(frame, text="Configuration"); config_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(config_frame, textvariable=self.config_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.load_button = ttk.Button(config_frame, text="Load...", command=self._load_config); self.load_button.pack(side=tk.LEFT, padx=5)
        self.edit_button = ttk.Button(config_frame, text="Edit Config...", command=self._launch_config_tool); self.edit_button.pack(side=tk.LEFT, padx=5)
        
        feedback_frame = ttk.LabelFrame(frame, text="Live Analysis Feedback"); feedback_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.status_light = tk.Canvas(feedback_frame, width=30, height=30, bg="gray", highlightthickness=0); self.status_light.grid(row=0, column=0, rowspan=2, padx=15, pady=5)
        self.class_var = tk.StringVar(value="Overall: --"); self.hf_ratio_var = tk.StringVar(value="Avg HF Ratio: --"); self.hf_energy_var = tk.StringVar(value="Avg HF Energy (pixels^2): --")
        ttk.Label(feedback_frame, textvariable=self.class_var, font=("Segoe UI", 14)).grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(feedback_frame, textvariable=self.hf_ratio_var, font=("Segoe UI", 10)).grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(feedback_frame, textvariable=self.hf_energy_var, font=("Segoe UI", 10)).grid(row=1, column=2, sticky=tk.W, padx=10)
        feedback_frame.columnconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(frame, text="Controls"); control_frame.pack(fill=tk.X, pady=5)
        self.start_stop_button = ttk.Button(control_frame, text="Start Monitoring", command=self._toggle_monitoring); self.start_stop_button.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)
        self.overlay_check = ttk.Checkbutton(control_frame, text="Show Overlay", variable=self.is_overlay_on, command=self._toggle_overlay); self.overlay_check.pack(side=tk.LEFT, padx=10, pady=10)
        
        params_frame = ttk.LabelFrame(control_frame, text="Parameters"); params_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        freq_frame = ttk.Frame(params_frame); ttk.Label(freq_frame, text="Sample Freq (Hz):").pack(side=tk.LEFT, padx=(5,2)); self.freq_spinbox = ttk.Spinbox(freq_frame, from_=0.1, to=30.0, increment=0.1, textvariable=self.sample_frequency, width=6); self.freq_spinbox.pack(side=tk.LEFT, padx=(0,5)); freq_frame.pack(pady=5)
        self.audio_check = ttk.Checkbutton(params_frame, text="Audio Feedback", variable=self.audio_feedback_on, command=self._toggle_audio_feedback); self.audio_check.pack(anchor=tk.W, padx=5, pady=(0, 5))
        if not SOUND_DEVICE_AVAILABLE: self.audio_check.config(state=tk.DISABLED); self.audio_feedback_on.set(False)
        
        logging_controls_frame = ttk.Frame(control_frame); logging_controls_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        logging_main_frame = ttk.LabelFrame(logging_controls_frame, text="General & Image Logging"); logging_main_frame.pack(fill=tk.X, side=tk.LEFT, anchor=tk.N)
        self.verbose_check = ttk.Checkbutton(logging_main_frame, text="Verbose Console Log", variable=self.verbose_logging_on); self.verbose_check.pack(anchor=tk.W, padx=5, pady=2)
        self.img_log_check = ttk.Checkbutton(logging_main_frame, text="Enable Image Logs", variable=self.image_logging_on, command=self._toggle_img_log_options_state); self.img_log_check.pack(anchor=tk.W, padx=5, pady=2)
        
        self.img_log_options_frame = ttk.Frame(logging_main_frame); self.img_log_options_frame.pack(fill=tk.X, pady=(5,0))
        for txt, var in [("ROI Screenshot",self.log_opt_screenshot), ("Color Filter Mask",self.log_opt_color_filter), ("Signal Plot",self.log_opt_signal_plot), ("FFT Plot",self.log_opt_fft_plot)]: 
            ttk.Checkbutton(self.img_log_options_frame, text=txt, variable=var).pack(anchor=tk.W, padx=15)
        
        data_logging_frame = ttk.LabelFrame(logging_controls_frame, text="Signal Data Logging"); data_logging_frame.pack(fill=tk.X, side=tk.LEFT, padx=10, anchor=tk.N)
        ttk.Checkbutton(data_logging_frame, text="Log to .mat file", variable=self.log_to_mat).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(data_logging_frame, text="Log to .unv file", variable=self.log_to_unv).pack(anchor=tk.W, padx=5, pady=2)

        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W); self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        self._toggle_img_log_options_state()

    def _toggle_audio_feedback(self):
        self.monitor.set_audio_feedback(self.audio_feedback_on.get())
    
    def _toggle_img_log_options_state(self):
        state = tk.NORMAL if self.image_logging_on.get() else tk.DISABLED
        for child in self.img_log_options_frame.winfo_children(): child.configure(state=state)

    def update_feedback_panel(self, hf, ratio, energy): self.root.after(0, self._update_feedback_ui, hf, ratio, energy)
    def _update_feedback_ui(self, is_hf, ratio, energy):
        self.class_var.set(f"Overall: {'HF' if is_hf else 'LF'}"); self.status_light.config(bg="red" if is_hf else "green")
        self.hf_ratio_var.set(f"Avg HF Ratio: {ratio:.3e}"); self.hf_energy_var.set(f"Avg HF Energy (pixels^2): {energy:.3e}")
    def _reset_feedback_ui(self):
        self.class_var.set("Overall: --"); self.status_light.config(bg="gray"); self.hf_ratio_var.set("Avg HF Ratio: --"); self.hf_energy_var.set("Avg HF Energy (pixels^2): --")
    
    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], initialdir="configs", title="Select Config")
        if not path: return
        self.config_path.set(path)
        self.monitor.update_config(path) 
        try:
            interval = self.monitor.app_config.screenshot_interval
            self.sample_frequency.set(round(1.0 / interval, 2))
        except (ZeroDivisionError, TypeError, AttributeError) as e:
            logger.warning(f"Could not calculate sample frequency from config: {e}")
            self.sample_frequency.set(4.0)
        if self.is_overlay_on.get(): self._toggle_overlay(); self._toggle_overlay()
        self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
    
    def _launch_config_tool(self):
        self.root.iconify()
        config_window = ConfigToolWindow(self.root, self.root)
        config_window.grab_set()

    def _toggle_monitoring(self):
        all_controls = [self.load_button, self.edit_button, self.verbose_check, 
                    self.img_log_check, self.overlay_check, self.freq_spinbox, 
                    self.audio_check]
        # Gather all logging checkboxes dynamically
        for child in self.img_log_options_frame.winfo_children(): all_controls.append(child)
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                     if 'data_logging_frame' in str(subchild):
                         for log_check in subchild.winfo_children():
                             all_controls.append(log_check)


        if self.is_monitoring.get():
            self.monitor.stop(); self.is_monitoring.set(False); self.start_stop_button.config(text="Start Monitoring")
            self.status_label.config(text="Stopped."); self._reset_feedback_ui()
            for w in all_controls:
                if w == self.audio_check and not SOUND_DEVICE_AVAILABLE: continue
                try: w.config(state=tk.NORMAL)
                except tk.TclError: pass
            self._toggle_img_log_options_state()
        else:
            if not os.path.exists(self.config_path.get()):
                return messagebox.showerror("Error", "Config file not found.")
            try:
                freq = self.sample_frequency.get()
                if freq <= 0: return messagebox.showerror("Error", "Sample frequency must be positive.")
            except tk.TclError: return messagebox.showerror("Error", "Invalid sample frequency.")
            
            self.monitor.app_config.screenshot_interval = 1.0 / self.sample_frequency.get()
            self.monitor.set_audio_feedback(self.audio_feedback_on.get())
            
            img_log_opts = ImageLogOptions(self.log_opt_screenshot.get(),self.log_opt_color_filter.get(),self.log_opt_signal_plot.get(),self.log_opt_fft_plot.get())
            data_log_opts = DataLogOptions(self.log_to_mat.get(), self.log_to_unv.get())
            
            if self.monitor.start(self.verbose_logging_on.get(), self.image_logging_on.get(), img_log_opts, data_log_opts):
                self.is_monitoring.set(True); self.start_stop_button.config(text="Stop Monitoring"); self.status_label.config(text="Monitoring active...")
                for w in all_controls: 
                    try: w.config(state=tk.DISABLED)
                    except tk.TclError: pass
                self._toggle_img_log_options_state()

    def _toggle_overlay(self):
        if self.overlay and self.overlay.winfo_exists(): 
            self.overlay.destroy(); self.overlay = None
        if self.is_overlay_on.get():
            path = self.config_path.get()
            if not os.path.exists(path):
                messagebox.showerror("Error", "Config file not found."); self.is_overlay_on.set(False); return
            try:
                with open(path, 'r') as f: json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid config file format."); self.is_overlay_on.set(False); return
            self.overlay = RegionOverlay(self.root, path)
            
    def _on_closing(self):
        if self.is_monitoring.get(): self.monitor.stop()
        if self.overlay and self.overlay.winfo_exists(): self.overlay.destroy()
        self.root.destroy()

# --- 8. APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = MonitorControlGUI(main_root)
    main_root.mainloop()

