#!/usr/bin/env python3
"""
USMA (Unified Screen Monitoring Application) - v.0.3.6

A single, GUI-driven application that combines a professional-grade region 
configuration tool, real-time screen monitoring, visual overlay, and clear 
image logging.

v.0.3.6 Changes:
- Audio feedback is now a continuous tone that plays while an HF signal is
  detected, providing clearer real-time status.
- Default sample frequency changed to 4 Hz for more responsive monitoring.
- All image logging options are now disabled by default to reduce disk usage
  unless explicitly enabled by the user.
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
    for folder in ['logs', 'configs', 'image_logs']:
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
class MonitoringRegion:
    """Defines a region of interest (ROI) on the screen."""
    name: str
    x: int
    y: int
    width: int
    height: int
    roi_type: str
    enabled: bool = field(default=True)

@dataclass
class WaveAnalysisResult:
    """Stores the complete results of a wave pattern analysis."""
    is_high_frequency: bool
    energy_ratio: float
    high_freq_energy: float
    signal_vector: np.ndarray
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
    hsv_lower2: List[int] = field(default_factory=lambda: [0, 0, 0])
    hsv_upper2: List[int] = field(default_factory=lambda: [179, 255, 240])
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
        self.audio_feedback_enabled = False
        self.audio_stream = None
        self.audio_phase = 0
        self.audio_frequency = 400
        self.sample_rate = 44100

    def start(self, verbose_logging=True, image_logging=True, image_log_options=None):
        if not self.app_config.regions:
            logger.error("Cannot start monitoring: No regions loaded.")
            messagebox.showerror("Error", "Cannot start monitoring. Please load a valid configuration file.")
            return False
        
        self.verbose_logging_enabled = verbose_logging
        self.image_logging_enabled = image_logging
        self.image_log_options = image_log_options if image_log_options else ImageLogOptions()
        self.frame_count = 0 
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"Screen monitoring thread started for USMA v.0.3.6")
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
            config.hsv_lower2 = metadata.get('hsv_lower2', config.hsv_lower2)
            config.hsv_upper2 = metadata.get('hsv_upper2', config.hsv_upper2)
            config.screenshot_interval = metadata.get('screenshot_interval', config.screenshot_interval)
            config.fft_cutoff_frequency = metadata.get('fft_cutoff_frequency', config.fft_cutoff_frequency)
            config.fft_energy_ratio_threshold = metadata.get('fft_energy_ratio_threshold', config.fft_energy_ratio_threshold)
            for name, data in config_data.items():
                if not name.startswith('_') and isinstance(data, dict):
                    if all(key in data for key in ['name', 'x', 'y', 'width', 'height', 'roi_type']):
                        config.regions[name] = MonitoringRegion(**data)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return AppConfig()

    def _audio_callback(self, outdata, frames, time, status):
        if status: logger.warning(f"Audio stream status: {status}")
        t = (self.audio_phase + np.arange(frames)) / self.sample_rate
        t = t.reshape(-1, 1)
        amplitude = np.iinfo(np.int16).max * 0.3
        outdata[:] = amplitude * np.sin(2 * np.pi * self.audio_frequency * t)
        self.audio_phase += frames

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
        for name, region in self.app_config.regions.items():
            if region.enabled and region.roi_type == 'wave':
                roi = image[region.y:region.y+region.height, region.x:region.x+region.width]
                analysis_result = self._analyze_wave_pattern(roi, name)
                if analysis_result:
                    results['waves'][name] = analysis_result
                    if self.image_logging_enabled and any(asdict(self.image_log_options).values()):
                        self._create_visual_log(analysis_result, name, frame_count)
        
        overall_is_hf, avg_energy_ratio, avg_high_freq_energy = None, None, None
        if results['waves']:
            classifications = [res.is_high_frequency for res in results['waves'].values()]
            overall_is_hf = sum(classifications) > len(classifications) / 2
            avg_energy_ratio = np.mean([res.energy_ratio for res in results['waves'].values()])
            avg_high_freq_energy = np.mean([res.high_freq_energy for res in results['waves'].values()])
            if self.image_logging_enabled:
                self._create_summary_log(results['waves'], overall_is_hf, frame_count)
        return results, overall_is_hf, avg_energy_ratio, avg_high_freq_energy

    def _validate_signal_quality(self, color_mask: np.ndarray) -> bool:
        height, width = color_mask.shape
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

    def _create_visual_log(self, result: WaveAnalysisResult, region_name: str, frame_count: int):
        plot_path = f"image_logs/temp_plot_{frame_count}_{region_name}.png"
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
            fig.patch.set_facecolor('#1E1E1E')
            
            if self.image_log_options.include_signal_plot:
                axes[0].plot(result.signal_vector, color='cyan')
                axes[0].set_title('Reconstructed Signal', color='white')
                axes[0].set_ylabel('Signal Amplitude (pixels)')
            else: axes[0].text(0.5, 0.5, 'Signal Plot Disabled', color='gray', ha='center', va='center')

            if self.image_log_options.include_fft_plot:
                axes[1].plot(result.fft_freqs, result.fft_mags, color='magenta')
                axes[1].set_title('FFT Magnitude Spectrum', color='white')
                axes[1].axvline(x=self.app_config.fft_cutoff_frequency, color='yellow', linestyle='--', linewidth=1)
                axes[1].set_xlim(left=0, right=0.5)
                axes[1].set_ylabel('Magnitude (A.U.)')
                axes[1].set_xlabel('Normalized Frequency')
            else: axes[1].text(0.5, 0.5, 'FFT Plot Disabled', color='gray', ha='center', va='center')

            for ax in axes: ax.set_facecolor('#2E2E2E'); ax.tick_params(axis='both', colors='white')
            fig.tight_layout(); fig.savefig(plot_path, facecolor=fig.get_facecolor()); plt.close(fig)

            plot_img = cv2.imread(plot_path)
            if plot_img is None: raise IOError("Failed to load temp plot image")
            
            h_roi, _, _ = result.roi_image.shape
            def resize(img, h): _h,w,_=img.shape; s=h/_h if _h>0 else 0; return cv2.resize(img,(int(w*s),h)) if s>0 else img
            
            image_parts = []
            if self.image_log_options.include_screenshot: image_parts.append(result.roi_image)
            if self.image_log_options.include_color_filter: image_parts.append(resize(cv2.cvtColor(result.color_mask, cv2.COLOR_GRAY2BGR), h_roi))
            image_parts.append(resize(plot_img, h_roi))

            cls, color = ("HF", (0, 0, 255)) if result.is_high_frequency else ("LF", (0, 255, 0))
            text_box = np.zeros((h_roi, 240, 3), dtype=np.uint8)
            cv2.putText(text_box, f"HF Ratio: {result.energy_ratio:.3e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(text_box, f"HF Energy: {result.high_freq_energy:.3e} (pixels^2)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(text_box, "Result:", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(text_box, f"{cls}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            image_parts.append(text_box)
            
            if image_parts:
                combined = cv2.hconcat(image_parts)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"image_logs/{ts}_frame{frame_count}_{region_name}.jpg", combined)
        except Exception as e: logger.error(f"Failed to create visual log for {region_name}: {e}")
        finally:
            if os.path.exists(plot_path): os.remove(plot_path)

    def _create_summary_log(self, results_dict: Dict[str, WaveAnalysisResult], overall_is_hf: bool, frame_count: int):
        if not results_dict: return
        try:
            num_regions = len(results_dict)
            fig, axes = plt.subplots(num_regions, 2, figsize=(12, 4 * num_regions), squeeze=False)
            fig.patch.set_facecolor('#1E1E1E')

            for i, (name, result) in enumerate(results_dict.items()):
                ax_signal, ax_fft = axes[i, 0], axes[i, 1]
                ax_signal.plot(result.signal_vector, color='cyan')
                ax_signal.set_title(f"{name} | HF Ratio: {result.energy_ratio:.3e}\nHF Energy: {result.high_freq_energy:.3e} (pixels^2)", color='white', fontsize=9)
                ax_signal.set_ylabel('Amplitude (pixels)', color='white', fontsize=8)
                ax_fft.plot(result.fft_freqs, result.fft_mags, color='magenta')
                ax_fft.set_title("FFT Magnitude Spectrum", color='white', fontsize=10)
                ax_fft.set_ylabel('Magnitude (A.U.)', color='white', fontsize=8)
                ax_fft.set_xlabel('Normalized Freq.', color='white', fontsize=8)
                ax_fft.axvline(x=self.app_config.fft_cutoff_frequency, color='yellow', linestyle='--', linewidth=1); ax_fft.set_xlim(left=0, right=0.5)
                for ax in [ax_signal, ax_fft]: ax.set_facecolor('#2E2E2E'); ax.tick_params(axis='both', colors='white', labelsize=8)

            avg_ratio = np.mean([res.energy_ratio for res in results_dict.values()])
            avg_energy = np.mean([res.high_freq_energy for res in results_dict.values()])
            fig.suptitle(f"Frame {frame_count} Summary | Avg HF Ratio: {avg_ratio:.3e} | Avg HF Energy: {avg_energy:.3e} | Overall: {'HF' if overall_is_hf else 'LF'}", 
                         color='white', fontsize=14, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            ts = time.strftime("%Y%m%d_%H%M%S")
            fig.savefig(f"image_logs/{ts}_frame{frame_count}_SUMMARY.jpg", facecolor=fig.get_facecolor()); plt.close(fig)
        except Exception as e: logger.error(f"Failed to create summary collage log: {e}")

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
    def __init__(self, parent):
        super().__init__(parent); self.title("Advanced Region & Color Configuration Tool"); self.geometry("1400x900")
        self.app_config = AppConfig(); self.screenshot, self.photo, self.scale = None, None, 1.0
        self.drawing, self.start_x, self.start_y, self.selected_region_name = False, 0, 0, None
        self._setup_gui()
    def _setup_gui(self):
        toolbar = ttk.Frame(self, padding=5); toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load Config", command=self._load_config).pack(side=tk.LEFT, padx=2)
        main_frame = ttk.Frame(self, padding=5); main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas_frame = ttk.LabelFrame(main_frame, text="Screenshot Preview"); canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, bg="black"); self.canvas.pack(fill=tk.BOTH, expand=True)
        right_panel = ttk.Frame(main_frame, width=400); right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5); right_panel.pack_propagate(False)
        ttk.Button(ttk.LabelFrame(right_panel, text="Capture"), text="Take Screenshot", command=self._take_screenshot).pack(pady=5, padx=5, fill=tk.X)
        params_frame = ttk.LabelFrame(right_panel, text="FFT Analysis Parameters"); params_frame.pack(fill=tk.X, pady=5)
        self.param_vars = {'fft_cutoff_frequency': tk.DoubleVar(value=self.app_config.fft_cutoff_frequency), 'fft_energy_ratio_threshold': tk.DoubleVar(value=self.app_config.fft_energy_ratio_threshold)}
        ttk.Label(params_frame, text="Cutoff Frequency (0-0.5):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=0.0, to=0.5, increment=0.01, textvariable=self.param_vars['fft_cutoff_frequency'], width=8).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(params_frame, text="Energy Ratio Threshold (0-1):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.001, textvariable=self.param_vars['fft_energy_ratio_threshold'], width=8).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(params_frame, text="Apply Parameters", command=self._apply_params).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(ttk.LabelFrame(right_panel, text="Signal Color Calibration"), text="Calibrate Signal Color...", command=self._launch_hsv_thresholder).pack(pady=5, fill=tk.X)
        list_frame = ttk.LabelFrame(right_panel, text="Defined Regions"); list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(list_frame, text="Delete Selected Region", command=self._delete_selected_region).pack(fill=tk.X, pady=(0, 5))
        self.region_listbox = tk.Listbox(list_frame, height=10); self.region_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.region_listbox.yview); list_scroll.pack(side=tk.RIGHT, fill=tk.Y); self.region_listbox.config(yscrollcommand=list_scroll.set)
        editor_frame = ttk.LabelFrame(right_panel, text="Region Editor"); editor_frame.pack(fill=tk.X, pady=5)
        self.editor_vars = {'name': tk.StringVar(),'x': tk.IntVar(),'y': tk.IntVar(),'width': tk.IntVar(),'height': tk.IntVar(),'roi_type': tk.StringVar(),'enabled': tk.BooleanVar()}
        ttk.Label(editor_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2); ttk.Entry(editor_frame, textvariable=self.editor_vars['name']).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        for i, key in enumerate(['x', 'y', 'width', 'height']):
            ttk.Label(editor_frame, text=f"{key.capitalize()}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2); ttk.Spinbox(editor_frame, from_=0, to=9999, textvariable=self.editor_vars[key], width=8).grid(row=i+1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(editor_frame, text="Type:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2); ttk.Combobox(editor_frame, textvariable=self.editor_vars['roi_type'], values=['wave', 'status', 'text'], state='readonly').grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Checkbutton(editor_frame, text="Enabled", variable=self.editor_vars['enabled']).grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(editor_frame, text="Update Region", command=self._update_region_from_editor).grid(row=7, column=0, columnspan=2, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click); self.canvas.bind("<B1-Motion>", self._update_selection); self.canvas.bind("<ButtonRelease-1>", self._end_selection); self.region_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
    def _take_screenshot(self): self.withdraw(); time.sleep(0.5); self.screenshot = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR); self.deiconify(); self.lift(); self.focus_force(); self._display_screenshot()
    def _display_screenshot(self):
        if self.screenshot is None: return
        self.canvas.delete("all"); self.after(50, self.__display_screenshot_resized)
    def __display_screenshot_resized(self):
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w<50 or canvas_h<50: self.after(100, self.__display_screenshot_resized); return
        img_h, img_w = self.screenshot.shape[:2]
        self.scale = min(canvas_w/img_w, canvas_h/img_h, 1.0)
        disp_w, disp_h = int(img_w*self.scale), int(img_h*self.scale)
        img_resized = Image.fromarray(cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2RGB)).resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img_resized); self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags="screenshot"); self._redraw_regions_on_canvas()
    def _on_canvas_click(self, event): self.drawing, self.start_x, self.start_y = True, event.x, event.y; self.canvas.delete("selection_rect")
    def _update_selection(self, event):
        if self.drawing: self.canvas.delete("selection_rect"); self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", width=2, tags="selection_rect")
    def _end_selection(self, event):
        if not self.drawing: return
        self.drawing = False; name = f"region_{len(self.app_config.regions)+1}"
        x1, y1 = int(min(self.start_x, event.x)/self.scale), int(min(self.start_y, event.y)/self.scale)
        x2, y2 = int(max(self.start_x, event.x)/self.scale), int(max(self.start_y, event.y)/self.scale)
        self.app_config.regions[name] = MonitoringRegion(name, x1, y1, x2-x1, y2-y1, 'wave', True); self.canvas.delete("selection_rect"); self._update_ui_from_data()
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
        for name, r in self.app_config.regions.items():
            x1,y1,x2,y2 = r.x*self.scale, r.y*self.scale, (r.x+r.width)*self.scale, (r.y+r.height)*self.scale
            color = colors.get(r.roi_type,"white") if r.enabled else "gray"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=("region", name)); self.canvas.create_text(x1+5, y1+5, text=name, fill=color, anchor="nw", tags=("region", name))
    def _on_listbox_select(self, _):
        if not self.region_listbox.curselection(): return
        self.selected_region_name = self.region_listbox.get(self.region_listbox.curselection()).replace(" (Disabled)", "")
        for key, var in self.editor_vars.items(): var.set(getattr(self.app_config.regions[self.selected_region_name], key))
    def _update_region_from_editor(self):
        if not self.selected_region_name: return messagebox.showerror("Error", "No region selected.", parent=self)
        old, new = self.selected_region_name, self.editor_vars['name'].get()
        if new != old and new in self.app_config.regions: return messagebox.showerror("Error", "Region name must be unique.", parent=self)
        del self.app_config.regions[old]; self.app_config.regions[new] = MonitoringRegion(**{k: v.get() for k, v in self.editor_vars.items()}); self.selected_region_name = new; self._update_ui_from_data()
    def _delete_selected_region(self):
        if not self.selected_region_name: return messagebox.showerror("Error", "No region selected.", parent=self)
        if messagebox.askyesno("Confirm Delete", f"Delete '{self.selected_region_name}'?", parent=self):
            del self.app_config.regions[self.selected_region_name]; self.selected_region_name = None
            for v in self.editor_vars.values(): v.set(""); self.editor_vars['enabled'].set(False); self._update_ui_from_data()
    def _save_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialdir="configs", parent=self)
        if not path: return
        try:
            self._apply_params(); data = {n: asdict(r) for n, r in self.app_config.regions.items()}
            data['_metadata'] = {k: getattr(self.app_config, k) for k in ['hsv_lower','hsv_upper','screenshot_interval','fft_cutoff_frequency','fft_energy_ratio_threshold']}
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
        self.root = root; self.root.title("USMA v.0.3.6"); self.root.geometry("800x400")
        self.config_path = tk.StringVar(value="configs/default_config.json")
        self.is_monitoring, self.is_overlay_on = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.verbose_logging_on = tk.BooleanVar(value=True)
        self.image_logging_on = tk.BooleanVar(value=False)
        self.log_opt_screenshot = tk.BooleanVar(value=False)
        self.log_opt_color_filter = tk.BooleanVar(value=False)
        self.log_opt_signal_plot = tk.BooleanVar(value=False)
        self.log_opt_fft_plot = tk.BooleanVar(value=False)
        self.audio_feedback_on = tk.BooleanVar(value=False)

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

        logging_main_frame = ttk.Frame(control_frame); logging_main_frame.pack(side=tk.LEFT, padx=10, pady=5)
        debug_frame = ttk.LabelFrame(logging_main_frame, text="General Logging"); debug_frame.pack(fill=tk.X)
        self.verbose_check = ttk.Checkbutton(debug_frame, text="Verbose Log", variable=self.verbose_logging_on); self.verbose_check.pack(anchor=tk.W, padx=5, pady=2)
        self.img_log_check = ttk.Checkbutton(debug_frame, text="Image Log", variable=self.image_logging_on, command=self._toggle_img_log_options_state); self.img_log_check.pack(anchor=tk.W, padx=5, pady=2)
        self.img_log_options_frame = ttk.LabelFrame(logging_main_frame, text="Image Log Options"); self.img_log_options_frame.pack(fill=tk.X, pady=(5,0))
        for txt, var in [("Screenshot",self.log_opt_screenshot), ("Color Filter",self.log_opt_color_filter), ("Signal Plot",self.log_opt_signal_plot), ("FFT Plot",self.log_opt_fft_plot)]: ttk.Checkbutton(self.img_log_options_frame, text=txt, variable=var).pack(anchor=tk.W, padx=15)
        
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
        self.config_path.set(path); self.monitor.update_config(path) 
        try: self.sample_frequency.set(round(1.0/self.monitor.app_config.screenshot_interval, 2))
        except (ZeroDivisionError,TypeError): self.sample_frequency.set(4.0)
        if self.is_overlay_on.get(): self._toggle_overlay(); self._toggle_overlay()
        self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
    
    def _launch_config_tool(self): config_window = ConfigToolWindow(self.root); config_window.grab_set()

    def _toggle_monitoring(self):
        controls = [self.load_button, self.edit_button, self.verbose_check, self.img_log_check, self.overlay_check, self.freq_spinbox, self.audio_check]
        if self.is_monitoring.get():
            self.monitor.stop(); self.is_monitoring.set(False); self.start_stop_button.config(text="Start Monitoring")
            self.status_label.config(text="Stopped."); self._reset_feedback_ui()
            for w in controls:
                if w == self.audio_check and not SOUND_DEVICE_AVAILABLE: continue
                w.config(state=tk.NORMAL)
            self._toggle_img_log_options_state()
        else:
            if not os.path.exists(self.config_path.get()): return messagebox.showerror("Error", "Config file not found.")
            try:
                freq = self.sample_frequency.get()
                if freq <= 0: return messagebox.showerror("Error", "Sample frequency must be positive.")
                self.monitor.app_config.screenshot_interval = 1.0 / freq
            except tk.TclError: return messagebox.showerror("Error", "Invalid sample frequency.")
            
            self.monitor.update_config(self.config_path.get())
            self.monitor.app_config.screenshot_interval = 1.0 / self.sample_frequency.get()
            self.monitor.set_audio_feedback(self.audio_feedback_on.get())
            
            log_opts = ImageLogOptions(self.log_opt_screenshot.get(),self.log_opt_color_filter.get(),self.log_opt_signal_plot.get(),self.log_opt_fft_plot.get())
            
            if self.monitor.start(self.verbose_logging_on.get(), self.image_logging_on.get(), log_opts):
                self.is_monitoring.set(True); self.start_stop_button.config(text="Stop Monitoring"); self.status_label.config(text="Monitoring active...")
                for w in controls: w.config(state=tk.DISABLED)
                self._toggle_img_log_options_state()

    def _toggle_overlay(self):
        if self.overlay and self.overlay.winfo_exists(): self.overlay.destroy(); self.overlay = None
        if self.is_overlay_on.get():
            if not os.path.exists(self.config_path.get()): messagebox.showerror("Error", "Config file not found."); self.is_overlay_on.set(False); return
            self.overlay = RegionOverlay(self.root, self.config_path.get())
            
    def _on_closing(self):
        if self.is_monitoring.get(): self.monitor.stop()
        if self.overlay and self.overlay.winfo_exists(): self.overlay.destroy()
        self.root.destroy()

# --- 8. APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = MonitorControlGUI(main_root)
    main_root.mainloop()

