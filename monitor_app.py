#!/usr/bin/env python3
"""
USMA (Unified Screen Monitoring Application) - v.0.3.4

A single, GUI-driven application that combines a professional-grade region 
configuration tool, real-time screen monitoring, visual overlay, and clear 
image logging.

v.0.3.4 Changes:
- Added customizable image log options to the main GUI, allowing users to select
  which components (screenshot, color mask, signal plot, FFT plot) to include.
- If no specific image log components are selected, only the summary log is created.
- Formatted all HF Energy Ratio and HF Energy outputs to scientific notation 
  with 3 significant figures for consistency and precision.
- Added HF Energy to the summary image log for each region.
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
logger.setLevel(logging.DEBUG) 
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
    include_screenshot: bool = True
    include_color_filter: bool = True
    include_signal_plot: bool = True
    include_fft_plot: bool = True

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
    screenshot_interval: float = 1.0
    fft_cutoff_frequency: float = 0.1  # Normalized frequency (0 to 0.5)
    fft_energy_ratio_threshold: float = 0.2 # 20% energy threshold


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
        logger.info(f"Screen monitoring thread started for USMA v.0.3.4")
        return True

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        logger.info("Screen monitoring stopped.")

    def update_config(self, new_config_path):
        self.config_path = new_config_path
        self.app_config = self._load_config(new_config_path)
        logger.info(f"Configuration updated to {new_config_path}")

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
            
            if not config.regions: logger.warning(f"Config {path} loaded, but no valid regions found.")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return AppConfig()
            
    def _monitoring_loop(self):
        while self.running:
            try:
                screenshot = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                results, overall_is_hf, avg_ratio, avg_hf_energy = self._process_frame(image, self.frame_count)
                
                if self.update_callback and overall_is_hf is not None:
                    self.update_callback(overall_is_hf, avg_ratio, avg_hf_energy)

                if self.frame_count % 10 == 0: 
                    self._log_results(results, self.frame_count)
                
                self.frame_count += 1
                time.sleep(self.app_config.screenshot_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}"); time.sleep(1)

    def _process_frame(self, image: np.ndarray, frame_count: int) -> (Dict, Optional[bool], Optional[float], Optional[float]):
        results = {'waves': {}}
        
        for name, region in self.app_config.regions.items():
            if region.enabled and region.roi_type == 'wave':
                roi = image[region.y:region.y+region.height, region.x:region.x+region.width]
                analysis_result = self._analyze_wave_pattern(roi, name)
                if analysis_result:
                    results['waves'][name] = analysis_result
                    # Only create detailed visual log if any sub-option is checked
                    if self.image_logging_enabled and any(asdict(self.image_log_options).values()):
                        self._create_visual_log(analysis_result, name, frame_count)
        
        overall_is_hf = None
        avg_energy_ratio = None
        avg_high_freq_energy = None

        if results['waves']:
            classifications = [res.is_high_frequency for res in results['waves'].values()]
            hf_count = sum(classifications)
            overall_is_hf = hf_count > len(classifications) / 2
            
            energy_ratios = [res.energy_ratio for res in results['waves'].values()]
            avg_energy_ratio = np.mean(energy_ratios)
            
            high_freq_energies = [res.high_freq_energy for res in results['waves'].values()]
            avg_high_freq_energy = np.mean(high_freq_energies)

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
            if self.verbose_logging_enabled:
                logger.debug(f"Region analysis skipped: Signal coverage of {coverage_ratio:.3e} is outside range.")
            return False

        cols_with_signal = np.count_nonzero(np.sum(color_mask, axis=0) > 0)
        continuity_ratio = cols_with_signal / width
        if continuity_ratio < 0.15:
            if self.verbose_logging_enabled:
                logger.debug(f"Region analysis skipped: Signal continuity of {continuity_ratio:.3e} is below threshold.")
            return False
            
        return True

    def _analyze_wave_pattern(self, roi: np.ndarray, region_name: str) -> Optional[WaveAnalysisResult]:
        if self.verbose_logging_enabled: logger.debug(f"--- Analyzing region: {region_name} ---")
        if roi.size == 0: 
            if self.verbose_logging_enabled: logger.debug("ROI is empty, skipping.")
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array(self.app_config.hsv_lower), np.array(self.app_config.hsv_upper)
        lower2, upper2 = np.array(self.app_config.hsv_lower2), np.array(self.app_config.hsv_upper2)
        mask1, mask2 = cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2)
        color_mask = mask1 + mask2
        
        if not self._validate_signal_quality(color_mask):
            return None

        y_coords, x_coords = np.nonzero(color_mask)
        width = color_mask.shape[1]
        
        if len(x_coords) == 0:
            signal_vector = np.full(width, roi.shape[0] / 2)
        else:
            unique_x = np.unique(x_coords)
            anchor_x, anchor_y = [], []
            for x_val in unique_x:
                y_values_for_x = y_coords[x_coords == x_val]
                if y_values_for_x.size > 0:
                    anchor_x.append(x_val)
                    anchor_y.append(np.mean(y_values_for_x))
            
            if len(anchor_x) < 2:
                default_y = anchor_y[0] if len(anchor_y) == 1 else roi.shape[0] / 2
                signal_vector = np.full(width, default_y)
            else:
                signal_vector = np.interp(np.arange(width), anchor_x, anchor_y)

        signal_vector = roi.shape[0] - signal_vector

        if signal_vector.size < 2: return None
        
        N = len(signal_vector)
        detrended_signal = signal_vector - np.mean(signal_vector)
        
        yf = rfft(detrended_signal)
        xf = rfftfreq(N, 1) 
        fft_mags = np.abs(yf)

        total_energy = np.sum(fft_mags**2)
        cutoff_freq = self.app_config.fft_cutoff_frequency
        cutoff_indices = np.where(xf >= cutoff_freq)[0]
        
        high_freq_energy = 0
        energy_ratio = 0
        is_hf = False

        if total_energy > 1e-9 and cutoff_indices.size > 0:
            first_cutoff_idx = cutoff_indices[0]
            high_freq_energy = np.sum(fft_mags[first_cutoff_idx:]**2)
            energy_ratio = high_freq_energy / total_energy
            is_hf = energy_ratio > self.app_config.fft_energy_ratio_threshold
        
        return WaveAnalysisResult(is_hf, energy_ratio, high_freq_energy, signal_vector, xf, fft_mags, roi.copy(), color_mask.copy())

    def _create_visual_log(self, result: WaveAnalysisResult, region_name: str, frame_count: int):
        plot_path = f"image_logs/temp_plot_{frame_count}_{region_name}.png"
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
            fig.patch.set_facecolor('#1E1E1E')
            
            # Conditionally generate plots
            if self.image_log_options.include_signal_plot:
                axes[0].plot(result.signal_vector, color='cyan')
                axes[0].set_title('Reconstructed Signal', color='white')
            else:
                axes[0].text(0.5, 0.5, 'Signal Plot Disabled', color='gray', ha='center', va='center')

            if self.image_log_options.include_fft_plot:
                axes[1].plot(result.fft_freqs, result.fft_mags, color='magenta')
                axes[1].set_title('FFT Magnitude Spectrum', color='white')
                axes[1].axvline(x=self.app_config.fft_cutoff_frequency, color='yellow', linestyle='--', linewidth=1)
                axes[1].set_xlim(left=0, right=0.5)
            else:
                 axes[1].text(0.5, 0.5, 'FFT Plot Disabled', color='gray', ha='center', va='center')

            for ax in axes:
                ax.set_facecolor('#2E2E2E')
                ax.tick_params(axis='both', colors='white')

            fig.tight_layout()
            fig.savefig(plot_path, facecolor=fig.get_facecolor())
            plt.close(fig)

            plot_img = cv2.imread(plot_path)
            if plot_img is None: raise IOError("Failed to load temp plot image")
            
            h_roi, _, _ = result.roi_image.shape
            
            def resize(img, h):
                _h, w, _ = img.shape; s = h / _h if _h > 0 else 0
                return cv2.resize(img, (int(w * s), h)) if s > 0 else img

            image_parts = []
            if self.image_log_options.include_screenshot:
                image_parts.append(result.roi_image)
            if self.image_log_options.include_color_filter:
                mask_bgr = cv2.cvtColor(result.color_mask, cv2.COLOR_GRAY2BGR)
                image_parts.append(resize(mask_bgr, h_roi))

            image_parts.append(resize(plot_img, h_roi))

            cls = "HF" if result.is_high_frequency else "LF"
            color = (0, 0, 255) if cls == "HF" else (0, 255, 0)
            text_box = np.zeros((h_roi, 240, 3), dtype=np.uint8)
            cv2.putText(text_box, f"HF Ratio: {result.energy_ratio:.3e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(text_box, f"HF Energy: {result.high_freq_energy:.3e}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(text_box, f"Result:", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
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
        num_regions = len(results_dict)
        if num_regions == 0: return

        try:
            fig, axes = plt.subplots(num_regions, 2, figsize=(12, 4 * num_regions), squeeze=False)
            fig.patch.set_facecolor('#1E1E1E')

            for i, (name, result) in enumerate(results_dict.items()):
                ax_signal, ax_fft = axes[i, 0], axes[i, 1]

                ax_signal.plot(result.signal_vector, color='cyan')
                title = (f"{name} | HF Ratio: {result.energy_ratio:.3e}\n"
                         f"HF Energy: {result.high_freq_energy:.3e}")
                ax_signal.set_title(title, color='white', fontsize=10)
                ax_signal.set_facecolor('#2E2E2E')
                ax_signal.tick_params(axis='both', colors='white', labelsize=8)

                ax_fft.plot(result.fft_freqs, result.fft_mags, color='magenta')
                ax_fft.set_title("FFT Magnitude Spectrum", color='white', fontsize=10)
                ax_fft.set_facecolor('#2E2E2E')
                ax_fft.tick_params(axis='both', colors='white', labelsize=8)
                ax_fft.axvline(x=self.app_config.fft_cutoff_frequency, color='yellow', linestyle='--', linewidth=1)
                ax_fft.set_xlim(left=0, right=0.5)

            avg_ratio = np.mean([res.energy_ratio for res in results_dict.values()])
            avg_energy = np.mean([res.high_freq_energy for res in results_dict.values()])
            classification = "HF" if overall_is_hf else "LF"
            fig.suptitle(f"Frame {frame_count} Summary | Avg HF Ratio: {avg_ratio:.3e} | Avg HF Energy: {avg_energy:.3e} | Overall: {classification}", 
                         color='white', fontsize=14, y=0.99)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_path = f"image_logs/{ts}_frame{frame_count}_SUMMARY.jpg"
            fig.savefig(final_path, facecolor=fig.get_facecolor())
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create summary collage log: {e}")

    def _log_results(self, results: Dict, frame_count: int):
        for name, res in results.get('waves', {}).items():
            status = "HF" if res.is_high_frequency else "LF"
            logger.info(f"Frame {frame_count}: {name} -> {status} (HF Ratio: {res.energy_ratio:.3e}, HF Energy: {res.high_freq_energy:.3e})")

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
        padding = 10 
        for name, region_data in data.items():
            if not name.startswith('_') and region_data.get('enabled', True):
                x,y,w,h = region_data['x'], region_data['y'], region_data['width'], region_data['height']
                color = colors.get(region_data.get('roi_type', 'wave'), "#95a5a6")
                canvas.create_rectangle(x - padding, y - padding, x + w + padding, y + h + padding, outline=color, width=3)
                canvas.create_text(x - padding, y - padding, text=name, anchor="sw", font=("Arial", 10, "bold"), fill=color)
        label = f"Config: {os.path.basename(self.config_path)}"
        canvas.create_text(self.winfo_screenwidth()-10, self.winfo_screenheight()-10, text=label, anchor="se", fill="#333")


# --- 5. INTERACTIVE HSV THRESHOLDER ---
class HSVThresholderWindow(tk.Toplevel):
    def __init__(self, parent, screenshot_cv2, regions, current_config, callback):
        super().__init__(parent)
        self.title("Interactive HSV Thresholder")
        self.geometry("1000x700")
        self.parent = parent
        self.full_screenshot = screenshot_cv2
        self.regions = regions
        self.callback = callback
        self.active_roi_cv2 = None
        self.hsv_image = None
        top_frame = ttk.Frame(self, padding=5); top_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top_frame, text="Select Region:").pack(side=tk.LEFT, padx=5)
        self.region_selector = ttk.Combobox(top_frame, state="readonly", values=[r.name for r in regions.values() if r.roi_type == 'wave'])
        self.region_selector.pack(side=tk.LEFT, padx=5)
        self.region_selector.bind("<<ComboboxSelected>>", self._on_region_select)
        self.img_label = ttk.Label(self); self.img_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        controls_frame = ttk.Frame(self, padding=10); controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.hsv_vars = {key: tk.IntVar() for key in ['HMin', 'SMin', 'VMin', 'HMax', 'SMax', 'VMax']}
        def create_slider(parent, text, var, from_, to):
            frame = ttk.Frame(parent)
            def on_change(val, op): var.set(var.get() + op); self._update_mask()
            ttk.Button(frame, text="-", width=2, command=lambda: on_change(var.get(), -1)).pack(side=tk.LEFT)
            tk.Scale(frame, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL, length=200, command=self._update_mask, showvalue=0).pack(side=tk.LEFT, expand=True, fill=tk.X)
            ttk.Button(frame, text="+", width=2, command=lambda: on_change(var.get(), 1)).pack(side=tk.LEFT)
            ttk.Label(frame, textvariable=var, width=4).pack(side=tk.LEFT)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(parent, text=text).pack(anchor='w')
        create_slider(controls_frame, "Hue Min", self.hsv_vars['HMin'], 0, 179)
        create_slider(controls_frame, "Hue Max", self.hsv_vars['HMax'], 0, 179)
        create_slider(controls_frame, "Sat Min", self.hsv_vars['SMin'], 0, 255)
        create_slider(controls_frame, "Sat Max", self.hsv_vars['SMax'], 0, 255)
        create_slider(controls_frame, "Val Min", self.hsv_vars['VMin'], 0, 255)
        create_slider(controls_frame, "Val Max", self.hsv_vars['VMax'], 0, 255)
        self.hsv_vars['HMin'].set(current_config.hsv_lower[0]); self.hsv_vars['SMin'].set(current_config.hsv_lower[1]); self.hsv_vars['VMin'].set(current_config.hsv_lower[2])
        self.hsv_vars['HMax'].set(current_config.hsv_upper[0]); self.hsv_vars['SMax'].set(current_config.hsv_upper[1]); self.hsv_vars['VMax'].set(current_config.hsv_upper[2])
        ttk.Button(controls_frame, text="Apply & Close", command=self._apply_and_close).pack(pady=20, side=tk.BOTTOM)
        self.grab_set()
        if self.region_selector['values']: self.region_selector.current(0); self._on_region_select(None)
    def _on_region_select(self, _=None):
        region = self.regions.get(self.region_selector.get())
        if not region: return
        self.active_roi_cv2 = self.full_screenshot[region.y:region.y+region.height, region.x:region.x+region.width]
        self.hsv_image = cv2.cvtColor(self.active_roi_cv2, cv2.COLOR_BGR2HSV)
        self._update_mask()
    def _update_mask(self, _=None):
        if self.hsv_image is None: return
        lower = np.array([self.hsv_vars['HMin'].get(), self.hsv_vars['SMin'].get(), self.hsv_vars['VMin'].get()])
        upper = np.array([self.hsv_vars['HMax'].get(), self.hsv_vars['SMax'].get(), self.hsv_vars['VMax'].get()])
        mask = cv2.inRange(self.hsv_image, lower, upper)
        output = cv2.bitwise_and(self.active_roi_cv2, self.active_roi_cv2, mask=mask)
        img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        w, h = img_pil.size; max_w, max_h = self.img_label.winfo_width(), self.img_label.winfo_height()
        if max_w < 50 or max_h < 50: max_w, max_h = 700, 600
        scale = min(max_w/w, max_h/h)
        if scale < 1: img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.img_label.config(image=self.photo)
    def _apply_and_close(self):
        self.callback({key: var.get() for key, var in self.hsv_vars.items()})
        self.destroy()


# --- 6. CONFIGURATION TOOL ---
class ConfigToolWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Advanced Region & Color Configuration Tool"); self.geometry("1400x900")
        self.app_config = AppConfig(); self.screenshot, self.photo, self.scale = None, None, 1.0
        self.drawing, self.start_x, self.start_y = False, 0, 0
        self.selected_region_name = None
        self._setup_gui()
    def _setup_gui(self):
        toolbar = ttk.Frame(self, padding=5); toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load Config", command=self._load_config).pack(side=tk.LEFT, padx=2)
        main_frame = ttk.Frame(self, padding=5); main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas_frame = ttk.LabelFrame(main_frame, text="Screenshot Preview"); canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, bg="black"); self.canvas.pack(fill=tk.BOTH, expand=True)
        right_panel = ttk.Frame(main_frame, width=400); right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5); right_panel.pack_propagate(False)
        screenshot_frame = ttk.LabelFrame(right_panel, text="Capture"); screenshot_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(screenshot_frame, text="Take Screenshot", command=self._take_screenshot).pack(pady=5, padx=5)
        params_frame = ttk.LabelFrame(right_panel, text="FFT Analysis Parameters"); params_frame.pack(fill=tk.X, pady=5)
        self.param_vars = {'fft_cutoff_frequency': tk.DoubleVar(value=self.app_config.fft_cutoff_frequency), 'fft_energy_ratio_threshold': tk.DoubleVar(value=self.app_config.fft_energy_ratio_threshold)}
        ttk.Label(params_frame, text="Cutoff Frequency (0-0.5):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=0.0, to=0.5, increment=0.01, textvariable=self.param_vars['fft_cutoff_frequency'], width=8).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(params_frame, text="Energy Ratio Threshold (0-1):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.param_vars['fft_energy_ratio_threshold'], width=8).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(params_frame, text="Apply Parameters", command=self._apply_params).grid(row=2, column=0, columnspan=2, pady=5)
        color_frame = ttk.LabelFrame(right_panel, text="Signal Color Calibration"); color_frame.pack(fill=tk.X, pady=5)
        ttk.Button(color_frame, text="Calibrate Signal Color...", command=self._launch_hsv_thresholder).pack(pady=5)
        list_frame = ttk.LabelFrame(right_panel, text="Defined Regions"); list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(list_frame, text="Delete Selected Region", command=self._delete_selected_region).pack(fill=tk.X, pady=(0, 5))
        self.region_listbox = tk.Listbox(list_frame, height=10); self.region_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.region_listbox.yview); list_scroll.pack(side=tk.RIGHT, fill=tk.Y); self.region_listbox.config(yscrollcommand=list_scroll.set)
        editor_frame = ttk.LabelFrame(right_panel, text="Region Editor"); editor_frame.pack(fill=tk.X, pady=5)
        self.editor_vars = {'name': tk.StringVar(),'x': tk.IntVar(),'y': tk.IntVar(),'width': tk.IntVar(),'height': tk.IntVar(),'roi_type': tk.StringVar(),'enabled': tk.BooleanVar()}
        ttk.Label(editor_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(editor_frame, textvariable=self.editor_vars['name']).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        for i, key in enumerate(['x', 'y', 'width', 'height']):
            ttk.Label(editor_frame, text=f"{key.capitalize()}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Spinbox(editor_frame, from_=0, to=9999, textvariable=self.editor_vars[key], width=8).grid(row=i+1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(editor_frame, text="Type:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Combobox(editor_frame, textvariable=self.editor_vars['roi_type'], values=['wave', 'status', 'text'], state='readonly').grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)
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
        self.photo = ImageTk.PhotoImage(image=img_resized)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags="screenshot")
        self._redraw_regions_on_canvas()
    def _on_canvas_click(self, event): self.drawing, self.start_x, self.start_y = True, event.x, event.y; self.canvas.delete("selection_rect")
    def _update_selection(self, event):
        if not self.drawing: return
        self.canvas.delete("selection_rect"); self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", width=2, tags="selection_rect")
    def _end_selection(self, event):
        if not self.drawing: return
        self.drawing = False; name = f"region_{len(self.app_config.regions)+1}"
        x1, y1 = int(min(self.start_x, event.x)/self.scale), int(min(self.start_y, event.y)/self.scale)
        x2, y2 = int(max(self.start_x, event.x)/self.scale), int(max(self.start_y, event.y)/self.scale)
        new_region = MonitoringRegion(name, x1, y1, x2-x1, y2-y1, 'wave', True)
        self.app_config.regions[name] = new_region; self.canvas.delete("selection_rect"); self._update_ui_from_data()
    def _launch_hsv_thresholder(self):
        wave_regions = {name: region for name, region in self.app_config.regions.items() if region.roi_type == 'wave'}
        if self.screenshot is None: return messagebox.showwarning("Warning", "Please take a screenshot first.", parent=self)
        if not wave_regions: return messagebox.showwarning("Warning", "Please define at least one 'wave' region first.", parent=self)
        HSVThresholderWindow(self, self.screenshot, wave_regions, self.app_config, self._apply_new_hsv)
    def _apply_new_hsv(self, hsv_values):
        self.app_config.hsv_lower = [hsv_values['HMin'], hsv_values['SMin'], hsv_values['VMin']]
        self.app_config.hsv_upper = [hsv_values['HMax'], hsv_values['SMax'], hsv_values['VMax']]
        self.app_config.hsv_lower2, self.app_config.hsv_upper2 = self.app_config.hsv_lower, self.app_config.hsv_upper
        logger.info("HSV thresholds updated via interactive calibrator.")
    def _apply_params(self):
        self.app_config.fft_cutoff_frequency = self.param_vars['fft_cutoff_frequency'].get()
        self.app_config.fft_energy_ratio_threshold = self.param_vars['fft_energy_ratio_threshold'].get()
        messagebox.showinfo("Success", "Analysis parameters updated.", parent=self)
    def _update_ui_from_data(self):
        sel_name = self.selected_region_name; self.region_listbox.delete(0, tk.END)
        for i, name in enumerate(sorted(self.app_config.regions.keys())):
            disp_name = f"{name}" if self.app_config.regions[name].enabled else f"{name} (Disabled)"
            self.region_listbox.insert(tk.END, disp_name)
            if name == sel_name: self.region_listbox.selection_set(i)
        self.param_vars['fft_cutoff_frequency'].set(self.app_config.fft_cutoff_frequency)
        self.param_vars['fft_energy_ratio_threshold'].set(self.app_config.fft_energy_ratio_threshold)
        self._redraw_regions_on_canvas()
    def _redraw_regions_on_canvas(self):
        self.canvas.delete("region")
        colors = {"wave":"lime","status":"cyan","text":"yellow"}
        for name, region in self.app_config.regions.items():
            x1,y1,x2,y2 = region.x*self.scale, region.y*self.scale, (region.x+region.width)*self.scale, (region.y+region.height)*self.scale
            color = colors.get(region.roi_type,"white") if region.enabled else "gray"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=("region", name))
            self.canvas.create_text(x1+5, y1+5, text=name, fill=color, anchor="nw", tags=("region", name))
    def _on_listbox_select(self, event):
        if not self.region_listbox.curselection(): return
        disp_name = self.region_listbox.get(self.region_listbox.curselection())
        name = disp_name.replace(" (Disabled)", "")
        self.selected_region_name = name; region = self.app_config.regions[name]
        for key, var in self.editor_vars.items(): var.set(getattr(region, key))
    def _update_region_from_editor(self):
        if not self.selected_region_name: return messagebox.showerror("Error", "No region selected.", parent=self)
        old, new = self.selected_region_name, self.editor_vars['name'].get()
        if new != old and new in self.app_config.regions: return messagebox.showerror("Error", "Region name must be unique.", parent=self)
        del self.app_config.regions[old]
        updated = MonitoringRegion(**{k: v.get() for k, v in self.editor_vars.items()})
        self.app_config.regions[new], self.selected_region_name = updated, new
        self._update_ui_from_data()
    def _delete_selected_region(self):
        if not self.selected_region_name: return messagebox.showerror("Error", "No region selected.", parent=self)
        if messagebox.askyesno("Confirm Delete", f"Delete '{self.selected_region_name}'?", parent=self):
            del self.app_config.regions[self.selected_region_name]; self.selected_region_name = None
            for var in self.editor_vars.values(): var.set("")
            self.editor_vars['enabled'].set(False); self._update_ui_from_data()
    def _save_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialdir="configs", parent=self)
        if not path: return
        try:
            self._apply_params() 
            data = {name: asdict(r) for name, r in self.app_config.regions.items()}
            data['_metadata'] = {'hsv_lower': self.app_config.hsv_lower, 'hsv_upper': self.app_config.hsv_upper, 'hsv_lower2': self.app_config.hsv_lower2, 'hsv_upper2': self.app_config.hsv_upper2, 'screenshot_interval': self.app_config.screenshot_interval, 'fft_cutoff_frequency': self.app_config.fft_cutoff_frequency, 'fft_energy_ratio_threshold': self.app_config.fft_energy_ratio_threshold}
            with open(path, 'w') as f: json.dump(data, f, indent=2)
            messagebox.showinfo("Success", f"Saved to {os.path.basename(path)}", parent=self)
        except Exception as e: messagebox.showerror("Error", f"Failed to save: {e}", parent=self)
    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], initialdir="configs", parent=self)
        if not path: return
        try:
            temp_monitor = ScreenMonitor(path); self.app_config = temp_monitor.app_config
            self._update_ui_from_data()
            if self.screenshot: self._redraw_regions_on_canvas()
            messagebox.showinfo("Success", f"Loaded {os.path.basename(path)}", parent=self)
        except Exception as e: messagebox.showerror("Error", f"Failed to load: {e}", parent=self)

# --- 7. MAIN GUI: THE CENTRAL CONTROL APPLICATION ---
class MonitorControlGUI:
    def __init__(self, root):
        self.root = root; self.root.title("USMA v.0.3.4"); self.root.geometry("650x400")
        self.config_path = tk.StringVar(value="configs/default_config.json")
        self.is_monitoring = tk.BooleanVar(value=False); self.is_overlay_on = tk.BooleanVar(value=False)
        self.verbose_logging_on = tk.BooleanVar(value=True); self.image_logging_on = tk.BooleanVar(value=True)
        
        # Image Log Options
        self.log_opt_screenshot = tk.BooleanVar(value=True)
        self.log_opt_color_filter = tk.BooleanVar(value=True)
        self.log_opt_signal_plot = tk.BooleanVar(value=True)
        self.log_opt_fft_plot = tk.BooleanVar(value=True)

        self.monitor = ScreenMonitor(self.config_path.get(), self.update_feedback_panel)
        self.overlay = None
        self._setup_main_gui(); self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    def _setup_main_gui(self):
        frame = ttk.Frame(self.root, padding=10); frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(frame, text="Configuration"); config_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(config_frame, textvariable=self.config_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.load_button = ttk.Button(config_frame, text="Load...", command=self._load_config); self.load_button.pack(side=tk.LEFT, padx=5)
        self.edit_button = ttk.Button(config_frame, text="Edit Config...", command=self._launch_config_tool); self.edit_button.pack(side=tk.LEFT, padx=5)
        
        feedback_frame = ttk.LabelFrame(frame, text="Live Analysis Feedback"); feedback_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.status_light = tk.Canvas(feedback_frame, width=30, height=30, bg="gray", highlightthickness=0)
        self.status_light.grid(row=0, column=0, rowspan=2, padx=15, pady=5)
        self.class_var = tk.StringVar(value="Overall: --"); self.hf_ratio_var = tk.StringVar(value="Avg HF Ratio: --"); self.hf_energy_var = tk.StringVar(value="Avg HF Energy: --")
        ttk.Label(feedback_frame, textvariable=self.class_var, font=("Segoe UI", 14)).grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(feedback_frame, textvariable=self.hf_ratio_var, font=("Segoe UI", 10)).grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(feedback_frame, textvariable=self.hf_energy_var, font=("Segoe UI", 10)).grid(row=1, column=2, sticky=tk.W, padx=10)
        feedback_frame.columnconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(frame, text="Controls"); control_frame.pack(fill=tk.X, pady=5)
        self.start_stop_button = ttk.Button(control_frame, text="Start Monitoring", command=self._toggle_monitoring)
        self.start_stop_button.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)
        self.overlay_check = ttk.Checkbutton(control_frame, text="Show Overlay", variable=self.is_overlay_on, command=self._toggle_overlay)
        self.overlay_check.pack(side=tk.LEFT, padx=10, pady=10)
        
        logging_main_frame = ttk.Frame(control_frame); logging_main_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        debug_frame = ttk.LabelFrame(logging_main_frame, text="General Logging"); debug_frame.pack(fill=tk.X)
        self.verbose_check = ttk.Checkbutton(debug_frame, text="Verbose Log", variable=self.verbose_logging_on); self.verbose_check.pack(anchor=tk.W, padx=5, pady=2)
        self.img_log_check = ttk.Checkbutton(debug_frame, text="Image Log", variable=self.image_logging_on, command=self._toggle_img_log_options_frame); self.img_log_check.pack(anchor=tk.W, padx=5, pady=2)

        self.img_log_options_frame = ttk.LabelFrame(logging_main_frame, text="Image Log Options"); self.img_log_options_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Checkbutton(self.img_log_options_frame, text="Screenshot", variable=self.log_opt_screenshot).pack(anchor=tk.W, padx=15)
        ttk.Checkbutton(self.img_log_options_frame, text="Color Filter", variable=self.log_opt_color_filter).pack(anchor=tk.W, padx=15)
        ttk.Checkbutton(self.img_log_options_frame, text="Signal Plot", variable=self.log_opt_signal_plot).pack(anchor=tk.W, padx=15)
        ttk.Checkbutton(self.img_log_options_frame, text="FFT Plot", variable=self.log_opt_fft_plot).pack(anchor=tk.W, padx=15)
        
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W); self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        self._toggle_img_log_options_frame() # Set initial state
    
    def _toggle_img_log_options_frame(self):
        state = tk.NORMAL if self.image_logging_on.get() else tk.DISABLED
        for child in self.img_log_options_frame.winfo_children():
            child.configure(state=state)

    def update_feedback_panel(self, overall_is_hf, avg_ratio, avg_hf_energy): self.root.after(0, self._update_feedback_ui, overall_is_hf, avg_ratio, avg_hf_energy)
    def _update_feedback_ui(self, overall_is_hf, avg_ratio, avg_hf_energy):
        if overall_is_hf: self.class_var.set("Overall: HF"); self.status_light.config(bg="red")
        else: self.class_var.set("Overall: LF"); self.status_light.config(bg="green")
        self.hf_ratio_var.set(f"Avg HF Ratio: {avg_ratio:.3e}"); self.hf_energy_var.set(f"Avg HF Energy: {avg_hf_energy:.3e}")
    def _reset_feedback_ui(self):
        self.class_var.set("Overall: --"); self.status_light.config(bg="gray"); self.hf_ratio_var.set("Avg HF Ratio: --"); self.hf_energy_var.set("Avg HF Energy: --")
    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], initialdir="configs", title="Select Config")
        if path:
            self.config_path.set(path); self.monitor.update_config(path) 
            if self.is_overlay_on.get(): self._toggle_overlay(); self._toggle_overlay()
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
    def _launch_config_tool(self): config_window = ConfigToolWindow(self.root); config_window.grab_set()
    def _toggle_monitoring(self):
        if self.is_monitoring.get():
            self.monitor.stop(); self.is_monitoring.set(False); self.start_stop_button.config(text="Start Monitoring")
            self.status_label.config(text="Stopped."); self._reset_feedback_ui()
            for w in [self.load_button, self.edit_button, self.verbose_check, self.img_log_check, self.overlay_check]: w.config(state=tk.NORMAL)
            self._toggle_img_log_options_frame()
        else:
            if not os.path.exists(self.config_path.get()): return messagebox.showerror("Error", "Config file not found.")
            self.monitor.update_config(self.config_path.get())
            
            log_opts = ImageLogOptions(
                include_screenshot=self.log_opt_screenshot.get(),
                include_color_filter=self.log_opt_color_filter.get(),
                include_signal_plot=self.log_opt_signal_plot.get(),
                include_fft_plot=self.log_opt_fft_plot.get()
            )
            
            if self.monitor.start(
                verbose_logging=self.verbose_logging_on.get(), 
                image_logging=self.image_logging_on.get(),
                image_log_options=log_opts):
                self.is_monitoring.set(True); self.start_stop_button.config(text="Stop Monitoring")
                self.status_label.config(text="Monitoring active...")
                for w in [self.load_button, self.edit_button, self.verbose_check, self.img_log_check, self.overlay_check]: w.config(state=tk.DISABLED)
                self._toggle_img_log_options_frame()

    def _toggle_overlay(self):
        if self.overlay and self.overlay.winfo_exists(): self.overlay.destroy(); self.overlay = None
        if self.is_overlay_on.get():
            if not os.path.exists(self.config_path.get()): messagebox.showerror("Error", "Config file not found."); self.is_overlay_on.set(False); return
            self.overlay = RegionOverlay(self.root, self.config_path.get()); self.status_label.config(text="Overlay enabled.")
        else: self.status_label.config(text="Overlay disabled.")
    def _on_closing(self):
        if self.is_monitoring.get(): self.monitor.stop()
        if self.overlay and self.overlay.winfo_exists(): self.overlay.destroy()
        self.root.destroy()

# --- 8. APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = MonitorControlGUI(main_root)
    main_root.mainloop()

