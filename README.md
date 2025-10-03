# USMA (Unified Screen Monitoring Application) - v.0.3.7 Project Overview

## 1. Project Goal
The primary goal of USMA is to provide a user-friendly graphical tool to monitor user-defined regions on a computer screen. It is specifically designed to analyze line graphs within these regions, extracting the signal, and classifying its frequency content as either High Frequency (HF) or Low Frequency (LF). The application is built for tasks requiring real-time visual feedback on the volatility of on-screen data visualizations.

## 2. What's New (v.0.3.5 - v.0.3.7)
This release cycle introduces major new features focused on real-time feedback and enhances the application's overall stability and robustness.

### Key Features
* **Real-time Audio Feedback:** A key feature has been added to provide continuous audio feedback. The application now generates a 400 Hz tone for the entire duration that a signal is classified as High Frequency (HF), offering immediate and intuitive status awareness. This feature can be toggled directly from the main GUI.
* **Adjustable Sample Frequency:** The main GUI now includes a control to set the monitoring sample rate in Hz, defaulting to a more responsive 4 Hz. This allows users to balance performance and analysis granularity on the fly.
* **New Usability Defaults:** Image logging is now disabled by default to prevent unintentional disk space usage, promoting a cleaner experience for users who primarily need live feedback.

### Stability and Robustness Enhancements (v.0.3.7)
* **Thread Safety:** Added lock synchronization for audio operations to prevent race conditions and ensure stable audio start/stop behavior.
* **Improved Error Handling:** The application is now more resilient. It gracefully handles audio device errors, prevents infinite loops during GUI initialization, and validates configuration files to avoid crashes from malformed JSON.
* **Code Refinement:** Removed redundant code and unused configuration fields, making the application more efficient and easier to maintain.

## 3. Core Components of monitor_app.py
The application is architecturally divided into a GUI layer and a backend analysis engine that run on separate threads to maintain a responsive user interface.

### a. Main GUI (MonitorControlGUI)
**Functionality:** The central control window for loading configurations, starting/stopping the monitor, viewing live classification and FFT metrics, and toggling detailed logging options.

### b. Configuration Tool (ConfigToolWindow & HSVThresholderWindow)
**Functionality:** A powerful sub-application for setting up monitoring tasks, including region definition, color calibration, and tuning of FFT analysis parameters.

### c. Analysis Engine (ScreenMonitor)
**Functionality:** The core backend logic that runs on a background thread to handle screen capture, signal analysis, and logging based on user-defined settings.

## 4. Key Algorithm: Signal Reconstruction & FFT Classification

### a. Signal Reconstruction (Mathematical Model)
This algorithm converts the 2D pixel data of a graph into a 1D signal vector (unit: pixels) using a robust four-step process: Color Filtering, Coordinate Extraction, Anchor Point Generation, and Linear Interpolation.

### b. Frequency Analysis: High-Frequency Energy Ratio
This model quantifies the signal's nature by calculating the percentage of its total "energy" contained within the high-frequency part of its spectrum. This "energy" is a signal processing term (unit: pixels²) and is not physical energy.

* **Signal Detrending:** The signal's mean is subtracted to remove the DC component.
* **Fast Fourier Transform (FFT):** `scipy.fft.rfft` is applied to transform the signal into the frequency domain.
* **Energy Calculation:**
    * A user-defined Cutoff Frequency divides the spectrum into "low" and "high" bands.
    * The total energy ($E_{total}$) and high-frequency energy ($E_{high}$) are calculated from the squared magnitudes of the spectrum.
* **Ratio and Classification:**
    * The dimensionless energy ratio is computed: $R = E_{high} / E_{total}$.
    * This ratio R is compared against a user-defined Energy Ratio Threshold.
    * If R is greater than the threshold, the signal is classified as HF; otherwise, it is LF.
