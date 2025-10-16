# USMA (Unified Screen Monitoring Application) - v.0.3.10 Project Overview

## 1. Project Goal
The primary goal of USMA is to provide a user-friendly graphical tool to monitor user-defined regions on a computer screen. It is specifically designed to analyze line graphs within these regions, extracting the 1D signal, and classifying its frequency content as either High Frequency (HF) or Low Frequency (LF).

The application is built for tasks requiring real-time visual feedback on data volatility and has been enhanced to **export physically scaled engineering data** for professional post-processing and analysis.

## 2. What's New (v.0.3.10)
This release marks a significant step towards professional engineering use cases by introducing physical unit scaling and industry-standard data logging formats.

### Key Features
* **Physical Unit Scaling:** Users can now map the pixel dimensions of a monitoring region to real-world physical units. This includes defining the min/max values for the X-axis (e.g., in Hz) and Y-axis (e.g., in g/N, Pa, etc.).
* **Professional Data Logging:** Signal data can now be saved in formats ready for engineering analysis software:
    * **.mat files:** Logged with properly scaled frequency (Hz) and amplitude axes.
    * **.unv files:** Exported as Universal File Format (UFF) Type 58 datasets, fully compliant with industry standards for direct import into analysis software like Siemens Testlab.
* **Expanded Configuration Tool:** The configuration GUI has been updated with new input fields to easily manage physical scaling parameters and metadata for `.unv` headers (e.g., node and DOF information).
* **Real-time Audio Feedback:** Provides a continuous 400 Hz tone for the entire duration that a signal is classified as High Frequency (HF), offering immediate and intuitive status awareness. This feature can be toggled from the main GUI.
* **Adjustable Sample Frequency:** The main GUI includes a control to set the monitoring sample rate in Hz (defaulting to 4 Hz), allowing users to balance performance and analysis granularity.

## 3. Core Components of monitor_app.py
The application is architecturally divided into a GUI layer and a backend analysis engine that run on separate threads to maintain a responsive user interface.

### a. Main GUI (MonitorControlGUI)
**Functionality:** The central control window for loading configurations, starting/stopping the monitor, viewing live classification and FFT metrics, and toggling detailed logging options, including the new **.mat and .unv data export**.

### b. Configuration Tool (ConfigToolWindow & HSVThresholderWindow)
**Functionality:** A powerful sub-application for setting up monitoring tasks. Its capabilities now include region definition, color calibration, tuning of FFT analysis parameters, and **configuring physical axis scaling and UNV metadata**.

### c. Analysis Engine (ScreenMonitor)
**Functionality:** The core backend logic that runs on a background thread. It handles screen capture, signal analysis, and logging. It now applies the user-defined physical scaling when exporting data files.

## 4. Key Algorithm: Signal Reconstruction & FFT Classification

### a. Signal Reconstruction (Mathematical Model)
This algorithm converts the 2D pixel data of a graph into a 1D signal vector (unit: pixels) using a robust four-step process: Color Filtering, Coordinate Extraction, Anchor Point Generation, and Linear Interpolation. This pixel-based vector serves as the foundation for both real-time classification and the final scaled data export.

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
