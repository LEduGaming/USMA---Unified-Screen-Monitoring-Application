USMA (Unified Screen Monitoring Application) - v.0.3.4 Project Overview
1. Project Goal
The primary goal of USMA is to provide a user-friendly graphical tool to monitor user-defined regions on a computer screen. It is specifically designed to analyze line graphs within these regions, extracting the signal, and classifying its frequency content as either High Frequency (HF) or Low Frequency (LF). The application is built for tasks requiring real-time visual feedback on the volatility of on-screen data visualizations.
2. Notable Changes in v.0.3.4
Customizable Image Logs: The main GUI now features a dedicated "Image Log Options" section. Users can select which specific components (screenshot, color filter, signal plot, FFT plot) are included in the detailed, per-region image logs. If "Image Log" is active but no sub-options are selected, only the summary collage is generated.
Consistent Scientific Notation: All outputs for HF Energy Ratio and HF Energy (in the live GUI, text logs, and image logs) are now formatted to scientific notation with three significant figures for improved precision and readability.
Enhanced Data in Logs: The high-frequency energy value is now displayed in all relevant logs, including the detailed image log's text box and the summary image log's titles, providing a more complete data picture.
3. Core Components of monitor_app.py
The application is architecturally divided into a GUI layer and a backend analysis engine that run on separate threads to maintain a responsive user interface.
a. Main GUI (MonitorControlGUI)
Functionality: The central control window for loading configurations, starting/stopping the monitor, viewing live classification and FFT metrics, and toggling detailed logging options.
b. Configuration Tool (ConfigToolWindow & HSVThresholderWindow)
Functionality: A powerful sub-application for setting up monitoring tasks, including region definition, color calibration, and tuning of FFT analysis parameters.
c. Analysis Engine (ScreenMonitor)
Functionality: The core backend logic that runs on a background thread to handle screen capture, signal analysis, and logging based on user-defined settings.
4. Key Algorithm: Signal Reconstruction & FFT Classification
a. Signal Reconstruction (Mathematical Model)
This algorithm converts the 2D pixel data of a graph into a 1D signal vector using a robust four-step process: Color Filtering, Coordinate Extraction, Anchor Point Generation, and Linear Interpolation.
b. Frequency Analysis: High-Frequency Energy Ratio
This model quantifies the signal's nature by calculating the percentage of its total energy contained within the high-frequency part of its spectrum.
Signal Detrending: The signal's mean is subtracted to remove the DC component.
Fast Fourier Transform (FFT): scipy.fft.rfft is applied to transform the signal into the frequency domain.
Energy Calculation:
A user-defined Cutoff Frequency divides the spectrum into "low" and "high" bands.
The total energy (E_total) and high-frequency energy (E_high) are calculated from the squared magnitudes of the spectrum.
Ratio and Classification:
The energy ratio is computed: R = E_high / E_total.
This ratio R is compared against a user-defined Energy Ratio Threshold.
If R is greater than the threshold, the signal is classified as HF; otherwise, it is LF.
