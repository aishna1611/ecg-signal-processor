# ECG Signal Processor

A clinical-grade ECG analysis pipeline built in Python.
Implements the Pan-Tompkins algorithm for R-peak detection
and computes HRV metrics on real PhysioNet data.

## Setup

```bash
pip install numpy scipy matplotlib wfdb
```

## Run day by day

```bash
python day1_raw_ecg.py        # Download + plot raw ECG
python day2_filtering.py      # Design + apply digital filters
python day3_peak_detection.py # Pan-Tompkins R-peak detection
python day4_hrv_analysis.py   # HRV metrics (SDNN, RMSSD, PSD)
python day5_report.py         # Generate PDF clinical report
```

## What it does

| Day | File | Output |
|-----|------|--------|
| 1 | day1_raw_ecg.py | Raw ECG plot with PQRST labels |
| 2 | day2_filtering.py | Filtered ECG + frequency response |
| 3 | day3_peak_detection.py | R-peaks with detection metrics |
| 4 | day4_hrv_analysis.py | Tachogram, Poincaré, PSD |
| 5 | day5_report.py | **ecg_clinical_report.pdf** |

## Algorithm

1. **Highpass filter** (0.5Hz) — removes baseline wander
2. **Lowpass filter** (40Hz) — removes EMG noise
3. **Pan-Tompkins** — differentiate → square → integrate → threshold → find peaks
4. **HRV metrics** — SDNN, RMSSD, pNN50, LF/HF ratio
5. **PDF report** — 6-panel clinical report with metrics table

## Data

MIT-BIH Arrhythmia Database, Record 100 (PhysioNet)
Downloaded automatically by the `wfdb` library.

## Skills demonstrated

- Digital filter design (Butterworth, zero-phase)
- Real-time signal processing (Pan-Tompkins 1985)
- Heart rate variability analysis (time + frequency domain)
- Clinical data visualization
- Automated report generation
