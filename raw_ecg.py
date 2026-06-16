
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# ── 1. Download record 100 from MIT-BIH Arrhythmia Database ──────────────────
# This is a real 30-minute ECG from a human patient.
# wfdb downloads it automatically on first run (~1MB).
print("Downloading ECG record 100 from PhysioNet...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

# ── 2. Extract signal ─────────────────────────────────────────────────────────
fs = record.fs                    # sampling frequency = 360 Hz
signal = record.p_signal[:, 0]   # channel 0 (MLII lead)
time = np.arange(len(signal)) / fs  # time axis in seconds

# Take first 10 seconds for display
duration = 10  # seconds
n_samples = duration * fs
sig_10s = signal[:n_samples]
time_10s = time[:n_samples]

# ── 3. Basic stats ────────────────────────────────────────────────────────────
print(f"\nRecord info:")
print(f"  Sampling frequency : {fs} Hz")
print(f"  Total duration     : {len(signal)/fs:.1f} seconds")
print(f"  Signal range       : {signal.min():.3f} to {signal.max():.3f} mV")
print(f"  Total samples      : {len(signal)}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
fig.suptitle('Day 1 — Raw ECG Signal (MIT-BIH Record 100)', fontsize=14, fontweight='bold')

# Full 10 seconds
axes[0].plot(time_10s, sig_10s, color='#1D9E75', linewidth=0.8, label='Raw ECG')
axes[0].set_title('Full 10 seconds — notice baseline drift and noise')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude (mV)')
axes[0].grid(True, alpha=0.3, color='#cccccc')
axes[0].legend()

# Zoom in on 2 beats (~1.5 seconds)
zoom_start = int(2.0 * fs)
zoom_end   = int(3.5 * fs)
axes[1].plot(time[zoom_start:zoom_end], signal[zoom_start:zoom_end],
             color='#1D9E75', linewidth=1.5, label='Zoomed ECG')
axes[1].set_title('Zoomed — one complete PQRST complex visible')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude (mV)')
axes[1].grid(True, alpha=0.3, color='#cccccc')

# Label PQRST on the zoomed view
for label, t_offset, y_offset in [('P', 2.08, 0.05), ('Q', 2.14, -0.15),
                                    ('R', 2.17, 1.2),  ('S', 2.20, -0.3),
                                    ('T', 2.30, 0.2)]:
    axes[1].annotate(label, xy=(t_offset, y_offset), fontsize=12,
                     fontweight='bold', color='#E24B4A',
                     ha='center', va='bottom')

plt.tight_layout()
plt.savefig('day1_raw_ecg.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: day1_raw_ecg.png")
print("\nWhat you're seeing:")
print("  - Baseline wander: the signal drifts up and down slowly (breathing artifact)")
print("  - High-frequency noise: tiny jagged fluctuations (muscle noise)")
print("  - PQRST complex: the characteristic heartbeat shape")
print("\nDay 1 complete! Tomorrow: filter this signal clean.")

