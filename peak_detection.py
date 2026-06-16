
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import wfdb

# ── 1. Load and filter ────────────────────────────────────────────────────────
print("Loading and filtering ECG...")
record     = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
fs         = record.fs
raw        = record.p_signal[:, 0]
time       = np.arange(len(raw)) / fs

# Filter (same as Day 2)
sos_hp = sig.butter(2, 0.5,  btype='highpass', fs=fs, output='sos')
sos_lp = sig.butter(4, 40.0, btype='lowpass',  fs=fs, output='sos')
clean  = sig.sosfiltfilt(sos_lp, sig.sosfiltfilt(sos_hp, raw))

# Work on first 30 seconds
n      = int(30 * fs)
ecg    = clean[:n]
t      = time[:n]

# ── 2. Pan-Tompkins algorithm (step by step) ──────────────────────────────────
print("Running Pan-Tompkins algorithm...")

# Step 1: Differentiate — emphasizes steep slopes (the R-peak is the steepest)
diff = np.diff(ecg, prepend=ecg[0])

# Step 2: Square every sample — all values positive, big slopes become huge
squared = diff ** 2

# Step 3: Moving-window integration (150ms window)
# Smooths into one big bump per heartbeat
win_size = int(0.150 * fs)   # 150ms = 54 samples at 360Hz
window   = np.ones(win_size) / win_size
integrated = np.convolve(squared, window, mode='same')

# Step 4: Find peaks in the integrated signal using scipy
# distance: minimum 200ms between beats (max ~300 bpm)
# height: 50% of the signal's max (adaptive threshold seed)
min_dist   = int(0.200 * fs)
threshold  = integrated.max() * 0.15
peak_idx, properties = sig.find_peaks(
    integrated,
    distance=min_dist,
    height=threshold
)

# Step 5: For each detected peak in the integrated signal,
# find the actual R-peak in the original ECG within ±150ms
search_win = int(0.150 * fs)
r_peaks = []
for p in peak_idx:
    start = max(0, p - search_win)
    end   = min(len(ecg), p + search_win)
    r_peaks.append(start + np.argmax(ecg[start:end]))
r_peaks = np.array(r_peaks)

# ── 3. Compare with reference annotations ────────────────────────────────────
# MIT-BIH provides expert-annotated beat locations — ground truth
ref_peaks = annotation.sample[annotation.sample < n]
ref_peaks = ref_peaks[annotation.symbol[:len(ref_peaks)] == 'N']  # normal beats only

# Evaluate: a detection is correct if within 50ms of a reference peak
tolerance = int(0.050 * fs)
tp = sum(any(abs(r - ref) <= tolerance for ref in ref_peaks) for r in r_peaks)
fp = len(r_peaks) - tp
fn = len(ref_peaks) - tp
sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
precision   = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
f1          = 2*tp / (2*tp + fp + fn) * 100 if (2*tp+fp+fn) > 0 else 0

print(f"\nDetection results (first 30 seconds):")
print(f"  Detected beats   : {len(r_peaks)}")
print(f"  Reference beats  : {len(ref_peaks)}")
print(f"  True positives   : {tp}")
print(f"  False positives  : {fp}")
print(f"  False negatives  : {fn}")
print(f"  Sensitivity      : {sensitivity:.1f}%")
print(f"  Precision        : {precision:.1f}%")
print(f"  F1 score         : {f1:.1f}%")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
fig.suptitle('Pan-Tompkins R-peak Detection', fontsize=14, fontweight='bold')

# Show only first 10 seconds for clarity
n10 = int(10 * fs)
r10 = r_peaks[r_peaks < n10]

axes[0].plot(t[:n10], ecg[:n10], color='#534AB7', linewidth=0.9, label='Filtered ECG')
axes[0].scatter(t[r10], ecg[r10], color='#E24B4A', s=50, zorder=5, label=f'R-peaks ({len(r10)} detected)')
for r in r10:
    axes[0].axvline(t[r], color='#E24B4A', alpha=0.2, linewidth=0.8)
axes[0].set_title('Step 0 — Filtered ECG with detected R-peaks')
axes[0].set_ylabel('mV'); axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot(t[:n10], diff[:n10], color='#1D9E75', linewidth=0.8, label='Differentiated')
axes[1].set_title('Step 1 — Differentiated signal (emphasizes slopes)')
axes[1].set_ylabel('mV/sample'); axes[1].grid(True, alpha=0.3); axes[1].legend()

axes[2].plot(t[:n10], squared[:n10], color='#BA7517', linewidth=0.8, label='Squared')
axes[2].set_title('Step 2 — Squared (all positive, slopes amplified)')
axes[2].set_ylabel('mV²'); axes[2].grid(True, alpha=0.3); axes[2].legend()

axes[3].plot(t[:n10], integrated[:n10], color='#D85A30', linewidth=1, label='Integrated')
p10 = peak_idx[peak_idx < n10]
axes[3].scatter(t[p10], integrated[p10], color='#E24B4A', s=50, zorder=5, label='Peaks')
axes[3].axhline(threshold, color='#888780', linestyle='--', linewidth=1, label=f'Threshold')
axes[3].set_title('Step 3 — Moving-window integrated (one bump per beat)')
axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('Amplitude'); axes[3].grid(True, alpha=0.3); axes[3].legend()

plt.tight_layout()
plt.savefig('day3_peak_detection.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: day3_peak_detection.png")

# Save r_peaks for Day 4
np.save('r_peaks.npy', r_peaks)
print("Saved r_peaks.npy for Day 4")
