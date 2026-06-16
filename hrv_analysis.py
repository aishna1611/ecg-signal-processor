"""
ECG Signal Processor — Day 4
Goal: Compute heart rate variability (HRV) metrics from R-peaks.
Run: python day4_hrv_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import wfdb
import os

# ── 1. Load ECG and get R-peaks ───────────────────────────────────────────────
print("Loading ECG and detecting R-peaks...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
fs     = record.fs
raw    = record.p_signal[:, 0]

# Filter
sos_hp = sig.butter(2, 0.5,  btype='highpass', fs=fs, output='sos')
sos_lp = sig.butter(4, 40.0, btype='lowpass',  fs=fs, output='sos')
clean  = sig.sosfiltfilt(sos_lp, sig.sosfiltfilt(sos_hp, raw))

# Load or recompute R-peaks
if os.path.exists('r_peaks.npy'):
    r_peaks = np.load('r_peaks.npy')
    print(f"Loaded {len(r_peaks)} R-peaks from day3")
else:
    integrated = np.convolve(
        sig.sosfiltfilt(sig.butter(4,40,'low',fs=fs,output='sos'), np.diff(clean, prepend=clean[0])**2),
        np.ones(int(0.15*fs))/int(0.15*fs), mode='same')
    peak_idx, _ = sig.find_peaks(integrated, distance=int(0.2*fs), height=integrated.max()*0.15)
    r_peaks = np.array([p-int(0.15*fs)+np.argmax(clean[max(0,p-int(0.15*fs)):min(len(clean),p+int(0.15*fs))]) for p in peak_idx])
    print(f"Recomputed {len(r_peaks)} R-peaks")

# ── 2. Compute RR intervals ───────────────────────────────────────────────────
rr_intervals = np.diff(r_peaks) / fs * 1000   # convert to milliseconds
rr_time      = r_peaks[1:] / fs               # time of each RR interval

# Remove physiologically impossible beats (outside 300-2000ms = 30-200bpm)
valid = (rr_intervals > 300) & (rr_intervals < 2000)
rr_intervals = rr_intervals[valid]
rr_time      = rr_time[valid]

print(f"\nRR interval statistics:")
print(f"  Count            : {len(rr_intervals)}")
print(f"  Mean RR          : {rr_intervals.mean():.1f} ms")
print(f"  Mean HR          : {60000/rr_intervals.mean():.1f} bpm")
print(f"  Min RR           : {rr_intervals.min():.1f} ms")
print(f"  Max RR           : {rr_intervals.max():.1f} ms")

# ── 3. HRV metrics ────────────────────────────────────────────────────────────
# Time-domain metrics (most commonly used clinically)
sdnn   = rr_intervals.std()                              # std of all RR intervals
rmssd  = np.sqrt(np.mean(np.diff(rr_intervals)**2))     # root mean square of successive differences
pnn50  = np.mean(np.abs(np.diff(rr_intervals)) > 50)*100 # % of successive differences > 50ms
mean_hr= 60000 / rr_intervals.mean()

print(f"\nHRV Metrics (clinical gold standard):")
print(f"  SDNN             : {sdnn:.2f} ms   (normal: 20-50ms)")
print(f"  RMSSD            : {rmssd:.2f} ms   (normal: 20-40ms)")
print(f"  pNN50            : {pnn50:.1f}%    (normal: >3%)")
print(f"  Mean HR          : {mean_hr:.1f} bpm")

# Frequency-domain HRV (basic)
# Interpolate RR to uniform grid for FFT
from scipy.interpolate import interp1d
rr_interp_fn = interp1d(rr_time, rr_intervals, kind='cubic',
                          fill_value='extrapolate')
t_uniform  = np.arange(rr_time[0], rr_time[-1], 1/4.0)  # 4Hz resampling
rr_uniform = rr_interp_fn(t_uniform)

freqs, psd = sig.welch(rr_uniform, fs=4.0, nperseg=min(256, len(rr_uniform)//2))
lf_band = (freqs >= 0.04) & (freqs <= 0.15)  # Low frequency: 0.04-0.15Hz
hf_band = (freqs >= 0.15) & (freqs <= 0.40)  # High frequency: 0.15-0.40Hz
lf_power = np.trapezoid(psd[lf_band], freqs[lf_band])
hf_power = np.trapezoid(psd[hf_band], freqs[hf_band])
lf_hf    = lf_power / hf_power if hf_power > 0 else 0

print(f"\nFrequency-domain HRV:")
print(f"  LF power         : {lf_power:.2f} ms²  (sympathetic + parasympathetic)")
print(f"  HF power         : {hf_power:.2f} ms²  (parasympathetic/vagal)")
print(f"  LF/HF ratio      : {lf_hf:.2f}   (autonomic balance)")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Day 4 — Heart Rate Variability Analysis', fontsize=14, fontweight='bold')

# Tachogram
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(rr_time, rr_intervals, color='#534AB7', linewidth=1.2, label='RR intervals')
ax1.fill_between(rr_time, rr_intervals, rr_intervals.mean(),
                 alpha=0.2, color='#534AB7')
ax1.axhline(rr_intervals.mean(), color='#E24B4A', linestyle='--',
            linewidth=1.5, label=f'Mean: {rr_intervals.mean():.1f} ms')
ax1.set_title('RR Interval Tachogram')
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('RR Interval (ms)')
ax1.grid(True, alpha=0.3); ax1.legend()

# Instantaneous heart rate
ax2 = fig.add_subplot(3, 2, 3)
inst_hr = 60000 / rr_intervals
ax2.plot(rr_time, inst_hr, color='#E24B4A', linewidth=1)
ax2.set_title('Instantaneous Heart Rate')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('HR (bpm)')
ax2.grid(True, alpha=0.3)

# RR interval histogram
ax3 = fig.add_subplot(3, 2, 4)
ax3.hist(rr_intervals, bins=30, color='#1D9E75', edgecolor='white',
         linewidth=0.5, alpha=0.85)
ax3.axvline(rr_intervals.mean(), color='#E24B4A', linestyle='--',
            linewidth=2, label=f'Mean={rr_intervals.mean():.0f}ms')
ax3.axvline(rr_intervals.mean()-sdnn, color='#BA7517', linestyle=':',
            linewidth=1.5, label=f'±1 SDNN ({sdnn:.0f}ms)')
ax3.axvline(rr_intervals.mean()+sdnn, color='#BA7517', linestyle=':', linewidth=1.5)
ax3.set_title('RR Interval Distribution')
ax3.set_xlabel('RR Interval (ms)'); ax3.set_ylabel('Count')
ax3.grid(True, alpha=0.3); ax3.legend(fontsize=9)

# Poincaré plot (RR[n] vs RR[n+1])
ax4 = fig.add_subplot(3, 2, 5)
ax4.scatter(rr_intervals[:-1], rr_intervals[1:], alpha=0.4,
            s=8, color='#534AB7')
ax4.plot([rr_intervals.min(), rr_intervals.max()],
         [rr_intervals.min(), rr_intervals.max()],
         'r--', linewidth=1, label='Line of identity')
ax4.set_title('Poincaré Plot (RR[n] vs RR[n+1])')
ax4.set_xlabel('RR[n] (ms)'); ax4.set_ylabel('RR[n+1] (ms)')
ax4.grid(True, alpha=0.3); ax4.legend(fontsize=9)
ax4.set_aspect('equal')

# Power spectral density
ax5 = fig.add_subplot(3, 2, 6)
ax5.semilogy(freqs, psd, color='#D85A30', linewidth=1.2)
ax5.fill_between(freqs[lf_band], psd[lf_band], alpha=0.4,
                 color='#BA7517', label=f'LF ({lf_power:.1f} ms²)')
ax5.fill_between(freqs[hf_band], psd[hf_band], alpha=0.4,
                 color='#1D9E75', label=f'HF ({hf_power:.1f} ms²)')
ax5.set_title('HRV Power Spectral Density')
ax5.set_xlabel('Frequency (Hz)'); ax5.set_ylabel('PSD (ms²/Hz)')
ax5.set_xlim(0, 0.5); ax5.grid(True, alpha=0.3); ax5.legend(fontsize=9)

# Metrics text box
metrics_text = (
    f"HRV Metrics Summary\n"
    f"{'─'*22}\n"
    f"Mean HR : {mean_hr:.1f} bpm\n"
    f"SDNN    : {sdnn:.1f} ms\n"
    f"RMSSD   : {rmssd:.1f} ms\n"
    f"pNN50   : {pnn50:.1f}%\n"
    f"LF/HF   : {lf_hf:.2f}"
)
fig.text(0.01, 0.02, metrics_text, fontsize=9, fontfamily='monospace',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('day4_hrv_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: day4_hrv_analysis.png")

# Save metrics for Day 5
np.save('hrv_metrics.npy', {
    'sdnn': sdnn, 'rmssd': rmssd, 'pnn50': pnn50,
    'mean_hr': mean_hr, 'lf_hf': lf_hf,
    'rr_intervals': rr_intervals, 'rr_time': rr_time
})
print("Saved hrv_metrics.npy for Day 5")
