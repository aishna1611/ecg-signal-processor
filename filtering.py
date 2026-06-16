
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import wfdb

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading ECG record...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
fs     = record.fs
raw    = record.p_signal[:, 0]
time   = np.arange(len(raw)) / fs

# Work on first 10 seconds
n = int(10 * fs)
raw_10s  = raw[:n]
time_10s = time[:n]

# ── 2. Design filters ─────────────────────────────────────────────────────────
# Highpass: removes baseline wander (slow drift from breathing)
# Cutoff 0.5Hz — anything slower than 0.5 cycles/sec is drift, not heart signal
sos_hp = sig.butter(2, 0.5, btype='highpass', fs=fs, output='sos')

# Lowpass: removes high-frequency noise (muscle EMG, power line)
# Cutoff 40Hz — real ECG energy is below 40Hz
sos_lp = sig.butter(4, 40.0, btype='lowpass', fs=fs, output='sos')

# ── 3. Apply filters (zero-phase with sosfiltfilt) ────────────────────────────
# sosfiltfilt applies the filter forward AND backward — zero phase shift.
# This means R-peaks don't shift in time, which is critical for timing analysis.
hp_signal = sig.sosfiltfilt(sos_hp, raw_10s)   # remove baseline wander
clean     = sig.sosfiltfilt(sos_lp, hp_signal)  # remove high-freq noise

print(f"Filtering complete.")
print(f"  Raw signal std   : {raw_10s.std():.4f} mV")
print(f"  Clean signal std : {clean.std():.4f} mV")
print(f"  Noise reduction  : {(1 - clean.std()/raw_10s.std())*100:.1f}%")

# ── 4. Frequency response plots ───────────────────────────────────────────────
w_hp, h_hp = sig.sosfreqz(sos_hp, worN=2000, fs=fs)
w_lp, h_lp = sig.sosfreqz(sos_lp, worN=2000, fs=fs)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle('Digital Filter Design & Application', fontsize=14, fontweight='bold')

# Row 0: Raw vs filtered signal
axes[0,0].plot(time_10s, raw_10s, color='#E24B4A', linewidth=0.7, alpha=0.9, label='Raw')
axes[0,0].set_title('Raw ECG — baseline wander visible')
axes[0,0].set_xlabel('Time (s)'); axes[0,0].set_ylabel('mV')
axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

axes[0,1].plot(time_10s, clean, color='#534AB7', linewidth=0.9, label='Filtered')
axes[0,1].set_title('Filtered ECG — clean PQRST complexes')
axes[0,1].set_xlabel('Time (s)'); axes[0,1].set_ylabel('mV')
axes[0,1].grid(True, alpha=0.3); axes[0,1].legend()

# Row 1: Overlay comparison on 2 seconds
t_start = int(1.0 * fs); t_end = int(3.0 * fs)
axes[1,0].plot(time[t_start:t_end], raw_10s[t_start:t_end],
               color='#E24B4A', linewidth=1.2, alpha=0.7, label='Raw')
axes[1,0].plot(time[t_start:t_end], clean[t_start:t_end],
               color='#534AB7', linewidth=1.5, label='Filtered')
axes[1,0].set_title('Overlay — 2 seconds zoomed')
axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('mV')
axes[1,0].grid(True, alpha=0.3); axes[1,0].legend()

# Row 1: Noise signal = raw - filtered
noise_signal = raw_10s - clean
axes[1,1].plot(time_10s, noise_signal, color='#888780', linewidth=0.7, label='Noise removed')
axes[1,1].set_title('What was removed (noise + drift)')
axes[1,1].set_xlabel('Time (s)'); axes[1,1].set_ylabel('mV')
axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()

# Row 2: Frequency responses
axes[2,0].plot(w_hp, 20*np.log10(np.abs(h_hp)+1e-12), color='#1D9E75', linewidth=2)
axes[2,0].axvline(0.5, color='#E24B4A', linestyle='--', label='Cutoff 0.5Hz')
axes[2,0].set_title('Highpass filter frequency response')
axes[2,0].set_xlabel('Frequency (Hz)'); axes[2,0].set_ylabel('Gain (dB)')
axes[2,0].set_xlim(0, 5); axes[2,0].set_ylim(-60, 5)
axes[2,0].grid(True, alpha=0.3); axes[2,0].legend()

axes[2,1].plot(w_lp, 20*np.log10(np.abs(h_lp)+1e-12), color='#534AB7', linewidth=2)
axes[2,1].axvline(40, color='#E24B4A', linestyle='--', label='Cutoff 40Hz')
axes[2,1].set_title('Lowpass filter frequency response')
axes[2,1].set_xlabel('Frequency (Hz)'); axes[2,1].set_ylabel('Gain (dB)')
axes[2,1].set_xlim(0, 180); axes[2,1].set_ylim(-60, 5)
axes[2,1].grid(True, alpha=0.3); axes[2,1].legend()

plt.tight_layout()
plt.savefig('day2_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: day2_filtering.png")
print("\nKey concepts learned today:")
print("  - Butterworth filter: maximally flat passband — no ripple in the signal you keep")
print("  - sosfiltfilt: zero-phase — applies filter twice (forward+backward) so peaks don't shift")
print("  - SOS (second-order sections): numerically stable for high-order filters")
