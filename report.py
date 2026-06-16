"""
ECG Signal Processor — Day 5
Goal: Generate a complete clinical PDF report — the final deliverable.
Run: python day5_report.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import scipy.signal as sig
import wfdb
import os
from datetime import datetime

# ── 1. Full pipeline (all 4 days combined) ────────────────────────────────────
print("Running full ECG analysis pipeline...")

record     = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
fs         = record.fs
raw        = record.p_signal[:, 0]
time       = np.arange(len(raw)) / fs

# Filter
sos_hp = sig.butter(2, 0.5,  btype='highpass', fs=fs, output='sos')
sos_lp = sig.butter(4, 40.0, btype='lowpass',  fs=fs, output='sos')
clean  = sig.sosfiltfilt(sos_lp, sig.sosfiltfilt(sos_hp, raw))

# Pan-Tompkins R-peak detection
diff_sig    = np.diff(clean, prepend=clean[0])
squared     = diff_sig ** 2
win_size    = int(0.150 * fs)
integrated  = np.convolve(squared, np.ones(win_size)/win_size, mode='same')
peak_idx, _ = sig.find_peaks(integrated, distance=int(0.2*fs), height=integrated.max()*0.15)
search_win  = int(0.150 * fs)
r_peaks     = np.array([max(0,p-search_win) + np.argmax(clean[max(0,p-search_win):min(len(clean),p+search_win)]) for p in peak_idx])

# RR intervals and HRV
rr_ms  = np.diff(r_peaks) / fs * 1000
rr_t   = r_peaks[1:] / fs
valid  = (rr_ms > 300) & (rr_ms < 2000)
rr_ms  = rr_ms[valid]; rr_t = rr_t[valid]

sdnn   = rr_ms.std()
rmssd  = np.sqrt(np.mean(np.diff(rr_ms)**2))
pnn50  = np.mean(np.abs(np.diff(rr_ms)) > 50) * 100
mean_hr= 60000 / rr_ms.mean()
inst_hr= 60000 / rr_ms

# Frequency domain
from scipy.interpolate import interp1d
rr_fn      = interp1d(rr_t, rr_ms, kind='cubic', fill_value='extrapolate')
t_uni      = np.arange(rr_t[0], rr_t[-1], 0.25)
rr_uni     = rr_fn(t_uni)
freqs, psd = sig.welch(rr_uni, fs=4.0, nperseg=min(256, len(rr_uni)//2))
lf_b = (freqs >= 0.04) & (freqs <= 0.15)
hf_b = (freqs >= 0.15) & (freqs <= 0.40)
lf_p = np.trapezoid(psd[lf_b], freqs[lf_b])
hf_p = np.trapezoid(psd[hf_b], freqs[hf_b])
lf_hf= lf_p / hf_p if hf_p > 0 else 0

print(f"Pipeline complete. Building PDF report...")

# ── 2. Interpret metrics ──────────────────────────────────────────────────────
def interpret(metric, value):
    ranges = {
        'sdnn':  [(0,20,'Low — possible autonomic dysfunction'),
                  (20,50,'Normal range'),
                  (50,999,'High — excellent HRV')],
        'rmssd': [(0,20,'Low — reduced vagal tone'),
                  (20,40,'Normal range'),
                  (40,999,'High — strong parasympathetic activity')],
        'hr':    [(0,60,'Bradycardia — below normal'),
                  (60,100,'Normal sinus rhythm'),
                  (100,999,'Tachycardia — above normal')],
        'pnn50': [(0,3,'Low — reduced HRV'),
                  (3,999,'Within normal limits')],
    }
    for lo, hi, label in ranges.get(metric, []):
        if lo <= value < hi:
            return label
    return 'N/A'

# ── 3. Build the PDF report ───────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 17))  # A4 portrait
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(5, 2, figure=fig,
                        hspace=0.45, wspace=0.3,
                        left=0.08, right=0.95,
                        top=0.93, bottom=0.04)

# ── Header ────────────────────────────────────────────────────────────────────
header_ax = fig.add_axes([0, 0.945, 1, 0.055])
header_ax.set_facecolor('#534AB7')
header_ax.axis('off')
header_ax.text(0.02, 0.55, 'ECG Signal Processor — Clinical Report',
               color='white', fontsize=16, fontweight='bold',
               transform=header_ax.transAxes, va='center')
header_ax.text(0.98, 0.55,
               f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               color='#CECBF6', fontsize=9,
               transform=header_ax.transAxes, va='center', ha='right')

# ── Patient info bar ──────────────────────────────────────────────────────────
info_ax = fig.add_axes([0, 0.91, 1, 0.035])
info_ax.set_facecolor('#EEEDFE')
info_ax.axis('off')
info_str = (f"Record: MIT-BIH 100  |  Lead: MLII  |  "
            f"Fs: {fs} Hz  |  Duration: {len(raw)/fs:.0f}s  |  "
            f"Beats detected: {len(r_peaks)}")
info_ax.text(0.5, 0.5, info_str, color='#3C3489', fontsize=9,
             transform=info_ax.transAxes, va='center', ha='center')

# ── Panel 1: Raw ECG (10s) ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
n10 = int(10 * fs)
ax1.plot(time[:n10], raw[:n10], color='#E24B4A', linewidth=0.6, alpha=0.8, label='Raw')
ax1.plot(time[:n10], clean[:n10], color='#534AB7', linewidth=0.9, label='Filtered')
r10 = r_peaks[r_peaks < n10]
ax1.scatter(time[r10], clean[r10], color='#E24B4A', s=30, zorder=5, label=f'{len(r10)} R-peaks')
ax1.set_title('ECG Signal — Raw vs Filtered (10 seconds)', fontweight='bold', fontsize=10)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude (mV)')
ax1.grid(True, alpha=0.25); ax1.legend(fontsize=8)

# ── Panel 2: Tachogram ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(rr_t, rr_ms, color='#534AB7', linewidth=0.9)
ax2.fill_between(rr_t, rr_ms, rr_ms.mean(), alpha=0.15, color='#534AB7')
ax2.axhline(rr_ms.mean(), color='#E24B4A', linestyle='--', linewidth=1.2,
            label=f'Mean = {rr_ms.mean():.1f} ms')
ax2.set_title('RR Interval Tachogram', fontweight='bold', fontsize=10)
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('RR Interval (ms)')
ax2.grid(True, alpha=0.25); ax2.legend(fontsize=8)

# ── Panel 3: HR over time ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(rr_t, inst_hr, color='#E24B4A', linewidth=0.9)
ax3.axhline(mean_hr, linestyle='--', color='#534AB7', linewidth=1,
            label=f'Mean {mean_hr:.1f} bpm')
ax3.set_title('Heart Rate Over Time', fontweight='bold', fontsize=10)
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('HR (bpm)')
ax3.grid(True, alpha=0.25); ax3.legend(fontsize=8)

# ── Panel 4: RR histogram ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(rr_ms, bins=25, color='#1D9E75', edgecolor='white', linewidth=0.5, alpha=0.85)
ax4.axvline(rr_ms.mean(), color='#E24B4A', linestyle='--', linewidth=1.5,
            label=f'Mean={rr_ms.mean():.0f}ms')
ax4.axvspan(rr_ms.mean()-sdnn, rr_ms.mean()+sdnn, alpha=0.15,
            color='#BA7517', label=f'±SDNN ({sdnn:.0f}ms)')
ax4.set_title('RR Interval Distribution', fontweight='bold', fontsize=10)
ax4.set_xlabel('RR Interval (ms)'); ax4.set_ylabel('Count')
ax4.grid(True, alpha=0.25); ax4.legend(fontsize=8)

# ── Panel 5: Poincaré plot ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
ax5.scatter(rr_ms[:-1], rr_ms[1:], alpha=0.35, s=6, color='#534AB7')
ax5.plot([rr_ms.min(), rr_ms.max()], [rr_ms.min(), rr_ms.max()],
         'r--', linewidth=1, label='Identity line')
ax5.set_title('Poincaré Plot', fontweight='bold', fontsize=10)
ax5.set_xlabel('RR[n] (ms)'); ax5.set_ylabel('RR[n+1] (ms)')
ax5.grid(True, alpha=0.25); ax5.legend(fontsize=8)
ax5.set_aspect('equal')

# ── Panel 6: PSD ─────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
ax6.semilogy(freqs, psd, color='#D85A30', linewidth=1)
ax6.fill_between(freqs[lf_b], psd[lf_b], alpha=0.4,
                 color='#BA7517', label=f'LF={lf_p:.1f}ms²')
ax6.fill_between(freqs[hf_b], psd[hf_b], alpha=0.4,
                 color='#1D9E75', label=f'HF={hf_p:.1f}ms²')
ax6.set_title('HRV Power Spectral Density', fontweight='bold', fontsize=10)
ax6.set_xlabel('Frequency (Hz)'); ax6.set_ylabel('PSD (ms²/Hz)')
ax6.set_xlim(0, 0.5); ax6.grid(True, alpha=0.25); ax6.legend(fontsize=8)

# ── Panel 7: Metrics summary table ───────────────────────────────────────────
ax7 = fig.add_subplot(gs[4, :])
ax7.axis('off')

metrics = [
    ('Mean Heart Rate', f'{mean_hr:.1f} bpm',     interpret('hr',    mean_hr), '60–100 bpm'),
    ('SDNN',           f'{sdnn:.1f} ms',           interpret('sdnn',  sdnn),    '20–50 ms'),
    ('RMSSD',          f'{rmssd:.1f} ms',          interpret('rmssd', rmssd),   '20–40 ms'),
    ('pNN50',          f'{pnn50:.1f}%',            interpret('pnn50', pnn50),   '>3%'),
    ('LF/HF Ratio',    f'{lf_hf:.2f}',            'Autonomic balance',          '1.0–2.0'),
    ('Total Beats',    f'{len(r_peaks)}',          'Detected by Pan-Tompkins',  '—'),
]

col_labels = ['Metric', 'Value', 'Interpretation', 'Normal Range']
col_widths = [0.18, 0.12, 0.45, 0.20]
row_h = 0.14
header_y = 0.92

# Table header
for ci, (label, width) in enumerate(zip(col_labels, col_widths)):
    x = sum(col_widths[:ci])
    ax7.add_patch(FancyBboxPatch((x+0.005, header_y), width-0.01, row_h-0.02,
                                  boxstyle='round,pad=0.01',
                                  facecolor='#534AB7', transform=ax7.transAxes,
                                  clip_on=False))
    ax7.text(x + width/2, header_y + (row_h-0.02)/2, label,
             ha='center', va='center', color='white', fontsize=9, fontweight='bold',
             transform=ax7.transAxes)

# Table rows
for ri, (metric, value, interp, normal) in enumerate(metrics):
    row_y = header_y - (ri+1) * row_h
    bg = '#EEEDFE' if ri % 2 == 0 else 'white'
    for ci, (cell, width) in enumerate(zip([metric, value, interp, normal], col_widths)):
        x = sum(col_widths[:ci])
        ax7.add_patch(FancyBboxPatch((x+0.005, row_y), width-0.01, row_h-0.02,
                                      boxstyle='round,pad=0.01',
                                      facecolor=bg, edgecolor='#CECBF6',
                                      linewidth=0.5, transform=ax7.transAxes, clip_on=False))
        color = '#534AB7' if ci == 1 else '#3d3d3a'
        ax7.text(x + width/2, row_y + (row_h-0.02)/2, cell,
                 ha='center', va='center', fontsize=8.5, color=color,
                 transform=ax7.transAxes)

ax7.set_title('Clinical Metrics Summary', fontweight='bold', fontsize=10, pad=8)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         'ECG Signal Processor · Python · NumPy · SciPy · MIT-BIH PhysioNet Database · '
         'Pan-Tompkins Algorithm (1985)',
         ha='center', fontsize=7, color='#888780')

plt.savefig('ecg_clinical_report.pdf', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.savefig('ecg_clinical_report.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()

print("\n" + "="*55)
print("  ECG Signal Processor — COMPLETE")
print("="*55)
print(f"\n  Saved: ecg_clinical_report.pdf")
print(f"  Saved: ecg_clinical_report.png")
print(f"\n  Final metrics:")
print(f"    Mean HR  : {mean_hr:.1f} bpm")
print(f"    SDNN     : {sdnn:.1f} ms")
print(f"    RMSSD    : {rmssd:.1f} ms")
print(f"    pNN50    : {pnn50:.1f}%")
print(f"    LF/HF    : {lf_hf:.2f}")
print(f"\n  Resume line:")
print(f"  'Implemented Pan-Tompkins R-peak detection on MIT-BIH")
print(f"   PhysioNet ECG data. Computed HRV metrics (SDNN={sdnn:.0f}ms,")
print(f"   RMSSD={rmssd:.0f}ms) and generated automated clinical PDF reports.'")
print("="*55)