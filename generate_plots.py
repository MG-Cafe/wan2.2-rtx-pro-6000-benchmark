#!/usr/bin/env python3
"""
Generate benchmark visualization plots for Wan2.2 SGLang G4 GPU Scaling Analysis.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'teal': '#009688',
    'grey': '#9E9E9E',
    'dark_blue': '#1565C0',
    'dark_green': '#2E7D32',
}

# ============================================================
# PLOT 1: Denoising Time per Step - Grouped Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

gpu_configs = ['1 GPU', '4 GPUs', '8 GPUs']
t2v_steps = [37.13, 24.27, 23.05]
i2v_steps = [0, 24.09, 22.89]  # 0 = OOM
t2v_oom = [False, False, False]
i2v_oom = [True, False, False]

x = np.arange(len(gpu_configs))
width = 0.35

bars1 = ax.bar(x - width/2, t2v_steps, width, label='T2V (Text-to-Video)', 
               color=COLORS['blue'], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, [s if s > 0 else 0 for s in i2v_steps], width, 
               label='I2V (Image-to-Video)', color=COLORS['orange'], edgecolor='white', linewidth=0.5)

# Mark OOM for I2V 1-GPU (no timing data available - OOM before denoising)
ax.bar(x[0] + width/2, 2, width, color=COLORS['red'], alpha=0.3, 
       edgecolor=COLORS['red'], linewidth=2, linestyle='--')
ax.text(x[0] + width/2, 5, 'OOM\n(no data)', ha='center', va='center', fontweight='bold', 
        color=COLORS['red'], fontsize=10)

# Add value labels
for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
for i, bar in enumerate(bars2):
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Speedup annotations
ax.annotate('', xy=(1, 24.27), xytext=(2, 23.05),
            arrowprops=dict(arrowstyle='<->', color=COLORS['dark_green'], lw=2))
ax.text(1.5, 25.5, '−5.0%', ha='center', fontsize=10, color=COLORS['dark_green'], fontweight='bold')

ax.set_ylabel('Seconds per Denoising Step', fontsize=13, fontweight='bold')
ax.set_title('Wan2.2-A14B Denoising Time per Step\n(40 steps, NVIDIA RTX PRO 6000)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(gpu_configs, fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 42)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/01_denoising_time_per_step.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 1: Denoising time per step")

# ============================================================
# PLOT 2: Total End-to-End Time - Horizontal Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

labels = ['I2V 8-GPU', 'I2V 4-GPU', 'I2V 1-GPU', '', 'T2V 8-GPU', 'T2V 4-GPU', 'T2V 1-GPU']
times = [947.54, 995.00, 0, 0, 944.01, 993.15, 0]
colors_list = [COLORS['orange'], COLORS['orange'], COLORS['red'], 'white',
               COLORS['blue'], COLORS['blue'], COLORS['red']]
alphas = [1.0, 0.7, 0.4, 0, 1.0, 0.7, 0.4]

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, times, color=colors_list, edgecolor='white', linewidth=0.5)

for i, (bar, alpha) in enumerate(zip(bars, alphas)):
    bar.set_alpha(alpha)

# Add value labels
for i, (v, label) in enumerate(zip(times, labels)):
    if v > 0 and label:
        status = '✅' if v < 1100 else '❌ OOM at decode'
        ax.text(v + 15, i, f'{v:.1f}s {status}', va='center', fontsize=10, fontweight='bold')
    elif label and v == 0:
        ax.text(200, i, '❌ OOM', va='center', fontsize=12, fontweight='bold', color=COLORS['red'])

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Total Generation Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('End-to-End Video Generation Time\n(1280×720, 40 denoising steps; 1-GPU=81fr, 4/8-GPU=93fr)', fontsize=15, fontweight='bold')
ax.set_xlim(0, 1700)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/02_total_generation_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 2: Total generation time")

# ============================================================
# PLOT 3: GPU Scaling Efficiency
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

gpus = [4, 8]
ideal_speedup = [1.0, 2.0]
t2v_speedup = [1.0, 970.96/921.93]
i2v_speedup = [1.0, 963.49/915.78]

# Left: Speedup
ax1.plot(gpus, ideal_speedup, 'k--', linewidth=2, label='Ideal Linear Scaling', marker='s', markersize=8)
ax1.plot(gpus, t2v_speedup, '-o', linewidth=3, label=f'T2V (actual: {t2v_speedup[1]:.2f}x)', 
         color=COLORS['blue'], markersize=10, markeredgecolor='white', markeredgewidth=2)
ax1.plot(gpus, i2v_speedup, '-^', linewidth=3, label=f'I2V (actual: {i2v_speedup[1]:.2f}x)', 
         color=COLORS['orange'], markersize=10, markeredgecolor='white', markeredgewidth=2)

ax1.fill_between(gpus, [1.0, 1.0], ideal_speedup, alpha=0.1, color='green', label='Scaling gap')
ax1.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax1.set_ylabel('Speedup (vs 4-GPU baseline)', fontsize=12, fontweight='bold')
ax1.set_title('GPU Scaling: Speedup', fontsize=14, fontweight='bold')
ax1.set_xticks(gpus)
ax1.set_xlim(3.5, 8.5)
ax1.set_ylim(0.8, 2.2)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: Efficiency
t2v_eff = [(s/i)*100 for s, i in zip(t2v_speedup, ideal_speedup)]
i2v_eff = [(s/i)*100 for s, i in zip(i2v_speedup, ideal_speedup)]

x_eff = np.arange(2)
width_eff = 0.35
bars_eff1 = ax2.bar(x_eff - width_eff/2, t2v_eff, width_eff, label='T2V', color=COLORS['blue'])
bars_eff2 = ax2.bar(x_eff + width_eff/2, i2v_eff, width_eff, label='I2V', color=COLORS['orange'])

for bar in list(bars_eff1) + list(bars_eff2):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_ylabel('Scaling Efficiency (%)', fontsize=12, fontweight='bold')
ax2.set_title('GPU Scaling: Efficiency', fontsize=14, fontweight='bold')
ax2.set_xticks(x_eff)
ax2.set_xticklabels(['4 GPUs\n(baseline)', '8 GPUs'], fontsize=11)
ax2.set_ylim(0, 115)
ax2.axhline(y=100, color='grey', linestyle='--', alpha=0.5)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/03_gpu_scaling_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 3: GPU scaling efficiency")

# ============================================================
# PLOT 4: Pipeline Stage Breakdown - Donut Charts
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# T2V Pipeline
t2v_stages = ['Denoising\n970.96s', 'Decoding\n18.36s', 'TextEncoding\n1.75s', 'Other\n2.08s']
t2v_times = [970.96, 18.36, 1.75, 2.08]
t2v_colors = [COLORS['blue'], COLORS['teal'], COLORS['purple'], COLORS['grey']]

wedges1, texts1, autotexts1 = ax1.pie(t2v_times, labels=t2v_stages, autopct='%1.1f%%',
                                       colors=t2v_colors, pctdistance=0.75,
                                       wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                                       textprops={'fontsize': 9})
ax1.set_title('T2V Pipeline (4 GPUs)\nTotal: 993.15s', fontsize=13, fontweight='bold')

# I2V Pipeline
i2v_stages = ['Denoising\n963.49s', 'Decoding\n18.13s', 'ImgVAE\n9.59s', 'TextEnc\n1.72s', 'Other\n2.07s']
i2v_times = [963.49, 18.13, 9.59, 1.72, 2.07]
i2v_colors = [COLORS['orange'], COLORS['teal'], COLORS['green'], COLORS['purple'], COLORS['grey']]

wedges2, texts2, autotexts2 = ax2.pie(i2v_times, labels=i2v_stages, autopct='%1.1f%%',
                                       colors=i2v_colors, pctdistance=0.75,
                                       wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                                       textprops={'fontsize': 9})
ax2.set_title('I2V Pipeline (4 GPUs)\nTotal: 995.00s', fontsize=13, fontweight='bold')

for autotexts in [autotexts1, autotexts2]:
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')

plt.suptitle('Pipeline Stage Breakdown', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/04_pipeline_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 4: Pipeline breakdown")

# ============================================================
# PLOT 5: Cost-Performance Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

configs = ['4 GPUs', '8 GPUs']
gpu_hours = [1.10, 2.10]
speedup = [1.0, 1.05]
cost_eff = [1.0, 0.55]

x = np.arange(len(configs))
width = 0.25

bars1 = ax.bar(x - width, [g/1.10 for g in gpu_hours], width, label='Relative Cost', 
               color=COLORS['red'], alpha=0.7)
bars2 = ax.bar(x, speedup, width, label='Speedup', color=COLORS['green'], alpha=0.7)
bars3 = ax.bar(x + width, cost_eff, width, label='Cost Efficiency', color=COLORS['blue'], alpha=0.7)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Ratio (vs 4-GPU baseline)', fontsize=12, fontweight='bold')
ax.set_title('Cost-Performance Analysis\n(Averaged across T2V and I2V)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=12)
ax.set_ylim(0, 2.3)
ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/05_cost_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 5: Cost-performance analysis")

# ============================================================
# PLOT 6: Memory Usage Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))

gpu_configs = ['1 GPU', '4 GPUs (TP=4)', '8 GPUs (TP=8)']
vram_per_gpu = [95, 24, 12]
total_vram = [95, 96, 96]
max_vram = 96

y_pos = np.arange(len(gpu_configs))
bars = ax.barh(y_pos, vram_per_gpu, color=[COLORS['red'], COLORS['blue'], COLORS['green']], 
               edgecolor='white', linewidth=0.5, height=0.5)

# Add capacity line
ax.axvline(x=95, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.5, label='GPU Capacity (~95GB usable)')

for i, (bar, v) in enumerate(zip(bars, vram_per_gpu)):
    status = '❌ OOM!' if v >= 95 else f'✅ {v}GB/GPU'
    ax.text(v + 1, i, status, va='center', fontsize=11, fontweight='bold',
            color=COLORS['red'] if v >= 95 else COLORS['dark_green'])

ax.set_yticks(y_pos)
ax.set_yticklabels(gpu_configs, fontsize=12)
ax.set_xlabel('VRAM per GPU (GB)', fontsize=12, fontweight='bold')
ax.set_title('GPU Memory Usage per Device\n(RTX PRO 6000: 96GB total, ~95GB usable)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 110)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/06_memory_usage.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 6: Memory usage analysis")

print("\n🎉 All plots generated in plots/ directory!")
