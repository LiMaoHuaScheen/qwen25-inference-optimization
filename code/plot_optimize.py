# -*- coding: utf-8 -*-
"""Plot inference optimization benchmark results."""
import json, os, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "PROJECT_ROOT/output_optimize/benchmark.json"
OUT = "PROJECT_ROOT/output_optimize"

data = json.load(io.open(SRC, encoding="utf-8"))
arms = [a for a in data["arms"] if "ppl" in a]

names = [a["arm"] for a in arms]
ppls  = [a["ppl"] for a in arms]
vrams = [a["peak_vram_mb"] for a in arms]
tps   = [a["decode_tokens_per_s"] for a in arms]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
colors = ["#4C72B0", "#DD8452"]

# PPL
bars = ax[0].bar(names, ppls, color=colors)
ax[0].set_title("Held-out PPL (lower better)")
ax[0].set_ylabel("PPL")
for b, v in zip(bars, ppls):
    ax[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.2f}", ha="center")

# VRAM
bars = ax[1].bar(names, vrams, color=colors)
ax[1].set_title("Peak VRAM (MB)")
ax[1].set_ylabel("MB")
for b, v in zip(bars, vrams):
    ax[1].text(b.get_x()+b.get_width()/2, b.get_height()+15, f"{v:.0f}", ha="center")

# Decode throughput
bars = ax[2].bar(names, tps, color=colors)
ax[2].set_title("Decode throughput (tok/s)")
ax[2].set_ylabel("tok/s")
for b, v in zip(bars, tps):
    ax[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f"{v:.2f}", ha="center")

plt.suptitle("Qwen2.5-0.5B Inference Optimization (n_eval=100, max_new_tokens=128)", fontsize=13)
plt.tight_layout()
out = os.path.join(OUT, "optimization_bar.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
