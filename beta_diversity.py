"""
================================================================
Day 02 — Beta Diversity & PCoA Visualization (REAL DATA)
Author  : Subhadip Jana
Dataset : peerj32 — LGG Probiotic vs Placebo intervention
          44 samples × 130 real gut taxa
          Source: microbiome R package (Lahti et al.)

Research Questions:
  1. Do LGG and Placebo groups have different microbiome
     compositions? (Bray-Curtis PERMANOVA)
  2. Does microbiome composition change before vs after
     intervention? (Time 1 vs Time 2)
  3. Which taxa drive group separation?
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.spatial.distance import braycurtis, jaccard
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────

print("🔬 Loading peerj32 dataset...")
otu_raw = pd.read_csv("data/otu_table.csv", index_col=0)
meta    = pd.read_csv("data/metadata.csv",  index_col=0)

otu_df  = otu_raw.T.astype(float)
taxa    = otu_df.columns.tolist()

# Relative abundance
rel_df = otu_df.div(otu_df.sum(axis=1), axis=0)

group      = meta["group"]
time       = meta["time"].astype(str)
group_time = (group + "_T" + time).rename("group_time")

print(f"✅ {len(otu_df)} samples × {len(taxa)} taxa")
print(f"   LGG: {sum(group=='LGG')} | Placebo: {sum(group=='Placebo')}")
print(f"   Time 1: {sum(time=='1')} | Time 2: {sum(time=='2')}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: DISTANCE MATRICES
# ─────────────────────────────────────────────────────────────

print("\n📐 Computing distance matrices...")

def pairwise_dm(df, metric_fn):
    arr = df.values; n = arr.shape[0]
    dm  = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = metric_fn(arr[i], arr[j])
            dm[i,j] = dm[j,i] = d
    return pd.DataFrame(dm, index=df.index, columns=df.index)

bc_dm  = pairwise_dm(rel_df, braycurtis)
# Use small epsilon to avoid zero vectors in Jaccard
jac_dm = pairwise_dm((rel_df > rel_df.quantile(0.1)).astype(float), jaccard)

bc_dm.to_csv("outputs/bray_curtis_matrix.csv")
print("✅ Distance matrices computed")

# ─────────────────────────────────────────────────────────────
# SECTION 3: PCoA
# ─────────────────────────────────────────────────────────────

def pcoa(dist_matrix):
    D = dist_matrix.values.copy(); n = D.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * H @ (D**2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:,idx]
    pos     = eigvals > 0
    eigvals = eigvals[pos]; eigvecs = eigvecs[:,pos]
    coords  = eigvecs * np.sqrt(eigvals)
    var_exp = eigvals / eigvals.sum() * 100
    ncols   = min(10, coords.shape[1])
    return pd.DataFrame(coords[:,:ncols], index=dist_matrix.index,
                        columns=[f"PC{i+1}" for i in range(ncols)]), var_exp

print("\n🔭 Running PCoA...")
bc_pcoa,  bc_var  = pcoa(bc_dm)
jac_pcoa, jac_var = pcoa(jac_dm)

bc_pcoa["Group"]      = group.values
bc_pcoa["Time"]       = time.values
bc_pcoa["Group_Time"] = group_time.values
jac_pcoa["Group"]     = group.values
jac_pcoa["Time"]      = time.values

bc_pcoa.to_csv("outputs/pcoa_braycurtis.csv")
print(f"✅ BC  PCoA: PC1={bc_var[0]:.1f}%  PC2={bc_var[1]:.1f}%")
print(f"✅ Jac PCoA: PC1={jac_var[0]:.1f}%  PC2={jac_var[1]:.1f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 4: PERMANOVA
# ─────────────────────────────────────────────────────────────

def permanova(dist_matrix, labels, n_perm=999):
    dm = dist_matrix.values; n = dm.shape[0]
    labels = np.array(labels)

    def f_stat(dm, lbl):
        n   = len(lbl); uniq = np.unique(lbl)
        sst = np.sum(dm**2) / n
        ssw = sum(np.sum(dm[np.ix_(np.where(lbl==g)[0],
                                    np.where(lbl==g)[0])]**2)
                  / len(np.where(lbl==g)[0]) for g in uniq)
        ssa = sst - ssw; a = len(uniq) - 1
        return (ssa/a) / (ssw/(n-len(uniq)))

    f_obs = f_stat(dm, labels)
    count = sum(1 for _ in range(n_perm)
                if f_stat(dm, np.random.permutation(labels)) >= f_obs)
    return f_obs, (count+1)/(n_perm+1)

print("\n📊 Running PERMANOVA (999 permutations)...")
bc_f_g,  bc_p_g  = permanova(bc_dm, group.values)
bc_f_t,  bc_p_t  = permanova(bc_dm, time.values)
jac_f_g, jac_p_g = permanova(jac_dm, group.values)

print(f"   BC  — LGG vs Placebo : F={bc_f_g:.3f} p={bc_p_g:.3f}")
print(f"   BC  — Time 1 vs 2    : F={bc_f_t:.3f} p={bc_p_t:.3f}")
print(f"   Jac — LGG vs Placebo : F={jac_f_g:.3f} p={jac_p_g:.3f}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: WITHIN vs BETWEEN DISTANCES
# ─────────────────────────────────────────────────────────────

n = len(bc_dm)
g = group.values
within_lgg     = [bc_dm.values[i,j] for i in range(n)
                  for j in range(i+1,n) if g[i]=="LGG"     and g[j]=="LGG"]
within_placebo = [bc_dm.values[i,j] for i in range(n)
                  for j in range(i+1,n) if g[i]=="Placebo" and g[j]=="Placebo"]
between        = [bc_dm.values[i,j] for i in range(n)
                  for j in range(i+1,n) if g[i] != g[j]]

_, p_disp = mannwhitneyu(within_lgg, within_placebo, alternative="two-sided")

print(f"\n   Within LGG mean     : {np.mean(within_lgg):.4f}")
print(f"   Within Placebo mean : {np.mean(within_placebo):.4f}")
print(f"   Between groups mean : {np.mean(between):.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: DASHBOARD
# ─────────────────────────────────────────────────────────────

print("\n🎨 Generating dashboard...")

PAL_GROUP = {"LGG": "#E74C3C", "Placebo": "#3498DB"}
PAL_TIME  = {"1": "#F39C12",   "2": "#27AE60"}
PAL_4G    = {"LGG_T1": "#E74C3C", "LGG_T2": "#C0392B",
             "Placebo_T1": "#3498DB", "Placebo_T2": "#1A5276"}

def draw_ellipse(ax, data, color):
    if len(data) < 3: return
    cov  = np.cov(data.T)
    mean = data.mean(axis=0)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h  = 2 * 2.0 * np.sqrt(np.abs(vals))
    ell   = Ellipse(mean, w, h, angle=angle, color=color, alpha=0.12)
    ax.add_patch(ell)

fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    "Beta Diversity & PCoA — REAL DATA\n"
    "LGG Probiotic vs Placebo | peerj32 dataset\n"
    "44 samples × 130 gut taxa (Lahti et al.)",
    fontsize=15, fontweight="bold", y=0.99
)

# ── Plot 1: PCoA Bray-Curtis — Group ──
ax1 = fig.add_subplot(3, 3, 1)
for g_lbl, color in PAL_GROUP.items():
    sub = bc_pcoa[bc_pcoa["Group"]==g_lbl]
    ax1.scatter(sub["PC1"], sub["PC2"], c=color, label=g_lbl,
                s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
    draw_ellipse(ax1, sub[["PC1","PC2"]].values, color)
ax1.set_xlabel(f"PC1 ({bc_var[0]:.1f}%)")
ax1.set_ylabel(f"PC2 ({bc_var[1]:.1f}%)")
ax1.set_title(f"PCoA — Bray-Curtis (Group)\nPERMANOVA F={bc_f_g:.2f} p={bc_p_g:.3f}",
              fontweight="bold", fontsize=10)
ax1.legend(fontsize=9)
ax1.axhline(0, color="gray", lw=0.5); ax1.axvline(0, color="gray", lw=0.5)

# ── Plot 2: PCoA Bray-Curtis — Time ──
ax2 = fig.add_subplot(3, 3, 2)
for t_lbl, color in PAL_TIME.items():
    sub = bc_pcoa[bc_pcoa["Time"]==t_lbl]
    ax2.scatter(sub["PC1"], sub["PC2"], c=color,
                label=f"Time {t_lbl}", s=80, alpha=0.8,
                edgecolors="white", linewidth=0.5)
    draw_ellipse(ax2, sub[["PC1","PC2"]].values, color)
ax2.set_xlabel(f"PC1 ({bc_var[0]:.1f}%)")
ax2.set_ylabel(f"PC2 ({bc_var[1]:.1f}%)")
ax2.set_title(f"PCoA — Bray-Curtis (Time)\nPERMANOVA F={bc_f_t:.2f} p={bc_p_t:.3f}",
              fontweight="bold", fontsize=10)
ax2.legend(fontsize=9)
ax2.axhline(0, color="gray", lw=0.5); ax2.axvline(0, color="gray", lw=0.5)

# ── Plot 3: PCoA Jaccard ──
ax3 = fig.add_subplot(3, 3, 3)
for g_lbl, color in PAL_GROUP.items():
    sub = jac_pcoa[jac_pcoa["Group"]==g_lbl]
    ax3.scatter(sub["PC1"], sub["PC2"], c=color, label=g_lbl,
                s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
                marker="D" if g_lbl=="LGG" else "o")
    draw_ellipse(ax3, sub[["PC1","PC2"]].values, color)
ax3.set_xlabel(f"PC1 ({jac_var[0]:.1f}%)")
ax3.set_ylabel(f"PC2 ({jac_var[1]:.1f}%)")
ax3.set_title(f"PCoA — Jaccard (Group)\nPERMANOVA F={jac_f_g:.2f} p={jac_p_g:.3f}",
              fontweight="bold", fontsize=10)
ax3.legend(fontsize=9)
ax3.axhline(0, color="gray", lw=0.5); ax3.axvline(0, color="gray", lw=0.5)

# ── Plot 4: Scree plot ──
ax4 = fig.add_subplot(3, 3, 4)
x = np.arange(1, min(11, len(bc_var)+1))
ax4.bar(x, bc_var[:len(x)], color="#9B59B6", edgecolor="black", alpha=0.8)
ax4.plot(x, np.cumsum(bc_var[:len(x)]), "ro-", lw=2, label="Cumulative")
ax4.set_xlabel("Principal Coordinate")
ax4.set_ylabel("% Variance Explained")
ax4.set_title("Scree Plot\n(Bray-Curtis)", fontweight="bold", fontsize=10)
ax4.legend(fontsize=9); ax4.set_xticks(x)

# ── Plot 5: Within vs Between distances ──
ax5 = fig.add_subplot(3, 3, 5)
dist_dict = {"Within\nLGG": within_lgg,
             "Within\nPlacebo": within_placebo,
             "Between\nGroups": between}
colors_v = ["#E74C3C", "#3498DB", "#2ECC71"]
parts = ax5.violinplot(list(dist_dict.values()), positions=[1,2,3],
                       showmeans=True, showmedians=True)
for pc, color in zip(parts["bodies"], colors_v):
    pc.set_facecolor(color); pc.set_alpha(0.6)
ax5.set_xticks([1,2,3])
ax5.set_xticklabels(list(dist_dict.keys()), fontsize=9)
ax5.set_ylabel("Bray-Curtis Dissimilarity")
ax5.set_title(f"Within vs Between\nGroup Distances (p={p_disp:.3f})",
              fontweight="bold", fontsize=10)

# ── Plot 6: Distance heatmap ──
ax6 = fig.add_subplot(3, 3, 6)
# Sort by group for cleaner heatmap
sorted_idx = (group.sort_values()).index
sub_dm = bc_dm.loc[sorted_idx, sorted_idx]
sns.heatmap(sub_dm, ax=ax6, cmap="YlOrRd",
            xticklabels=False, yticklabels=False,
            cbar_kws={"shrink": 0.8, "label": "Bray-Curtis"})
ax6.set_title("Bray-Curtis Distance Heatmap\n(sorted by group)",
              fontweight="bold", fontsize=10)
# Add group color bar
n_lgg = sum(group.sort_values()=="LGG")
ax6.add_patch(plt.Rectangle((-1.8, 0), 1.2, n_lgg,
              color="#E74C3C", clip_on=False))
ax6.add_patch(plt.Rectangle((-1.8, n_lgg), 1.2, len(group)-n_lgg,
              color="#3498DB", clip_on=False))

# ── Plot 7: Top taxa driving separation ──
ax7 = fig.add_subplot(3, 3, 7)
lgg_mean = rel_df[group=="LGG"].mean()
plc_mean = rel_df[group=="Placebo"].mean()
diff     = (lgg_mean - plc_mean)
top15    = diff.abs().nlargest(15)
colors_bar = ["#E74C3C" if diff[t] > 0 else "#3498DB" for t in top15.index]
ax7.barh(range(len(top15)), top15.values[::-1],
         color=colors_bar[::-1], edgecolor="black", linewidth=0.4)
ax7.set_yticks(range(len(top15)))
ax7.set_yticklabels([t.replace("et rel.","").strip()
                     for t in top15.index[::-1]], fontsize=7)
ax7.set_title("Top 15 Taxa Driving\nGroup Separation",
              fontweight="bold", fontsize=10)
ax7.set_xlabel("Mean Rel. Abundance Difference")
ax7.axvline(0, color="black", lw=0.8)
lgg_patch = mpatches.Patch(color="#E74C3C", label="Higher in LGG")
plc_patch = mpatches.Patch(color="#3498DB", label="Higher in Placebo")
ax7.legend(handles=[lgg_patch, plc_patch], fontsize=8)

# ── Plot 8: PC1 distribution by group ──
ax8 = fig.add_subplot(3, 3, 8)
for g_lbl, color in PAL_GROUP.items():
    vals = bc_pcoa[bc_pcoa["Group"]==g_lbl]["PC1"].values
    sns.kdeplot(vals, ax=ax8, color=color, fill=True,
                alpha=0.3, label=g_lbl, linewidth=2)
_, p_pc1 = mannwhitneyu(
    bc_pcoa[bc_pcoa["Group"]=="LGG"]["PC1"].values,
    bc_pcoa[bc_pcoa["Group"]=="Placebo"]["PC1"].values)
ax8.set_title(f"PC1 Distribution by Group\np={p_pc1:.4f}",
              fontweight="bold", fontsize=10)
ax8.set_xlabel("PC1 Score"); ax8.set_ylabel("Density")
ax8.legend(fontsize=9)

# ── Plot 9: Summary stats table ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
rows = [
    ["BC — LGG vs Placebo",   f"{bc_f_g:.3f}",  f"{bc_p_g:.3f}",
     "***" if bc_p_g<0.001 else "*" if bc_p_g<0.05 else "ns"],
    ["BC — Time 1 vs 2",      f"{bc_f_t:.3f}",  f"{bc_p_t:.3f}",
     "***" if bc_p_t<0.001 else "*" if bc_p_t<0.05 else "ns"],
    ["Jac — LGG vs Placebo",  f"{jac_f_g:.3f}", f"{jac_p_g:.3f}",
     "***" if jac_p_g<0.001 else "*" if jac_p_g<0.05 else "ns"],
    ["Within LGG (mean)",     f"{np.mean(within_lgg):.4f}",     "-", "-"],
    ["Within Placebo (mean)", f"{np.mean(within_placebo):.4f}", "-", "-"],
    ["Between groups (mean)", f"{np.mean(between):.4f}",        "-", "-"],
    ["BC PC1 variance",       f"{bc_var[0]:.1f}%",              "-", "-"],
    ["BC PC2 variance",       f"{bc_var[1]:.1f}%",              "-", "-"],
]
tbl = ax9.table(cellText=rows,
                colLabels=["Metric", "F / Value", "p-value", "Sig"],
                cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.3, 2.0)
for j in range(4):
    tbl[(0,j)].set_facecolor("#BDC3C7")
ax9.set_title("Summary Statistics", fontweight="bold", fontsize=11, pad=20)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("outputs/beta_diversity_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → outputs/beta_diversity_dashboard.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"BC PERMANOVA — LGG vs Placebo : F={bc_f_g:.3f}, p={bc_p_g:.3f}")
print(f"BC PERMANOVA — Time 1 vs 2   : F={bc_f_t:.3f}, p={bc_p_t:.3f}")
print(f"Jac PERMANOVA — LGG vs Plac  : F={jac_f_g:.3f}, p={jac_p_g:.3f}")
print(f"BC PC1 variance              : {bc_var[0]:.1f}%")
print(f"BC PC2 variance              : {bc_var[1]:.1f}%")
print(f"Within LGG mean dist         : {np.mean(within_lgg):.4f}")
print(f"Within Placebo mean dist     : {np.mean(within_placebo):.4f}")
print(f"Between-group mean dist      : {np.mean(between):.4f}")
print("\n✅ All outputs saved!")
