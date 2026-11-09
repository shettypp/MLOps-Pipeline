"""Build paper figures + LaTeX tables from results/*.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
F = Path(__file__).resolve().parent.parent / "paper" / "figs"; F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
COL = {"No retraining": "#9a9a9a", "Fixed (every 4 win)": "#5b8fd6", "KS-triggered": "#e08a3c",
       "Multi-signal (ours)": "#2a7d4f"}
summary = {}

# ---------- Exp 1 (all available seeds) ----------
runs = [json.loads(f.read_text()) for f in sorted(R.glob("exp1_results_s*.json"))]
df1 = pd.DataFrame([{**r, "seed": i} for i, run in enumerate(runs) for r in run["results"]])
agg = df1.groupby("strategy", sort=False).agg(acc_mean=("mean_acc", "mean"), acc_sd=("mean_acc", "std"),
        min_acc=("min_acc", "mean"), retrains=("retrains", "mean"), rt_sd=("retrains", "std"),
        promoted=("promoted", "mean"))
summary["exp1"] = agg.reset_index().to_dict("records"); summary["exp1_seeds"] = len(runs)
summary["exp1_initial"] = [r["initial"] for r in runs]
e1 = runs[-1]
fig, ax = plt.subplots(figsize=(3.45, 1.9))
for name, c in COL.items():
    t = e1["traces"][name]; a = np.array(t["acc"]); x = np.arange(len(a))
    sm = pd.Series(a).rolling(3, min_periods=1).mean()
    ax.plot(x, sm, color=c, lw=1.1 if "ours" in name else .8, label=name.replace(" (every 4 win)", " /4"))
    if t["retrain_at"] and name != "No retraining":
        ax.plot(t["retrain_at"], [0.47 + 0.02 * list(COL).index(name)] * len(t["retrain_at"]),
                "|", color=c, ms=4)
ax.set_xlabel("Window (500 records)"); ax.set_ylabel("Accuracy (3-win avg)"); ax.set_ylim(.45, 1)
ax.legend(fontsize=6, ncol=2, loc="upper center", frameon=False)
fig.tight_layout(pad=.3); fig.savefig(F / "elec2_acc.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
summary["exp1_static_first40"] = [float(np.mean(r["traces"]["No retraining"]["acc"][:40])) for r in runs]

# ---------- Exp 2 ----------
d = pd.DataFrame(json.loads((R / "exp2_raw.json").read_text()))
order = ["KS", "PSI", "MS 1-vote (OR)", "MS no-persistence", "Multi-signal", "ErrorRate (labels)"]
pre = d.groupby("detector").pre_alarm_rate.mean()
post = d.pivot_table(index="detector", columns="scenario", values="post_alarm_rate", aggfunc="mean")
det = d.pivot_table(index="detector", columns="scenario", values="detected", aggfunc="mean")
delay = d[d.detected].pivot_table(index="detector", columns="scenario", values="delay", aggfunc="mean")
acc = d[d.detector == "KS"].groupby("scenario")[["acc_pre", "acc_post"]].mean()
summary["exp2"] = {"pre_alarm": pre.to_dict(), "post_alarm": post.to_dict(), "detected": det.to_dict(),
                   "delay": delay.to_dict(), "acc": acc.to_dict(), "seeds": int(d.seed.nunique()),
                   "estimators": d.drop_duplicates("seed").estimator.value_counts().to_dict()}
scen = ["benign-1.0", "benign-2.0", "harmful-0.3", "harmful-0.6", "harmful-1.0"]
lab = ["Benign\n1.0σ", "Benign\n2.0σ", "Harmful\n0.3σ", "Harmful\n0.6σ", "Harmful\n1.0σ"]
fig, ax = plt.subplots(figsize=(3.45, 1.85))
show = ["KS", "PSI", "Multi-signal", "ErrorRate (labels)"]
cc = ["#e08a3c", "#c9a227", "#2a7d4f", "#7b6bb0"]
w = .2
for i, (k, c) in enumerate(zip(show, cc)):
    ax.bar(np.arange(len(scen)) + (i - 1.5) * w, [post.loc[k, s] for s in scen], w, color=c,
           label=k.replace(" (labels)", "*"))
ax.axvline(1.5, color="k", lw=.5, ls=":")
ax.set_xticks(range(len(scen))); ax.set_xticklabels(lab, fontsize=6.5)
ax.set_ylabel("Windows alarmed\nafter drift", fontsize=7); ax.set_ylim(0, 1.18)
ax.text(.5, 1.08, "should NOT alarm", ha="center", fontsize=6); ax.text(3, 1.08, "should alarm", ha="center", fontsize=6)
ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.32), frameon=False)
fig.tight_layout(pad=.3); fig.savefig(F / "controlled.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------- Exp 3 ----------
p3 = R / "exp3_live.json"
if p3.exists():
    e3 = json.loads(p3.read_text())
    summary["exp3"] = {k: e3[k] for k in ("rows", "requests", "batch", "wall_s", "latency_ms", "live_acc", "static_v1_acc_same_rows",
                                            "versions_served", "initial")}
    summary["exp3"]["retrains"] = [{k: r.get(k) for k in ("version", "promoted", "acc_old", "acc_new",
                                    "estimator", "seconds", "alarm_to_serving_s", "at_row")} for r in e3["retrains"]]
    summary["exp3"]["retrains"] = [{k: r.get(k) for k in ("version", "promoted", "acc_old", "acc_new", "estimator", "seconds", "detect_s", "swap_s", "alarm_to_serving_s", "at_row")} for r in e3["retrains"]]
    summary["exp3"]["windows"] = len(e3["drift_events"])
    summary["exp3"]["alarms"] = int(sum(e["alarm"] for e in e3["drift_events"]))
(R / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float)[:6000])

# ---------- LaTeX table for Exp 1 ----------
rows = []
for r in summary["exp1"]:
    name = r["strategy"].replace("Multi-signal (ours)", r"\textbf{Multi-signal (ours)}").replace("(labels)", "(labels)$^*$")
    sd = f"$\\pm${r['acc_sd']*100:.1f}" if r["acc_sd"] == r["acc_sd"] else ""
    rows.append(f"{name} & {r['acc_mean']*100:.1f}{sd} & {r['min_acc']*100:.1f} & {r['retrains']:.1f} & {r['promoted']:.1f} \\\\")
tab = (r"""\begin{table}[t]
\caption{Elec2 stream, mean over %d runs (82 windows of 500). Retrains and promotions are counts per run.}
\label{tab:elec2}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Strategy & Acc.\ (\%%) & Worst win. & Retrains & Promoted \\
\midrule
""" % summary["exp1_seeds"]) + "\n".join(rows) + r"""
\bottomrule
\multicolumn{5}{l}{\scriptsize $^*$needs true labels at check time (supervised reference).}
\end{tabular}
\end{table}"""
(Path(__file__).resolve().parent.parent / "paper" / "table_elec2.tex").write_text(tab)
