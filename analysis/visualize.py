"""Part 2, Task 11: two visualisations, generated entirely from code.

Re-uses the cleaning pipeline in clean_and_eda.py (quietly) so the charts can
never drift from the printed numbers.

Run from anywhere:   python analysis/visualize.py
Outputs:             visualizations/return_rate_by_payment.png
                     visualizations/monthly_revenue_trend.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_and_eda import run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "visualizations"

ACCENT, GREY, INK = "#C9822A", "#9AA79F", "#1F2A24"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9AA79F",
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def chart_return_rate(R):
    by_pay = R["_by_pay"].sort_values("return_rate_pct", ascending=False)
    pays = list(by_pay.index)
    vals = by_pay["return_rate_pct"].tolist()
    cod_rate = float(by_pay.loc["COD", "return_rate_pct"])
    card_rate = float(by_pay.loc["CARD", "return_rate_pct"])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(pays, vals, color=[ACCENT if p == "COD" else GREY for p in pays], width=0.62)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("Return rate (% of orders)")
    ax.set_ylim(0, max(vals) + 12)
    ax.set_title(f"COD Returns at {cod_rate:.1f}% — {cod_rate / card_rate:.1f}x Card", loc="left", fontsize=13,
                 fontweight="bold", color=INK, pad=14)
    fig.tight_layout()
    path = OUT / "return_rate_by_payment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_monthly(R):
    trend = R["_trend"]
    months = list(trend.index)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.plot(months, trend["without_outliers"], color=ACCENT, lw=2.8, marker="o",
            label="Outlier-corrected revenue")
    tpk = R["true_peak_month"]
    ax.annotate(f"True peak {tpk}\n{trend.loc[tpk, 'without_outliers']:,.0f}", (tpk, trend.loc[tpk, "without_outliers"]),
                textcoords="offset points", xytext=(-10, 28), fontsize=9.5, color=ACCENT,
                fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-", color=ACCENT))
    ax.set_ylabel("Revenue (INR)")
    ax.set_xlabel("Order month")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, trend["without_outliers"].max() * 1.18)
    ax.set_title(f"Outlier-corrected monthly revenue peaks in {tpk}", loc="left", fontsize=13,
                 fontweight="bold", color=INK, pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    path = OUT / "monthly_revenue_trend.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(exist_ok=True)
    R = run(verbose=False, write_findings=False)
    for p in (chart_return_rate(R), chart_monthly(R)):
        print("Saved", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
