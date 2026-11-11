"""Build and save the three Section B(b) figures."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline import CLEANED, ROOT, USERS


NAVY = "#145F82"
ORANGE = "#BA531A"
CHARGER_COLORS = {"AC": NAVY, "DC": ORANGE}


def hourly_charging_energy(sessions):
    """Sum each session's delivered energy in its recorded start-hour bin."""
    return sessions.groupby("Start_Hour")["Energy_kWh"].sum().reindex(
        range(24), fill_value=0
    )


def charging_energy_figure(sessions):
    """Plot the total charging energy of sessions beginning in each hour."""
    energy = hourly_charging_energy(sessions)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(9.2, 3.55), layout="constrained")
        ax.plot(energy.index, energy, color=NAVY, lw=2.3)
        peak_hour = energy.idxmax()
        peak_energy = energy.loc[peak_hour]
        ax.scatter(peak_hour, peak_energy, color=NAVY, zorder=4)
        ax.annotate(
            f"{peak_hour:02d}:00  |  {peak_energy:,.2f} kWh",
            (peak_hour, peak_energy), xytext=(0, 8), ha="center",
            textcoords="offset points", color=NAVY,
        )
        ax.set(
            title="Delivered charging energy by session start hour",
            xlabel="Recorded start hour", ylabel="Total delivered energy (kWh)",
        )
        ticks = range(0, 24, 2)
        ax.set_xticks(ticks, [f"{hour:02d}:00" for hour in ticks])
        ax.set_ylim(bottom=0, top=max(1, peak_energy * 1.25))
        sns.despine(ax=ax)
    return fig


def cost_figure(sessions):
    """Compare billed charges by user type with 1.5-IQR whiskers."""
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(9.2, 3.55), layout="constrained")
        sns.boxplot(
            data=sessions, x="User_Type", y="Net_Billable_MYR",
            order=USERS, color="#66A8BE", whis=1.5, width=0.52,
            fliersize=3, ax=ax,
        )
        counts = sessions["User_Type"].value_counts()
        ax.set_xticks(
            range(len(USERS)),
            [f"{kind}  (n={counts.get(kind, 0):,})" for kind in USERS],
        )
        ax.set(title="Charging session cost by user type",
               xlabel="User type", ylabel="Net billed amount (MYR)")
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)
    return fig


def temperature_figure(sessions):
    """Show battery temperature versus energy with separate linear fits."""
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(9.2, 3.7), layout="constrained")
        for kind, color in CHARGER_COLORS.items():
            group = sessions.loc[sessions["Charger_Type"].eq(kind)]
            if group.empty:
                continue
            x = group["Battery_Temp_C"]
            y = group["Energy_kWh"]
            ax.scatter(
                x, y, color=color, s=12, alpha=0.32, label=f"{kind} sessions"
            )
            # A straight line needs at least two distinct x values.
            if x.nunique() >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                span = np.array([x.min(), x.max()])
                ax.plot(span, slope * span + intercept, color=color,
                        lw=2.4, ls="-" if kind == "AC" else "--",
                        label=f"{kind} linear fit")
        ax.set(title="Energy delivered versus battery temperature",
               xlabel="Battery temperature (°C)",
               ylabel="Energy delivered (kWh)")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, ncol=2, loc="best")
        sns.despine(ax=ax)
    return fig


def save_figures(sessions, destination):
    """Export 300 dpi PNGs of the complete dataset and release figures."""
    destination.mkdir(parents=True, exist_ok=True)
    panels = (
        ("figure_3_hourly_charging_energy.png", charging_energy_figure(sessions)),
        ("figure_4_cost_boxplot.png", cost_figure(sessions)),
        ("figure_5_temperature_energy.png", temperature_figure(sessions)),
    )
    for name, figure in panels:
        try:
            figure.savefig(destination / name, dpi=300, facecolor="white")
            print(f"Saved {name}")
        finally:
            plt.close(figure)


def main():
    """Read the clean CSV and export the three visualizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEANED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    try:
        sessions = pd.read_csv(args.input)
        save_figures(sessions, args.output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Chart export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

