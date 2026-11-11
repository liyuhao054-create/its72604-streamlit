"""Streamlit dashboard for campus energy, charging revenue, and alerts."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from b_visualizations import (
    charging_energy_figure,
    cost_figure,
    temperature_figure,
)
from dashboard_data import (
    CRITICAL_MARGIN,
    DEFAULT_TEMP,
    OVERSTAY_LIMIT,
    daily_sales_revenue,
    filter_sessions,
    flag_alerts,
    indicators,
    prepare_sales,
    prepare_sessions,
    sales_in_range,
)
from pipeline import CHARGER_RATES, CLEANED, ROOT


SALES = ROOT / "sales_data.csv"


@st.cache_data
def load_sessions(path, modified_ns):
    """Load the cleaned data and invalidate the cache when the CSV changes."""
    return prepare_sessions(pd.read_csv(path))



@st.cache_data
def load_sales(path, modified_ns):
    """Load the dated transaction file and invalidate the cache on edits."""
    return prepare_sales(pd.read_csv(path))


def display_chart(figure):
    """Render and close one figure on every dashboard rerun."""
    try:
        st.pyplot(figure, use_container_width=True)
    finally:
        plt.close(figure)


def show_overview(data, filtered, threshold):
    """Display comparable KPI deltas and three charts for the shared subset."""
    st.title("Campus EV charging overview")
    st.caption(
        "Deltas compare the selection with all sessions at the same "
        "temperature threshold."
    )
    current = indicators(filtered, threshold)
    baseline = indicators(data, threshold)
    labels = (
        "Energy delivered", "Carbon offset", "Net revenue", "Active alerts"
    )
    units = (" kWh", " kg CO₂", "", "")
    for i, card in enumerate(st.columns(4)):
        prefix = "RM " if i == 2 else ""
        digits = 0 if i == 3 else 2
        card.metric(
            labels[i],
            f"{prefix}{current[i]:,.{digits}f}{units[i]}",
            f"{current[i] - baseline[i]:+,.{digits}f}{units[i]}",
            delta_color="inverse" if i == 3 else "normal",
        )

    if filtered.empty:
        st.info("No charging sessions match these filters.")
        return

    st.subheader("Charging energy by start hour")
    st.caption("Each session contributes its total kWh to its start-hour group.")
    display_chart(charging_energy_figure(filtered))
    st.subheader("Charging cost by user")
    display_chart(cost_figure(filtered))
    st.subheader("Battery temperature and energy delivered")
    display_chart(temperature_figure(filtered))


def show_alerts(filtered, threshold):
    """Show threshold-based alerts and export the displayed rows."""
    st.title("Charging alerts and session data")
    st.caption(
        f"Alert rule: overstay > {OVERSTAY_LIMIT} min and battery temperature "
        f"> {threshold:g} °C. Critical alerts also reach "
        f"{threshold + CRITICAL_MARGIN:g} °C."
    )
    view = filtered.assign(Alert_Level=flag_alerts(filtered, threshold))
    sort_columns = ["Start_Hour", "Session_ID"]
    if "Session_Date" in view:
        view["Session_Date"] = view["Session_Date"].dt.strftime("%Y-%m-%d")
        sort_columns.insert(0, "Session_Date")
    view = view.sort_values(sort_columns)
    counts = view["Alert_Level"].value_counts()
    if counts.get("Critical", 0):
        st.error(f"{counts['Critical']} critical charging sessions.")
    if counts.get("Warning", 0):
        st.warning(f"{counts['Warning']} charging sessions require review.")
    if view.empty:
        st.info("No sessions match. The CSV contains column headings.")
    elif not counts.index.isin(["Critical", "Warning"]).any():
        st.success("No high-risk sessions at the current threshold.")

    st.subheader(f"Filtered sessions ({len(view):,})")
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button(
        "Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_charging_sessions.csv", mime="text/csv",
        on_click="ignore",
    )



def show_sales():
    """Explore recorded sales on their own transaction dates."""
    st.title("Campus sales")
    st.caption("Source: sales_data.csv | Transaction_Date")
    try:
        sales = load_sales(SALES, SALES.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to read sales_data.csv. {error}")
        return

    first, last = sales["Transaction_Date"].agg(["min", "max"])
    selection = st.sidebar.date_input(
        "Transaction date range", value=(first.date(), last.date()),
        min_value=first.date(), max_value=last.date(), key="sales_dates",
    )
    if len(selection) != 2:
        st.info("Select both dates to apply the date filter.")
        return

    filtered = sales_in_range(sales, selection)
    st.metric("Transactions", f"{len(filtered):,}")
    recorded_revenue = filtered["Total_Revenue_MYR"].dropna()
    value = f"RM {recorded_revenue.sum():,.2f}" if not recorded_revenue.empty else "—"
    st.metric("Recorded sales revenue", value)
    st.caption(f"{len(filtered) - len(recorded_revenue):,} transactions lack a recorded revenue amount.")
    if filtered.empty:
        st.info("No sales transactions match this date range.")
        return

    st.subheader("Daily sales revenue")
    st.line_chart(daily_sales_revenue(filtered).rename("Revenue (MYR)"))
    view = filtered.sort_values(["Transaction_Date", "Transaction_ID"]).copy()
    view["Transaction_Date"] = view["Transaction_Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(view, hide_index=True, use_container_width=True)


def main():
    """Initialize sales and the shared filters for both charging pages."""
    st.set_page_config(page_title="Campus EV Analytics", layout="wide")
    overview = st.Page(
        lambda: show_overview(sessions, filtered, threshold),
        title="Overview", default=True,
    )
    alerts = st.Page(
        lambda: show_alerts(filtered, threshold),
        title="Alerts and data", url_path="alerts",
    )
    sales = st.Page(show_sales, title="Sales", url_path="sales")
    page = st.navigation([overview, alerts, sales])
    if page.url_path == "sales":
        page.run()
        return

    try:
        sessions = load_sessions(CLEANED, CLEANED.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to read the cleaned CSV. Run pipeline.py. {error}")
        st.stop()

    st.sidebar.header("Session filters")
    # Start_Hour is recorded for every session in the supplied EV file.
    start_hours = st.sidebar.slider(
        "Session start hour", 0, 23, (0, 23), key="start_hours",
        help="Both selected hours are included.",
    )
    date_range = None
    if "Session_Date" in sessions:
        first, last = sessions["Session_Date"].agg(["min", "max"])
        date_range = st.sidebar.date_input(
            "Session date range", value=(first.date(), last.date()),
            min_value=first.date(), max_value=last.date(), key="session_dates",
        )
        if len(date_range) != 2:
            st.info("Select both dates to apply the date filter.")
            st.stop()
    minimum = st.sidebar.slider(
        "Minimum delivered energy (kWh)", 0.0,
        float(max(1, math.ceil(sessions["Energy_kWh"].max()))),
        0.0, step=0.5, key="minimum_energy",
    )
    chargers = st.sidebar.multiselect(
        "Charger type", tuple(CHARGER_RATES), default=tuple(CHARGER_RATES),
        key="chargers",
    )
    with st.sidebar:
        threshold = st.slider(
            "Battery temperature alert threshold (°C)",
            min_value=20.0, max_value=60.0, value=DEFAULT_TEMP,
            step=0.5, key="alert_temperature",
        )
    filtered = filter_sessions(
        sessions, start_hours, minimum, chargers, date_range
    )
    page.run()


if __name__ == "__main__":
    main()
