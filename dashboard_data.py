"""Prepare filtered views, alert flags, and KPIs from cleaned EV sessions."""

import numpy as np
import pandas as pd


OVERSTAY_LIMIT = 20
DEFAULT_TEMP = 40.0
CRITICAL_MARGIN = 5.0
DASHBOARD_COLUMNS = {
    "Session_ID", "Charger_Type", "User_Type", "Start_Hour", "Energy_kWh",
    "Battery_Temp_C", "Overstay_Mins", "Carbon_Savings_kgCO2",
    "Net_Billable_MYR",
}


def prepare_sessions(data):
    """Validate dashboard inputs and parse a supplied session date if present."""
    missing = DASHBOARD_COLUMNS.difference(data.columns)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Cleaned CSV is missing: {fields}")
    if data.empty:
        raise ValueError("The cleaned CSV has no sessions.")
    if "Session_Date" in data:
        dates = pd.to_datetime(data["Session_Date"], format="%Y-%m-%d")
        if dates.isna().any():
            raise ValueError("Session_Date is incomplete.")
        data = data.assign(Session_Date=dates)
    return data


def filter_sessions(data, start_hour_range, minimum, charger_types,
                    date_range=None):
    """Filter by recorded start hour, energy, charger and any supplied date."""
    first_hour, last_hour = start_hour_range
    matches = (
        data["Start_Hour"].between(first_hour, last_hour)
        & data["Energy_kWh"].ge(minimum)
        & data["Charger_Type"].isin(charger_types)
    )
    if date_range is not None:
        first_date, last_date = map(pd.Timestamp, date_range)
        matches &= data["Session_Date"].between(first_date, last_date)
    return data.loc[matches].copy()


def flag_alerts(data, threshold):
    """Flag long overstays above the selected battery-temperature limit."""
    hot_overstay = (
        data["Overstay_Mins"].gt(OVERSTAY_LIMIT)
        & data["Battery_Temp_C"].gt(threshold)
    )
    critical = hot_overstay & data["Battery_Temp_C"].ge(
        threshold + CRITICAL_MARGIN
    )
    return pd.Series(
        np.select([critical, hot_overstay], ["Critical", "Warning"], "Clear"),
        index=data.index,
        name="Alert_Level",
    )


def indicators(data, threshold):
    """Return the four KPI values in dashboard order."""
    return (
        data["Energy_kWh"].sum(),
        data["Carbon_Savings_kgCO2"].sum(),
        data["Net_Billable_MYR"].sum(),
        int(flag_alerts(data, threshold).ne("Clear").sum()),
    )


def prepare_sales(data):
    """Parse supplied transaction dates and validate daily revenue inputs."""
    required = {"Transaction_ID", "Transaction_Date", "Total_Revenue_MYR"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Sales CSV is missing: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(data["Transaction_Date"], format="%Y-%m-%d")
    revenue = pd.to_numeric(data["Total_Revenue_MYR"], errors="raise")
    if data.empty or dates.isna().any() or not np.isfinite(revenue.dropna()).all():
        raise ValueError("Sales dates must be complete and revenue values must be finite.")
    return data.assign(Transaction_Date=dates, Total_Revenue_MYR=revenue)


def sales_in_range(data, date_range):
    """Return sales within both inclusive transaction-date bounds."""
    start, end = map(pd.Timestamp, date_range)
    return data.loc[data["Transaction_Date"].between(start, end)]


def daily_sales_revenue(data):
    """Sum the supplied transaction revenue for each observed calendar day."""
    return data.groupby("Transaction_Date")["Total_Revenue_MYR"].sum(min_count=1)
