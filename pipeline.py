"""Clean EV sessions, calculate billing features, and export a CSV."""

import argparse
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ev_campus_dataset (1).csv"
CLEANED = ROOT / "cleaned_energy_sales_data.csv"
CHARGER_RATES = {"AC": 6.0, "DC": 15.0}
USERS = ("Student", "Staff", "Visitor")
GRID_STATES = ("Grid_Surge", "Normal", "Overheated")
NUMERIC = (
    "Start_Hour", "Duration_Mins", "Energy_kWh",
    "Overstay_Mins", "Battery_Temp_C",
)
CATEGORIES = {
    "Charger_Type": tuple(CHARGER_RATES),
    "User_Type": USERS,
    "Grid_Status_Label": GRID_STATES,
}
REQUIRED = {"Session_ID", *NUMERIC, *CATEGORIES}


def clean_sessions(raw):
    """Validate the session schema and return cleaned rows and an audit."""
    missing = REQUIRED.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if raw.empty:
        raise ValueError("No sessions were supplied.")

    clean = raw.copy()
    ids = clean["Session_ID"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Session_ID must be present and unique.")
    clean["Session_ID"] = ids

    # Case-insensitive matching keeps approved spellings in the exported CSV.
    for column, allowed in CATEGORIES.items():
        values = clean[column].astype("string").str.strip()
        lookup = {value.casefold(): value for value in allowed}
        normalized = values.str.casefold().map(lookup)
        if normalized.isna().any():
            invalid = values[normalized.isna()].fillna("<missing>").unique()
            raise ValueError(f"Invalid values for {column}: {list(invalid)}")
        clean[column] = normalized

    audit_rows = []
    for column in NUMERIC:
        values = clean[column]
        parsed = pd.to_numeric(values, errors="coerce")
        bad_text = values.notna() & parsed.isna()
        if bad_text.any() or np.isinf(parsed).any():
            raise ValueError(f"{column} contains invalid numeric values.")
        present = parsed.dropna()
        if present.empty:
            raise ValueError(f"{column} has no values for median imputation.")
        if column == "Duration_Mins" and present.le(0).any():
            raise ValueError("Duration_Mins must be positive.")
        if column in ("Energy_kWh", "Overstay_Mins") and present.lt(0).any():
            raise ValueError(f"{column} must be nonnegative.")
        if column == "Start_Hour" and (
            not present.between(0, 23).all() or present.mod(1).ne(0).any()
        ):
            raise ValueError("Start_Hour must be an integer from 0 to 23.")
        median = float(present.median())
        fill = np.floor(median + 0.5) if column == "Start_Hour" else median
        clean[column] = parsed.fillna(fill)
        if column == "Start_Hour":
            clean[column] = clean[column].astype(int)
        audit_rows.append((column, int(parsed.isna().sum()), float(fill)))

    if "Session_Date" in clean:
        # Keep a real date when it exists. Reject partial or ambiguous dates.
        dates = pd.to_datetime(clean["Session_Date"], format="%Y-%m-%d")
        if dates.isna().any():
            raise ValueError("Session_Date must be present on every session.")
        clean["Session_Date"] = dates.dt.strftime("%Y-%m-%d")

    audit = pd.DataFrame(
        audit_rows, columns=["Column", "Missing before", "Fill value"]
    )
    return clean, audit


def add_features(clean):
    """Add charging power, carbon savings, and tariff components."""
    data = clean.copy()
    hours = data["Duration_Mins"] / 60
    peak = data["Start_Hour"].between(12, 17)
    rates = data["Charger_Type"].map(CHARGER_RATES)

    data["Duration_Hours"] = hours
    data["Power_Rate_kW"] = data["Energy_kWh"] / hours
    data["Carbon_Savings_kgCO2"] = (data["Energy_kWh"] * 0.475).round(2)
    data["Base_Rate_MYR"] = rates
    data["Base_Fee_MYR"] = hours * rates
    data["Peak_Surge_MYR"] = data["Base_Fee_MYR"].where(peak, 0) * 0.25
    billable_overstay = (data["Overstay_Mins"] - 15).clip(lower=0)
    data["Overstay_Fee_MYR"] = billable_overstay * 0.50
    # Keep fee components at calculation precision until net billing.
    data["Net_Billable_MYR"] = (
        data["Base_Fee_MYR"]
        + data["Peak_Surge_MYR"]
        + data["Overstay_Fee_MYR"]
    ).round(2)
    return data


def export_csv(data, destination):
    """Replace the destination only after the full CSV has been written."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent,
        prefix=f".{destination.stem}-", suffix=".tmp", delete=False,
    ) as temporary:
        staged = Path(temporary.name)
    try:
        data.to_csv(staged, index=False, encoding="utf-8")
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def run_pipeline(source, destination):
    """Load, clean, transform, export, and return the data with its audit."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Use separate input and output paths.")
    raw = pd.read_csv(source, encoding="utf-8-sig")
    clean, audit = clean_sessions(raw)
    result = add_features(clean)
    export_csv(result, destination)
    return result, audit


def main():
    """Run the pipeline from the command line with a readable summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=CLEANED)
    args = parser.parse_args()
    try:
        data, audit = run_pipeline(args.input, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1

    print(f"Sessions retained: {len(data):,}")
    print(audit.to_string(index=False))
    print(f"Missing required cells after cleaning: "
          f"{data[list(REQUIRED)].isna().sum().sum()}")
    mean_power = data["Power_Rate_kW"].mean()
    print(f"Mean session charging power: {mean_power:.2f} kW")
    print(f"Energy delivered: {data['Energy_kWh'].sum():,.2f} kWh")
    carbon = data["Carbon_Savings_kgCO2"].sum()
    print(f"Estimated carbon savings: {carbon:,.2f} kg CO2")
    print(f"Net billed revenue: RM {data['Net_Billable_MYR'].sum():,.2f}")
    print(f"Exported: {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

