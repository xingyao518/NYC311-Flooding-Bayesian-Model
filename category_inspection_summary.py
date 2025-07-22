import pandas as pd

# Load data
df = pd.read_csv("data/flood_311.csv", encoding="latin1", low_memory=False)

# Convert to datetime
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

# Compute Time to Inspection (in days)
df["Time to Inspection"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / (60 * 60 * 24)

# Remove negative values
df = df[df["Time to Inspection"] >= 0]

# Group and summarize by complaint_type
summary = df.groupby("complaint_type").agg(
    Requests=("unique_key", "count"),
    Inspected_Requests=("closed_date", lambda x: x.notna().sum()),
    Fraction_Inspected=("closed_date", lambda x: x.notna().mean()),
    Unique_Incidents=("incident_address", "nunique"),
    Avg_Reports_Incident=("unique_key", lambda x: len(x) / df["incident_address"].nunique()),
    Median_Inspection_Days=("Time to Inspection", "median")
).reset_index()

# Save summary
summary.to_csv("category_inspection_summary.csv", index=False)
print("Summary saved to category_inspection_summary.csv")

