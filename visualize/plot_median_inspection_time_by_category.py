import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read CSV
df = pd.read_csv("data/flood_311.csv", encoding="latin1", low_memory=False)

# Step 2: Convert to datetime
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

# Step 3: Compute inspection delay in days
df["inspection_delay"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / (3600 * 24)

# Step 4: Group by complaint_type and compute median inspection time
summary = (
    df.groupby("complaint_type")
      .agg(Median_Inspection_Time_Days=("inspection_delay", "median"))
      .reset_index()
)


# Optional: sort by delay
summary_sorted = summary.sort_values("Median_Inspection_Time_Days", ascending=False)

# Step 5: Plot
plt.figure(figsize=(10, 6))
plt.barh(summary_sorted["complaint_type"], summary_sorted["Median_Inspection_Time_Days"])
plt.xlabel("Median Time to Inspection (Days)")
plt.title("Median Inspection Time by Complaint Type")
plt.tight_layout()
plt.savefig("median_inspection_time_by_category.png", dpi=300)
plt.show()
