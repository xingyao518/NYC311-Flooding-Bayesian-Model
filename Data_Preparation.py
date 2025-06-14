import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
import json
from pathlib import Path


df = pd.read_csv("/NYC311-Flooding-Bayesian-Model/data/311_Service.csv", encoding='latin1', on_bad_lines='warn', low_memory=False)  # flooding complaints
df = df.dropna(subset=["Latitude", "Longitude"])

geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
gdf_311 = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

nyc_shape = gpd.read_file("/NYC311-Flooding-Bayesian-Model/data/cb_2018_36_tract_500k.shp")
nyc_shape = nyc_shape.to_crs("EPSG:4326")

joined = gpd.sjoin(gdf_311, nyc_shape, how="inner", predicate="within")

category_counts = (
    joined.groupby(["GEOID", "Complaint Type"]).size().unstack(fill_value=0)
)
joined["Category"] = joined["Complaint Type"]
joined["census_tract"] = joined["GEOID"].astype(str).str[:12]



#  Clean + build stan_data

# ---------- 0. Path configuration ----------
adj_path = "/NYC311-Flooding-Bayesian-Model/data/tract_adjacency_connected.npz"

# ---------- 1. Basic cleaning ----------
df = joined.copy()

# Drop rows with missing values in key columns
df = df.dropna(subset=["census_tract", "Category", "Created Date", "Borough"])

# Truncate to 12-character tract string
df["census_tract"] = df["census_tract"].astype(str).str[:12]

# Parse dates
df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
df["Closed Date"]  = pd.to_datetime(df["Closed Date"],  errors="coerce")

df["Borough"] = df["Borough"].str.strip().str.title()   # e.g. 'BRONX' → 'Bronx'
valid_boros = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
df = df[df["Borough"].isin(valid_boros)].copy()

# Keep only tracts with a valid borough
tract_borough_map = (
    df[["census_tract", "Borough"]]
    .drop_duplicates()
    .dropna()
    .set_index("census_tract")["Borough"]
)
df = df[df["census_tract"].isin(tract_borough_map.index)].copy()

# ---------- 2. Mapping dictionaries ----------
tract_list = sorted(df["census_tract"].unique())
cat_list   = sorted(df["Category"].unique())

tract_to_id   = {t: i for i, t in enumerate(tract_list)}
cat_to_id     = {c: i for i, c in enumerate(cat_list)}
tract_to_boro = tract_borough_map.to_dict()

borough_list  = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
borough_to_id = {b: i for i, b in enumerate(borough_list)}
N_tract = len(tract_to_id)
N_category = len(cat_to_id)

# ---------- 3. Compute 'duration' column ----------
delta = (df["Closed Date"] - df["Created Date"]).dt.total_seconds() / 86400
df["dur_days"] = delta.fillna(1.0).clip(lower=0.1)

# ---------- 4. Aggregate to (tract, category) ----------
records, y, durations, borough_ids = [], [], [], []

for (t, c), g in df.groupby(["census_tract", "Category"]):
    tid = tract_to_id[t]
    cid = cat_to_id[c]
    bid = borough_to_id[tract_to_boro[t]]

    records.append((tid, cid))
    y.append(len(g))
    durations.append(g["dur_days"].mean())
    borough_ids.append(bid)

df_events     = pd.DataFrame(records, columns=["tract_id", "cat_id"])
N_incidents   = len(df_events)
cov_width     = 0  # No extra continuous covariates, can be added manually
covariate_matrix_width = cov_width

# ---------- 5. Design matrix ----------
# category one-hot
X_category = np.eye(len(cat_list))[df_events["cat_id"].values]

# borough one-hot
X_borough  = np.eye(5)[borough_ids]

# Continuous covariate matrix (currently empty)
X = np.zeros((N_incidents, cov_width))

# Other vectors
ones  = np.ones(N_incidents)
duration  = durations

# ---------- 6. Adjacency edge list ----------
adj_npz = np.load(adj_path, allow_pickle=True)
A_coo   = csr_matrix(
            (adj_npz["data"], adj_npz["indices"], adj_npz["indptr"]),
            shape=adj_npz["shape"]).tocoo()

node1, node2 = [], []
for i, j in zip(A_coo.row, A_coo.col):
    if i < j:             # Avoid duplicates
        node1.append(i + 1)   # Stan uses 1-based indexing
        node2.append(j + 1)
N_edges = len(node1)

# ---------- 7. Assemble stan_data ----------
# ==== Construct X_tract / X_category one-hot ====
import scipy.sparse as sp

row_idx = np.arange(N_incidents)

X_tract_mat = sp.coo_matrix(
        (np.ones(N_incidents), (row_idx, df_events["tract_id"])),
        shape=(N_incidents, N_tract)).toarray()

X_cat_mat = sp.coo_matrix(
        (np.ones(N_incidents), (row_idx, df_events["cat_id"])),
        shape=(N_incidents, N_category)).toarray()
# ==========================================

# ---- Assemble stan_data ----
stan_data = {
    "N_incidents": N_incidents,
    "N_tract":     N_tract,
    "N_category":  N_category,
    "covariate_matrix_width": covariate_matrix_width,
    "y": y,
    "duration": duration,
    "ones": ones,
    "X": X,                       # Placeholder for other continuous covariates
    "X_tract":    X_tract_mat.tolist(),
    "X_category": X_cat_mat.tolist(),
    "N_edges": N_edges,
    "node1": node1,
    "node2": node2,
}

stan_data["N"] = stan_data["N_tract"]
stan_data["K"] = stan_data["N_category"]

def _np_convert(obj):
    if isinstance(obj, (np.integer,)):      # int32/64
        return int(obj)
    if isinstance(obj, (np.floating,)):     # float32/64
        return float(obj)
    if isinstance(obj, np.ndarray):         # vector/matrix
        return obj.tolist()
    raise TypeError(f"{type(obj)} not serializable")

out_path = Path("cleaned_stan_data.json")
out_path.write_text(json.dumps(stan_data, default=_np_convert))   # Key
print(" cleaned_stan_data.json written")
print(f"   N_incidents : {N_incidents}")
print(f"   N_tract     : {len(tract_list)}")
print(f"   N_category  : {len(cat_list)}")
