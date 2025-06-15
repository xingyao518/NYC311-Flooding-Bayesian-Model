import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
import json
from pathlib import Path


df = pd.read_csv("/NYC311-Flooding-Bayesian-Model/data/flood_311.csv", encoding='latin1', on_bad_lines='warn', low_memory=False)  # flooding complaints
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


import cmdstanpy, json, os, multiprocessing, textwrap, pathlib
from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
from pathlib import Path
import textwrap, json, tempfile, os

# ================================================================
#  8.  Stan program (inline)  +  fit  +  posterior prediction
# ================================================================


stan_program = textwrap.dedent(r"""
functions {
  /* ---- partial log-likelihood for reduce_sum ------------- */
  real partial_sum_lpmf(int[] y_slice,
                        int start, int end,
                        matrix X_tot,
                        vector log_d,
                        vector beta_tot,
                        real theta) {

    real lp = 0;
    for (n in start:end) {
      if (y_slice[n - start + 1] == 0) {
        lp += log_sum_exp(   bernoulli_lpmf(1 | theta),
                             bernoulli_lpmf(0 | theta)
                           + poisson_log_lpmf(0 | X_tot[n] * beta_tot + log_d[n]) );
      } else {
        lp += bernoulli_lpmf(0 | theta)
            + poisson_log_lpmf(y_slice[n - start + 1] | X_tot[n] * beta_tot + log_d[n]);
      }
    }
    return lp;
  }
}

data {
  int<lower=0> N_incidents;
  int<lower=0> N_tract;
  int<lower=0> N_category;
  int<lower=0> covariate_matrix_width;

  // design matrices
  matrix[N_incidents, covariate_matrix_width] X;
  matrix[N_incidents, N_category]            X_category;
  matrix[N_incidents, N_tract]               X_tract;
  vector[N_incidents]                        ones;

  // exposure
  vector<lower=0>[N_incidents] duration;
  int<lower=0> y[N_incidents];

  // ICAR adjacency
  int<lower=0> N_edges;
  int<lower=1, upper=N_tract> node1[N_edges];
  int<lower=1, upper=N_tract> node2[N_edges];
}

transformed data {
  vector[N_incidents] log_d = log(duration);
  matrix[N_incidents, 1 + N_tract + N_category + covariate_matrix_width] X_tot;

  X_tot = append_col( append_col(append_col( ones, X_tract ), X_category ), X );
}

parameters {
  vector[N_tract - 1]    beta_tract_raw;
  vector[N_category - 1] beta_cat_raw;
  vector[covariate_matrix_width] beta_cont;
  real  beta0;
  real<lower=0,upper=1> theta;
}

transformed parameters {
  vector[N_tract]    beta_tract;
  vector[N_category] beta_cat;
  vector[1 + N_tract + N_category + covariate_matrix_width] beta_tot;

  // sum-to-zero constraints (avoid non-identifiability)
  beta_tract[1:(N_tract-1)] = beta_tract_raw;
  beta_tract[N_tract]       = -sum(beta_tract_raw);

  beta_cat[1:(N_category-1)] = beta_cat_raw;
  beta_cat[N_category]       = -sum(beta_cat_raw);

  beta_tot = append_row(append_row(append_row(beta0, beta_tract), beta_cat), beta_cont);
}

model {
  /* ---- priors ------------------------------------------- */
  beta_tract      ~ normal(0, 1);
  beta_cat        ~ normal(0, 2 / sqrt(1 + N_category) );
  beta_cont       ~ normal(0, 1);
  beta0           ~ normal(0, 5);

  // spatial smoothing (ICAR): τ = 5 fixed
  target += -5 * dot_self(beta_tract[node1] - beta_tract[node2]);

  /* ---- likelihood via reduce_sum ------------------------ */
  target += reduce_sum(partial_sum_lpmf, y, 1,
                       X_tot, log_d, beta_tot, theta);
}

generated quantities {
  array[N_incidents] int y_rep;
  vector[N_incidents] log_lik;

  for (i in 1:N_incidents) {
    real eta = X_tot[i] * beta_tot + log_d[i];
    real lam = exp(eta);

    // posterior predictive
    int z = bernoulli_rng(theta);
    y_rep[i] = z == 1 ? 0 : poisson_rng(lam);

    // pointwise log-likelihood for loo/ic
    if (y[i] == 0) {
      log_lik[i] = log_sum_exp( bernoulli_lpmf(1 | theta),
                                bernoulli_lpmf(0 | theta)
                                + poisson_log_lpmf(0 | eta) );
    } else {
      log_lik[i] = bernoulli_lpmf(0 | theta)
                 + poisson_log_lpmf(y[i] | eta);
    }
  }
}
""")

# ---------- compile ----------------------------------------
model = CmdStanModel(stan_code=stan_program)

# ---------- sample -----------------------------------------
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=400,
    iter_sampling=400,
    seed=20240527,
    refresh=100,
    output_dir="stan_output/my_run",
)

print(fit.summary().loc[["beta0","theta"],["Mean","R_hat","Ess_bulk"]])

# ---------- quick posterior-predictive check ---------------
draws = fit.draws_pd(vars=["y_rep"])
ppc_counts = draws.filter(regex=r"y_rep\[\d+\]").astype(int)
print("\nPosterior-predictive mean duplicate-count (first 10 incidents):")
print(ppc_counts.mean(axis=0)[:10].to_numpy())
