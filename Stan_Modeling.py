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
