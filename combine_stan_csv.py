import pandas as pd
import glob

csv_files = glob.glob("stan_output/my_run/*.csv")
dfs = []

for f in csv_files:
    if not f.endswith("-stdout.txt"):
        df = pd.read_csv(f, comment='#')
        dfs.append(df)

combined_df = pd.concat(dfs, axis=0)

beta_tract_raw_cols = [col for col in combined_df.columns if col.startswith("beta_tract_raw")]
beta_df = combined_df[beta_tract_raw_cols]

print(f"Number of beta_tract_raw columns: {len(beta_tract_raw_cols)}")
beta_df.to_csv("stan_output/my_run/beta_tract_raw.csv", index=False)
