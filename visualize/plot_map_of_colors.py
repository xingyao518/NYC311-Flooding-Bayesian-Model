import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_map_of_colors(
    shapefile_path,
    beta_tract_path,
    tract_list,
    output_path="tract_posterior_map.png",
    title="Posterior Mean of Beta_Tract"
):
    nyc_shape = gpd.read_file(shapefile_path)
    nyc_shape = nyc_shape.to_crs("EPSG:4326")

    draws_df = pd.read_csv(beta_tract_path)
    beta_raw_cols = [col for col in draws_df.columns if col.startswith("beta_tract_raw[")]
    beta_raw = draws_df[beta_raw_cols].to_numpy()
    beta_last = -np.sum(beta_raw, axis=1).reshape(-1, 1)
    beta_full = np.hstack([beta_raw, beta_last])
    beta_mean = beta_full.mean(axis=0)

    beta_df = pd.DataFrame({
        "GEOID": tract_list,
        "beta": beta_mean
    })

    nyc_shape["GEOID"] = nyc_shape["GEOID"].astype(str).str[:12]
    merged = nyc_shape.merge(beta_df, on="GEOID", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(
        column="beta",
        cmap="RdYlBu_r",
        linewidth=0.5,
        edgecolor="gray",
        legend=True,
        ax=ax
    )
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Map saved to: {output_path}")
    plt.show()
