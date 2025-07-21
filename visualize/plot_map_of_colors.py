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
    beta_raw_cols = [col for col in draws_df.columns if col.startswith("beta_tract_raw.")]
    beta_raw = draws_df[beta_raw_cols].to_numpy()
     
    print(" tract_list length:", len(tract_list))
    print(" beta_raw columns:", beta_raw.shape[1])
 
    min_len = min(len(tract_list), beta_raw.shape[1])
    beta_raw = beta_raw[:, :min_len]
    tract_list = tract_list[:min_len]
   
    beta_last = -np.sum(beta_raw, axis=1).reshape(-1, 1)
    beta_full = np.hstack([beta_raw, beta_last])
    beta_mean = beta_full.mean(axis=0)

    tract_indices = [int(col.split(".")[-1]) for col in beta_raw_cols]
    tract_ids = [tract_list[i] for i in tract_indices if i < len(tract_list)]
    beta_mean = beta_mean[:len(tract_ids)] 

    print("Max index from beta_tract_raw:", max(tract_indices))
    print("Length of tract_list:", len(tract_list))

    beta_df = pd.DataFrame({
        "GEOID": tract_ids,
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
