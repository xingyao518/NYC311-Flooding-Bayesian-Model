import pickle
from visualize.plot_map_of_colors import plot_map_of_colors

with open("data/tract_list.pkl", "rb") as f:
    tract_list = pickle.load(f)

plot_map_of_colors(
    shapefile_path="data/cb_2018_36_tract_500k.shp",
    beta_tract_path="combined_output.csv",
    tract_list=tract_list,
    output_path="beta_tract_map.png",
    title="Posterior Mean of Beta_tract"
)
