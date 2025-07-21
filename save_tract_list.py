import pandas as pd
import geopandas as gpd
import pickle
from shapely.geometry import Point

# Load raw data
df = pd.read_csv("data/flood_311.csv", encoding='latin1', on_bad_lines='warn', low_memory=False)
df.columns = df.columns.str.lower()
df = df.dropna(subset=["latitude", "longitude"])

# Build geometry and project to EPSG:4326
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Load shapefile and ensure same CRS
nyc_shape = gpd.read_file("data/cb_2018_36_tract_500k.shp")
nyc_shape = nyc_shape.to_crs("EPSG:4326")

# Spatial join to get GEOID
joined = gpd.sjoin(gdf, nyc_shape, how="inner", predicate="within")

# Extract tract list
joined["census_tract"] = joined["GEOID"].astype(str).str[:12]
tract_list = sorted(joined["census_tract"].unique())

# Save
with open("data/tract_list.pkl", "wb") as f:
    pickle.dump(tract_list, f)

print("✅ tract_list.pkl saved to data/")

