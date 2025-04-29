import geopandas as gpd
import shapely


gdf = gpd.read_file("./bench_data/8dc58605f9dd484295c7d065694cdc0f_0.geojson", engine="pyogrio")
bounds = shapely.bounds(gdf.geometry)
print(bounds.shape)
buf = bounds.tobytes("C")
with open("./bench_data/bounds.raw", "wb") as f:
    f.write(buf)
