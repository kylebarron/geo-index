import geopandas as gpd
import shapely


gdf = gpd.read_file("Utah.geojson.zip", engine="pyogrio")
bounds = shapely.bounds(gdf.geometry)
print(bounds.shape)
buf = bounds.tobytes("C")
with open("bounds.raw", "wb") as f:
    f.write(buf)
