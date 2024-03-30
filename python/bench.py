import geopandas as gpd
import numpy as np
import shapely
from geoindex_rs import RTree

# wget https://storage.googleapis.com/open-geodata/linz-examples/nz-building-outlines.parquet
path = "nz-building-outlines.parquet"
gdf = gpd.read_parquet(path)

wgs84_gdf = gdf.to_crs("epsg:4326")
wgs84_bounds = wgs84_gdf.bounds


# %%timeit
wgs84_tree = RTree.from_separated(
    np.array(wgs84_bounds["minx"].values),
    np.array(wgs84_bounds["miny"].values),
    np.array(wgs84_bounds["maxx"].values),
    np.array(wgs84_bounds["maxy"].values),
    method="hilbert",
)

# %%timeit
shapely_tree = shapely.STRtree(wgs84_gdf.geometry)


wgs84_tree = RTree.from_separated(
    wgs84_bounds["minx"].values,
    wgs84_bounds["miny"].values,
    wgs84_bounds["maxx"].values,
    wgs84_bounds["maxy"].values,
    method="str",
)

bbox = [172.566811, -43.541619, 172.609802, -43.504971]
# %%timeit
indices = wgs84_tree.search(*bbox)
box = shapely.box(*bbox)
# %%timeit
shapely_indices = shapely_tree.query(box)

# Assert equal
len(indices)
len(shapely_indices)
assert (np.sort(indices) == np.sort(shapely_indices)).all()
