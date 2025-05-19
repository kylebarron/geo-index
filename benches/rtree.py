import geopandas as gpd
import numpy as np
import shapely
from geoindex_rs import rtree as rt
import timeit
import pyogrio
import requests
def load_data():
    path = "./bench_data/nz-building-outlines.parquet"
    gdf = gpd.read_parquet(path)

    bounds = gdf.bounds
    print(bounds)
    return bounds


def construct_wsg84_tree(bounds):
    builder = rt.RTreeBuilder(bounds.shape[0])
    min_x= np.array(bounds["minx"].values)
    min_y=np.array(bounds["miny"].values)
    max_x=np.array(bounds["maxx"].values)
    max_y=np.array(bounds["maxy"].values)
    builder.add(min_x, min_y, max_x, max_y)
    return builder.finish()

def construct_shapely_tree():
    path = "./bench_data/taxi_zones_4326.parquet"
    gdf = gpd.read_parquet(path)
    tree = shapely.SRTree(gdf.geometry)
    return tree


if __name__ == "__main__":
    bounds = load_data()

    print(timeit.timeit(stmt='construct_wsg84_tree(bounds)', number=100, globals=globals()))





