# Generate Data for Running Benchmarks

## For running real_world_case.rs

You need access to `s3://wherobots-benchmark-prod` S3 bucket to proceed.

```
mkdir -p data
aws s3 cp --recursive s3://wherobots-benchmark-prod/data/geo-index data
cd data && python ../convert_csv_to_raw.py
```

## For runnnig rtree.rs

```
wget https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/Utah.geojson.zip
pip install -U geopandas pyogrio
python generate_data.py
```
