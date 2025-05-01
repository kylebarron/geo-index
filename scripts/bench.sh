#!/bin/bash
mkdir -p ./benches/bench_data  
cd ./benches/bench_data 
if [ ! -f "Utah.geojson.zip" ]
  then 
    echo "Downloading geojson benchmark data..."
    wget https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/Utah.geojson.zip 
  else
    echo "Benchmark data already downloaded"
fi  
if [ ! -f "Utah.geojson" ]
  then 
    echo "Unzipping Utah.geojson.zip.."
    unzip Utah.geojson.zip
  else 
    echo "Utah.geojson already unzipped "
fi

if [ ! -f "taxi_zones_4326.parquet" ]
  then 
    echo "Downloading parquet benchmark data..."
    wget https://data.source.coop/cholmes/nyc-taxi-zones/taxi_zones_4326.parquet 
  else
    echo "Parquet Benchmark data already downloaded"
fi    


cd ../
uv venv 
uv pip install -r pyproject.toml  
uv run generate_data.py
cd ../
echo "Running base benchmarks..."
cargo bench --bench rtree  
echo "Running benchmarks with rayon feature..."
cargo bench --bench rtree --features rayon
cd ./benches
uv run rtree.py
