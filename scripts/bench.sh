#!/bin/bash
mkdir -p ./benches/bench_data  
cd ./benches/bench_data 
if [ ! -f "8dc58605f9dd484295c7d065694cdc0f_0.geojson" ]
  then 
    echo "Downloading geojson benchmark data..."
    wget https://opendata.arcgis.com/datasets/8dc58605f9dd484295c7d065694cdc0f_0.geojson 
  else
    echo "Benchmark data already downloaded"
fi    
if [ ! -f "taxi_zones_4326.parquet" ]
  then 
    echo "Downloading parquet benchmark data..."
    wget https://data.source.coop/cholmes/nyc-taxi-zones/taxi_zones_4326.parquet 
  else
    echo "Parquet Benchmark data already downloaded"
fi    


cd ../
pip install -r requirements.txt
python generate_data.py
cd ../
echo "Running base benchmarks..."
cargo bench --bench rtree  
echo "Running benchmarks with rayon feature..."
cargo bench --bench rtree --features rayon
cd ./benches
python rtree.py
