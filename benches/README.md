If the download link is dead, find the most recent download link here: https://github.com/microsoft/USBuildingFootprints

```
wget https://usbuildingdata.blob.core.windows.net/usbuildings-v2/Utah.geojson.zip
pip install -U geopandas pyogrio
python generate_data.py
```
