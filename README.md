# Urban Functional Use Delineation

This project delineates urban functional zones in London, UK by analyzing Point of Interest (POI) data extracted from OpenStreetMap.

It generates vector representations of POI categories (e.g. cafes, salons, bus stops) using word2vec architecture applied on the spatial context from neighbouring POIs. These vectors encapsulate semantic relationships between different urban functions.

The POI vectors are then clustered using agglomerative clustering into 9 functional zones, reflecting the division of functions from the dense city center to the lower density outer borders.

## Methodology

- Extract POI data for London from OpenStreetMap 
- For each POI, capture spatial context by finding k-nearest neighbours  
- Generate vector for each POI using word2vec skip-gram model on neighbouring POI categories
- Cluster POI vectors using agglomerative clustering into functional zones
- Analyze and map the functional regions

## Requirements

- Python 3.6+
- Geopandas, Shapely
- Gensim, Sklearn, Matplotlib
