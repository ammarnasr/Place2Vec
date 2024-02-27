from data_handler import  get_gdf, filter_by_roads_network, filter_points_within_distance, accessible_pois, get_full_sequence_gdf
import os
import pandas as pd
import geopandas as gpd


def get_poi_sequences(locality = 'Exeter', min_distance = 50, max_distance = 200, max_sequence_size = 200, force_recreate = False):
    dataset_parquet_path = f'data/{locality}_min_{min_distance}_max_{max_distance}_max_sequence_{max_sequence_size}.parquet'
    if os.path.exists(dataset_parquet_path) and not force_recreate:
        print(f'Loading {dataset_parquet_path}')
        full_sequence_gdf = gpd.read_parquet(dataset_parquet_path)
        print(f'Loaded {dataset_parquet_path} with shape {full_sequence_gdf.shape}')
    else:
        gdf = get_gdf()
        gdf = gdf[gdf['locality'] == locality]
        gdf = gdf.reset_index(drop=True)
        seed_points = filter_by_roads_network(gdf)
        seed_points = filter_points_within_distance(seed_points, min_distance)

        ordered_sequences, _ = accessible_pois(gdf, seed_points, max_distance)
        ordered_sequences = ordered_sequences.sort_values(by='size', ascending=False).reset_index(drop=True)

        full_sequence_gdf = get_full_sequence_gdf(gdf, ordered_sequences, max_sequence_size = max_sequence_size)
        full_sequence_gdf = full_sequence_gdf.sort_values(by='size', ascending=False).reset_index(drop=True)
        full_sequence_gdf = full_sequence_gdf[[
            'group', 'category', 'pointx_class', 'name','unique_reference_number', 'distance', 'postcode' , 'administrative_boundary','size', 'is_center', 'seq_id', 'category_code', 'group_code', 'pointx_class_code', 'longitude', 'latitude', 'geometry'
            ]]
        full_sequence_gdf['gr_cat_class'] = full_sequence_gdf.apply(gr_cat_class, axis=1)
        full_sequence_gdf['gr_cat_class_code'] = full_sequence_gdf.apply(gr_cat_class_code, axis=1)


        full_sequence_gdf.to_parquet(dataset_parquet_path)
        print(f'Created {dataset_parquet_path} with shape {full_sequence_gdf.shape} and nunique urns {full_sequence_gdf["unique_reference_number"].nunique()}')
    return full_sequence_gdf


def cat_class(row):
    category = row['category']
    pointx_class = row['pointx_class']
    cat_class = f'{category}_{pointx_class}'
    cat_class = cat_class.replace(' ', '_')
    return cat_class

def cat_class_code(row):
    category_code = row['category_code']
    pointx_class_code = row['pointx_class_code']
    cat_class_code = f'{category_code}-{pointx_class_code}'
    return cat_class_code


def gr_cat_class(row):
    group = row['group']
    category = row['category']
    pointx_class = row['pointx_class']
    gr_cat_class = f'{group}__{category}__{pointx_class}'
    gr_cat_class = gr_cat_class.replace(' ', '_')
    return gr_cat_class

def gr_cat_class_code(row):
    group_code = row['group_code']
    category_code = row['category_code']
    pointx_class_code = row['pointx_class_code']
    gr_cat_class_code = f'{group_code}-{category_code}-{pointx_class_code}'
    return gr_cat_class_code

