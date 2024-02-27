import pandas as pd
from schema.classification_schema import get_classification_schema
import os
from pyproj import Proj, transform
import geopandas as gpd
from shapely.geometry import Point
import haversine as hs
from haversine import Unit
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_full_sequence_gdf(df, ordered_sequences, max_sequence_size = 10):
    full_sequence_gdf = pd.DataFrame()
    for i in range(len(ordered_sequences)):
        center_urn = ordered_sequences['center'].values[i]
        context_urns = ordered_sequences['pois'].values[i]
        context_distances = ordered_sequences['distances'].values[i]
        seq_id = ordered_sequences['seq_id'].values[i]
        size = ordered_sequences['size'].values[i]

        if size > max_sequence_size:
            context_urns = context_urns[:max_sequence_size - 1]

        all_urns = [center_urn] + list(context_urns)
        all_distances = [0] + list(context_distances)
        distance_urns = dict(zip(all_urns, all_distances))

        def get_distance(urn):
            return distance_urns[urn]
        
        sequence_gdf = df[df['unique_reference_number'].isin(all_urns)]
        sequence_gdf['seq_id'] = seq_id
        sequence_gdf['size'] = size
        sequence_gdf['is_center'] = sequence_gdf['unique_reference_number'] == center_urn
        sequence_gdf['distance'] = sequence_gdf['unique_reference_number'].apply(get_distance)
        


        full_sequence_gdf = pd.concat([full_sequence_gdf, sequence_gdf])
    return full_sequence_gdf




def on_roads_network(row):
    return row['street_name'] is not None

def filter_by_roads_network(gdf):
    gdf['on_roads_network'] = gdf.apply(on_roads_network, axis=1)
    return gdf[gdf['on_roads_network']]


def add_latlon_column(geom):
    return (geom.y, geom.x)

def distances_from_point(x, p):
    return hs.haversine(x, p, unit=Unit.METERS)

def filter_points_within_distance(df, distance, random_state=None, shuffle=False):
    df['latlon'] = df['geometry'].apply(add_latlon_column)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        df = df.sort_values(by='latlon').reset_index(drop=True)
    verified_df = pd.DataFrame(columns=['unique_reference_number', 'latlon'])
    unique_reference_numbers = df['unique_reference_number'].tolist()
    latlons = df['latlon'].tolist()
    tbar = tqdm(zip(unique_reference_numbers, latlons), total=len(df), desc='Number of Points Verified = 0')
    for ref, p in tbar:
        distances = verified_df['latlon'].apply(distances_from_point, p=p)
        if len(distances) > 0 and distances.min() < distance:
            continue
        verified_df = pd.concat([verified_df, pd.DataFrame({'unique_reference_number': [ref], 'latlon': [p]})])
        tbar.set_description(f'Number of Points Verified = {len(verified_df)}')
    verified_df = verified_df.reset_index(drop=True)
    verified_df = verified_df['unique_reference_number']
    df = df.merge(verified_df, on='unique_reference_number', how='inner')
    return df

def accessible_pois(src, seed_points, max_distance):
    src['latlon'] = src['geometry'].apply(add_latlon_column)
    accessible_pois_set = pd.DataFrame(columns=['centroid_unique_reference_number', 'context_unique_reference_number', 'distance'])
    for centroid_urn, centroid_latlon in tqdm(zip(seed_points['unique_reference_number'], seed_points['latlon']), total=len(seed_points)):
        distances = src['latlon'].apply(distances_from_point, args=(centroid_latlon,))
        df  = pd.DataFrame({'centroid_unique_reference_number': centroid_urn, 'context_unique_reference_number': src['unique_reference_number'], 'distance': distances})
        df = df[df['distance'] <= max_distance]
        accessible_pois_set = pd.concat([accessible_pois_set, df], axis=0, ignore_index=True)


    accessible_pois_set = accessible_pois_set[accessible_pois_set['centroid_unique_reference_number'] != accessible_pois_set['context_unique_reference_number']]
    accessible_pois_set = accessible_pois_set[accessible_pois_set['distance'] > 0]
    print(f'Number of Accessible POIs (Within {max_distance}m): {accessible_pois_set.shape[0]}')
    print(f'Number of Unique Accessible POIs (centroid): {accessible_pois_set["centroid_unique_reference_number"].nunique()}')
    print(f'Number of Unique Accessible POIs (context): {accessible_pois_set["context_unique_reference_number"].nunique()}')

    centers = accessible_pois_set['centroid_unique_reference_number'].unique()
    ordered_sequences = {}
    seq_id = '000'
    for center in tqdm(centers, desc='Generating Ordered Sequences'):
        current_group = accessible_pois_set[accessible_pois_set['centroid_unique_reference_number'] == center]
        current_group = current_group.sort_values(by='distance', ascending=True)
        pois = current_group['context_unique_reference_number'].values
        size = current_group.shape[0]
        distances = current_group['distance'].values
        ordered_sequences[center] = {'pois' : pois, 'size' : size, 'seq_id' : seq_id, 'distances': distances}
        seq_id = str(int(seq_id) + 1).zfill(3)
    ordered_sequences = pd.DataFrame(ordered_sequences).T
    ordered_sequences['center'] = ordered_sequences.index
    ordered_sequences = ordered_sequences.reset_index(drop=True)
    ordered_sequences = ordered_sequences[['center', 'size', 'pois', 'distances', 'seq_id']]
    return ordered_sequences, accessible_pois_set

def national_grid_to_lat_lon(df):
    v84 = Proj(proj="latlong",towgs84="0,0,0",ellps="WGS84")
    v36 = Proj(proj="latlong", k=0.9996012717, ellps="airy", towgs84="446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894")
    v_grid = Proj(init="world:bng")
    v_lon36, v_lat36 = v_grid(df['feature_easting'].values, df['feature_northing'].values, inverse=True)
    converted = transform(v36, v84, v_lon36, v_lat36)
    df['longitude'] = converted[0]
    df['latitude'] = converted[1]
    return df


def get_gdf(data_path = 'data/processed_pois_data.csv'):
    if os.path.exists('./data/geo_data.parquet'):
        print('Loading data from parquet file')
        gdf = gpd.read_parquet('./data/geo_data.parquet')
    else:
        if not os.path.exists(data_path):
            process_data()
        df = pd.read_csv(data_path)
        print(f'Loaded data from {data_path} with shape {df.shape}')

        df = national_grid_to_lat_lon(df)
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        print('Converted Data to Latitude and Longitude from National Grid')
        
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf = gdf.set_crs('epsg:4326')
        gdf.to_parquet('./data/geo_data.parquet')
        print('Saved GeoDataFrame to parquet file: ', './data/geo_data.parquet')
    print(f'Loaded GeoDataFrame with shape {gdf.shape} and crs {gdf.crs}')
    return gdf



def process_data(data_path = 'data/pois_data.csv'):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    cols = [
        'name',
        'brand',
        'qualifier_data',
        'qualifier_type',
        'locality',
        'street_name',
        'distance',
        'url',
        'address_detail',
        'geographic_county',
        'postcode',
        'administrative_boundary',
        'telephone_number',
        'feature_easting',
        'feature_northing',
        'pointx_classification_code',
        'unique_reference_number',
    ]
    df = df[cols]
    classification_schema = get_classification_schema()
    pointx_classification_code_dict = generate_schema(classification_schema)
    df = add_schema(df, pointx_classification_code_dict)
    processed_data_path = 'data/processed_pois_data.csv'
    df.to_csv(processed_data_path, index=False)
    print(f'Processed data and saved to {processed_data_path}')
    return df


def generate_schema(classification_schema):
    pointx_classification_code_dict = {}
    for group in classification_schema.keys():
        group_code = group[0:2]
        group_name = group[3:]
        for category in classification_schema[group]:
            category_code = category[0:2]
            category_name = category[3:]
            for pointx_class in classification_schema[group][category]:
                pointx_class_code = pointx_class[0:4]
                pointx_class_name = pointx_class[5:]
                pointx_classification_code = int(f'{group_code}{category_code}{pointx_class_code}')
                pointx_classification_code_dict[pointx_classification_code] = {
                    'group': group_name,
                    'category': category_name,
                    'pointx_class': pointx_class_name,
                    'group_code': group_code,
                    'category_code': category_code,
                    'pointx_class_code': pointx_class_code
                }
    return pointx_classification_code_dict

def add_schema(data, schema):
    group_code = []
    category_code = []
    pointx_class_code = []
    group = []
    category = []
    pointx_class = []
    for code in data['pointx_classification_code']:
        code_dict = schema[code]
        group_code.append(code_dict['group_code'])
        category_code.append(code_dict['category_code'])
        pointx_class_code.append(code_dict['pointx_class_code'])
        group.append(code_dict['group'])
        category.append(code_dict['category'])
        pointx_class.append(code_dict['pointx_class'])
    data['group_code'] = group_code
    data['category_code'] = category_code
    data['pointx_class_code'] = pointx_class_code
    data['group'] = group
    data['category'] = category
    data['pointx_class'] = pointx_class
    return data


def get_data(data_path = 'data/processed_pois_data.csv'):
    if not os.path.exists(data_path):
        print(f'No data found at {data_path}. Processing data...')
        process_data()
    df = pd.read_csv(data_path)
    print(f'Loaded data from {data_path} with shape {df.shape}')
    return df

