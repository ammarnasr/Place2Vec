import matplotlib
import folium
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple([int(255*val) for val in rgb[:3]])

def full_sequence_map(full_sequence_gdf, max_distance = 200):

    m = folium.Map(location=[full_sequence_gdf['latitude'].mean(), full_sequence_gdf['longitude'].mean()], zoom_start=15)
    cmap = matplotlib.cm.get_cmap('tab20b')

    m = full_sequence_gdf.explore(column='seq_id', m=m, marker_kwds={'radius': 5, 'fill': True}, style_kwds={'fillOpacity': 1}, cmap=cmap, legend=False)

    center_points = full_sequence_gdf[full_sequence_gdf['is_center']]
    for i, center_point in center_points.iterrows():
        center_coords = [center_point['latitude'], center_point['longitude']]
        folium.Circle(
            center_coords,
            radius=max_distance,
            fill=True,
            fill_color=rgb_to_hex(cmap.colors[i%20]),
            fill_opacity=0.05,
            stroke=True,
            color=rgb_to_hex(cmap.colors[i%20]),
            weight=0.75 
            ).add_to(m)


    #add line from center to each context point
    for i, center_point in center_points.iterrows():
        center_coords = [center_point['latitude'], center_point['longitude']]
        for j, context_point in full_sequence_gdf[full_sequence_gdf['seq_id'] == center_point['seq_id']].iterrows():
            context_coords = [context_point['latitude'], context_point['longitude']]
            folium.PolyLine(
                [center_coords, context_coords],
                color=rgb_to_hex(cmap.colors[i%20]),
                weight=1,
                opacity=0.75
                ).add_to(m)
            
    return m

def sample_sequence_map(full_sequence_gdf, max_distance=200, sample_seq_id = '129'):

    sample_sequence_gdf = full_sequence_gdf[full_sequence_gdf['seq_id'] == sample_seq_id]
    rest_sequence_gdf = full_sequence_gdf[full_sequence_gdf['seq_id'] != sample_seq_id]
    m = folium.Map(location=[sample_sequence_gdf['latitude'].mean(), sample_sequence_gdf['longitude'].mean()], zoom_start=20)


    m = rest_sequence_gdf.explore(m=m, marker_kwds={'radius': 5, 'fill': True}, style_kwds={'fillOpacity': 1}, color='red', legend_kwds={'name': 'Rest of the Points'}, legend=True)
    m = sample_sequence_gdf.explore(m=m, marker_kwds={'radius': 5, 'fill': True}, style_kwds={'fillOpacity': 1}, color='blue', legend_kwds={'name': 'Sample Sequence'}, legend=True)


    center_points = sample_sequence_gdf[sample_sequence_gdf['is_center']]
    for i, center_point in center_points.iterrows():
        center_coords = [center_point['latitude'], center_point['longitude']]
        circle = folium.Circle(
            center_coords,
            radius=max_distance,
            fill=True,
            fill_color='blue',
            fill_opacity=0.05,
            stroke=True,
            color='blue',
            weight=0.75 )
        circle.add_to(m)
        
    for i, center_point in center_points.iterrows():
        center_coords = [center_point['latitude'], center_point['longitude']]
        for j, context_point in sample_sequence_gdf[sample_sequence_gdf['seq_id'] == center_point['seq_id']].iterrows():
            context_coords = [context_point['latitude'], context_point['longitude']]
            folium.PolyLine(
                [center_coords, context_coords],
                color='blue',
                weight=1,
                opacity=0.75
                ).add_to(m)
            

    #draw line from center to the edge of the circle
    point_on_circle = [center_coords[0] + max_distance*0.000008983, center_coords[1]]
    folium.PolyLine(
        [center_coords, point_on_circle],
        color='black',
        weight=10,
        opacity=0.75
        ).add_to(m)

    #write the distance on the line
    folium.Marker(
        point_on_circle,
        icon=folium.DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 30pt; color: black;">200m</div>'
        )
    ).add_to(m)

    return m