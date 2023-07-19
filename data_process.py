import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
import mapclassify
from shapely.geometry import Point


def calculate_freq_pie_chart(df, years=None):
    df = df.drop(columns=['Id'])
    df[df > 1] = 1
    zeros_count = df.eq(0).sum().sum()
    ones_count = df.eq(1).sum().sum()
    # Create a pie chart
    if len(years) > 0:
        fig_freq = px.pie(
            values=[zeros_count, ones_count],
            names=['Not Sampled', 'Sampled'],
            title=f'Monthly sampling frequency by {years}')
    else:
        fig_freq = px.pie(
            values=[zeros_count, ones_count],
            names=['Not Sampled', 'Sampled'],
            title='Monthly sampling frequency by all years')
    return fig_freq


def calculate_invalid_pie_chart(df, years=None):
    invalid = ((df['Chlorine'].lt(1.5) | df['Chlorine'].gt(3)) | (df['Coliforms'].gt(10)) |
               (df['pH'].lt(7) | df['pH'].gt(8)) | (df['Pseudomonas'].gt(0)) |
               df['Turbidity'].gt(1) | df['Staphylococcus'].gt(2))
    df['Invalid'] = invalid
    # make pie chart of invalid data
    if len(years) > 0:
        fig = px.pie(df, names='Invalid', title=f'Invalid data by {years}')
    else:
        fig = px.pie(df, names='Invalid', title='Invalid data by all years')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    return fig


# def filter_by_year(df, start_year, end_year):
#     df['Date'] = pd.to_datetime(df['Date'])
#     df['Month_Year'] = df['Date'].dt.strftime('%m/%y')
#     df['Month_Year'] = pd.to_datetime(df['Month_Year'], format='%m/%y')
#     df['year'] = df['Date'].dt.year
#     filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
#     return df

#
# def mesurments_per_month(df):
#     new_df = df.copy()
#     new_df['Month_Year'] = new_df['Date'].dt.strftime('%m/%y')
#     new_df['Month_Year'] = pd.to_datetime(new_df['Month_Year'], format='%m/%y')
#     new_df['year'] = new_df['Date'].dt.year
#     new_df = new_df.sort_values(by='Month_Year')
#     new_df = new_df.groupby(['Id', 'Month_Year']).size().unstack(fill_value=0)
#     return new_df
#
#
# def calculate_count_ratio(df):
#     df["Test_percentages"] = df[df >= 1.0].count(axis=1) / (df.shape[1] - 1)
#     return df
#
#
# def map_coordinates(df, count_df):
#     coordinates_dict = dict(zip(df['Id'], df[['Latitude', 'Longitude']].values))
#     count_df['Latitude'] = count_df.index.map(coordinates_dict).str[0]
#     count_df['Longitude'] = count_df.index.map(coordinates_dict).str[1]
#     return count_df
#
#
# def aggrigate_to_mikveh(df):
#     df['Id_Location'] = pd.factorize(df[['Longitude', 'Latitude']].apply(tuple, axis=1))[0]
#     return df
#
#
# def calculate_Samples_Percentage_per_mikveh(df):
#     df = df.dropna()
#     group_mean = df.groupby('Id_Location')['Test_percentages'].transform('mean')
#     df['Mean_Samples_Percentage'] = df.apply(
#         lambda row: row['Test_percentages'] if pd.notna(row['Test_percentages']) and row['Test_percentages'] >= 0 else
#         row['Mean_Samples_Percentage'], axis=1)
#     df['Mean_Samples_Percentage'] = group_mean
#     return df
#
#
# def create_map_freq(df):
#     count_df = mesurments_per_month(df)
#     count_df = calculate_count_ratio(count_df)
#     count_df = map_coordinates(df, count_df)
#     count_df = aggrigate_to_mikveh(count_df)
#     count_df = calculate_Samples_Percentage_per_mikveh(count_df)
#     selected_columns = ['Latitude', 'Longitude', 'Id_Location', 'Mean_Samples_Percentage', 'Test_percentages']
#     df_to_gdf = count_df[selected_columns].copy()
#     df_to_gdf = df_to_gdf.reset_index()
#     geometry = [Point(xy) for xy in zip(df_to_gdf['Longitude'], df_to_gdf['Latitude'])]
#     gdf = gpd.GeoDataFrame(df_to_gdf, geometry=geometry)
#     gdf.crs = 'EPSG:4326'
#     m = gdf.explore(
#         column="Mean_Samples_Percentage",
#         popup=True,
#         scheme="EqualInterval",
#         cmap="RdYlGn",
#         marker_kwds=dict(radius=5),
#         legend=True,  # Set legend to False to customize it manually
#         legend_kwds=dict(colorbar=True),
#         k=8,
#         # popup=["Mean_Samples_Percentage", "Id"],
#     )
#     return m
