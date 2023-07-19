import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import mapclassify
from shapely.geometry import Point
import plotly.express as px


def calculate_freq_pie_chart(df, years=None):
    # df = df.drop(columns=['Id'])
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


def calculate_freq_and_invalid_df(df):
    # Group by 'Id' and extract unique months and years
    df['year_month'] = df['Date'].dt.to_period('M')
    # Count the number of unique months for each ID
    result = df.groupby('Id')['year_month'].nunique().reset_index()
    # Rename the column for clarity
    result.rename(columns={'year_month': 'unique_months'}, inplace=True)
    # Create a dictionary from the result DataFrame to map ID to its unique month count
    id_count_dict = result.set_index('Id')['unique_months'].to_dict()
    # Add the 'unique_months' column to the original DataFrame based on 'ID'
    df['unique_months'] = df['Id'].map(id_count_dict)
    # calculate all the possible months
    all_months = 0
    n_years = df['Date'].dt.year.unique()
    for year in n_years:
        year_df = df[df['Date'].dt.year == year]
        all_months += year_df['Date'].dt.month.nunique()
    df['Frequency'] = np.round(df['unique_months'] / all_months, 2)
    # Invalid measurments
    invalid_mask = (
            df['Chlorine'].lt(1.5) |
            df['Chlorine'].gt(3) |
            df['Coliforms'].gt(10) |
            df['pH'].lt(7) |
            df['pH'].gt(8) |
            df['Pseudomonas'].gt(0) |
            df['Turbidity'].gt(1) |
            df['Staphylococcus'].gt(2)
    )
    # Fill DataFrame with -1 instead of NaN
    # df.fillna(-1, inplace=True)
    df['invalid'] = 0
    df.loc[invalid_mask, 'invalid'] = 1
    invalid_counts = df[df['invalid'] == 1].groupby('Id')['invalid'].count().reset_index()
    total_counts = df.groupby('Id')['invalid'].count().reset_index()
    # Merge the valid and total counts DataFrames based on 'ID'
    counts = pd.merge(invalid_counts, total_counts, on='Id', how='outer', suffixes=('_count', '_total')).fillna(0)
    # Calculate the fraction of valid rows for each ID
    counts['Validity'] = np.round(1 - (counts['invalid_count'] / counts['invalid_total']), 2)
    # Create a dictionary from the counts DataFrame to map ID to its unique month count
    id_count_dict = counts.set_index('Id')['Validity'].to_dict()
    # Add the 'fraction_valid' column to the original DataFrame based on 'ID'
    df['Validity'] = df['Id'].map(id_count_dict)
    # The DataFrame to show in the dashboard
    selected_columns = ['Id', 'Frequency', 'Validity']
    display_df = df[selected_columns].copy()
    display_df = display_df.drop_duplicates()
    display_df = display_df.reset_index(drop=True)
    return display_df


def mesurments_per_month(df):
    df['Month_Year'] = df['Date'].dt.strftime('%m/%y')
    df['Month_Year'] = pd.to_datetime(df['Month_Year'], format='%m/%y')
    all_measurments_df = df.sort_values(by='Month_Year')
    count_df = df.groupby(['Id', 'Month_Year']).size().unstack(fill_value=0)
    return count_df


def calculate_count_ratio(df):
    df["Test_percentages"] = df[df >= 1.0].count(axis=1) / (df.shape[1] - 1)
    return df


def map_coordinates(df, count_df):
    coordinates_dict = dict(zip(df['Id'], df[['Latitude', 'Longitude']].values))
    count_df['Latitude'] = count_df.index.map(coordinates_dict).str[0]
    count_df['Longitude'] = count_df.index.map(coordinates_dict).str[1]
    return count_df


def aggrigate_to_mikveh(df):
    df['Id_Location'] = pd.factorize(df[['Longitude', 'Latitude']].apply(tuple, axis=1))[0]
    return df


def calculate_Samples_Percentage_per_mikveh(df):
    df = df.copy()
    df = df.dropna()
    group_mean = df.groupby('Id_Location')['Test_percentages'].transform('mean')
    df['Mean_Samples_Percentage'] = df.apply(
        lambda row: row['Test_percentages'] if pd.notna(row['Test_percentages']) and row['Test_percentages'] >= 0 else
        row['Mean_Samples_Percentage'], axis=1)
    df['Mean_Samples_Percentage'] = group_mean
    return df


def create_map_freq(df):
    count_df = mesurments_per_month(df)
    count_df = calculate_count_ratio(count_df)
    count_df = map_coordinates(df, count_df)
    count_df = aggrigate_to_mikveh(count_df)
    count_df = calculate_Samples_Percentage_per_mikveh(count_df)
    selected_columns = ['Latitude', 'Longitude', 'Id_Location', 'Mean_Samples_Percentage', 'Test_percentages']
    df_to_gdf = count_df[selected_columns].copy()
    df_to_gdf = df_to_gdf.reset_index()

    # Assuming 'gdf' is your GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_to_gdf['Longitude'], df_to_gdf['Latitude'])]
    gdf = gpd.GeoDataFrame(df_to_gdf, geometry=geometry)
    gdf.crs = 'EPSG:4326'

    m = gdf.explore(
        column="Mean_Samples_Percentage",
        popup=True,
        scheme="EqualInterval",
        cmap="RdYlGn",
        marker_kwds=dict(radius=5),
        legend=True,  # Set legend to False to customize it manually
        legend_kwds=dict(colorbar=True),
        k=8,
        # popup=["Mean_Samples_Percentage", "Id"],
    )
    return m


def create_validity_df(df):
    df['pH_valid'] = 0
    df['Turbidity_valid'] = 0
    df['Coliforms_valid'] = 0
    df['Pseufomonas_valid'] = 0
    df['Chlorine_valid'] = 0
    df['Staphylococcus_valid'] = 0
    parameters = ['pH', 'Coliforms', 'Chlorine', 'Pseudomonas',
                  'Staphylococcus', 'Turbidity']
    for parm in parameters:
        df[parm] = pd.to_numeric(df[parm])

    df.loc[(df['pH'].lt(7.1) | df['pH'].gt(7.9)), 'pH_valid'] = 1
    df.loc[(df['Turbidity'].gt(0.9)), 'Turbidity_valid'] = 1
    df.loc[df['Coliforms'].gt(9.9), 'Coliforms_valid'] = 1
    df.loc[df['Pseudomonas'].gt(1), 'Pseufomonas_valid'] = 1
    df.loc[(df['Chlorine'].lt(1.6) | df['Chlorine'].gt(2.9)), 'Chlorine_valid'] = 1

    df.loc[df['Staphylococcus'].gt(2.0), 'Staphylococcus_valid'] = 1
    df['is_valid'] = df[['pH_valid', 'Turbidity_valid', 'Coliforms_valid',
                         'Pseufomonas_valid', 'Chlorine_valid', 'Staphylococcus_valid']].max(axis=1)

    return df


def aggrigate_to_mikveh(df):
    df['Id_Location'] = pd.factorize(df[['Longitude', 'Latitude']].apply(tuple, axis=1))[0]
    return df


def count_invails_presentege(df):
    df['Not_valid'] = df.groupby('Id')['is_valid'].transform('sum')
    df['ID_Location_count'] = df.groupby('Id_Location')['Id_Location'].transform('count')

    # Divide the sum of 'is_valid' values by the count of occurrences for each 'ID_Location'
    df['presentege_of_non-valid_samples'] = df['Not_valid'] / df['ID_Location_count']

    # Drop the intermediate 'ID_Location_count' column if you don't need it
    df.drop('ID_Location_count', axis=1, inplace=True)
    return df


def create_map_invalid(df):
    valid_df = create_validity_df(df)
    valid_df = aggrigate_to_mikveh(valid_df)
    valid_df = count_invails_presentege(valid_df)
    selected_columns = ['Latitude', 'Longitude', 'Id_Location', 'Id', 'Not_valid', 'is_valid',
                        'presentege_of_non-valid_samples']
    df_analyt_to_gdf = valid_df[selected_columns].copy()
    df_analyt_to_gdf = df_analyt_to_gdf.dropna()
    geometry_ana = [Point(xy) for xy in zip(df_analyt_to_gdf['Longitude'], df_analyt_to_gdf['Latitude'])]
    gdf_analyt = gpd.GeoDataFrame(df_analyt_to_gdf, geometry=geometry_ana)
    gdf_analyt.crs = 'EPSG:4326'
    m = gdf_analyt.explore(
        column="presentege_of_non-valid_samples",
        popup=True,
        scheme="EqualInterval",
        cmap="RdYlGn",
        marker_kwds=dict(radius=5),
        legend=True,  # Set legend to False to customize it manually
        legend_kwds=dict(colorbar=True),
        k=8,
        # popup=["Mean_Samples_Percentage", "Id"],
    )
    return m
