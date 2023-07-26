import pandas as pd
import streamlit as st
import plotly.express as px
import data_process as dp
import numpy as np
from streamlit_folium import st_folium

st.set_page_config(page_title="ITIM", page_icon=":computer:", layout="wide")
df = pd.read_csv("data_to_dashboard.csv")
df['Date'] = pd.to_datetime(df['Date'])
with st.container():
    st.title("Mikveh Data Analysis for ITIM")
    st.write(
        "[ITIM](https://www.itim.org.il/) is the leading advocacy organization working to build a Jewish and democratic Israel in which all Jews can lead full Jewish lives.")
    st.write(
        "On this Dashboard, we analyze data from the the Health Ministry montly Mikveh samples to help ITIM find the mikvaot that are not sampled enough or have invalid data.")

    # upload new csv file
    uploaded_file = st.file_uploader("Upload a new csv file to update the data.")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("---")
        st.write("Data uploaded successfully!")
    st.write("---")

st.sidebar.title("ITIM")
# Initialize years and sites with default values if they are not already set in session_state
rows_per_page = 50
current_page = st.session_state.get("current_page", 0)
st.session_state.years = st.session_state.get("years", [])
st.session_state.sites = st.session_state.get("sites", [])

years = st.sidebar.multiselect("Select specific years", df['Date'].dt.year.unique())
cities = st.sidebar.multiselect("Select specific cities", df['Settlement'].unique())
sites = st.sidebar.multiselect("Select specific Mikveh code", df['Id'].unique())

# Check if the selections have changed, and reset the current_page to 0 if they have
if st.session_state.years != years or st.session_state.sites != sites:
    current_page = 0

# Update session_state with the current selections
st.session_state.years = years
st.session_state.sites = sites
if len(years) > 0:
    df = df[df['Date'].dt.year.isin(years)]
if len(sites) > 0:
    df = df[df['Id'].isin(sites)]
if len(cities) > 0:
    df = df[df['Settlement'].isin(cities)]
df_freq = dp.mesurments_per_month(df)
fig_freq = dp.calculate_freq_pie_chart(df_freq, years)
fig_invalid = dp.calculate_invalid_pie_chart(df, years)
st.sidebar.write("---")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_freq)
        st.write("The pie chart shows the percentage of sites that were sampled in each month. Each site should be sampled at least once a month.")
    with col2:
        st.plotly_chart(fig_invalid)
        st.write("The pie chart shows the percentage of sites that were invalid in each month. Invalid means that one or more of the parameters were out of range.")
    st.write("---")

with st.container():
    st.write("This map can be changed using the 'Select Map Type' button, and it displays the percentage, indicated by colors, for each Mikveh based on the frequency of sample taking and the validity of the samples collected. In the table next to the map, you can view the Mikveh ID, its settlement, as well as the frequency and validation percentages.")
    selected_map = st.selectbox("Select map type", ['Frequency', 'Validity'])
    col1, col2 = st.columns(2)
    with col1:
        try:
            if selected_map == 'Frequency':
                site_map = dp.create_map_freq(df)
            else:
                site_map = dp.create_map_invalid(df)
            st_folium(site_map, width=400, height=500)
        except:
            st.write("Not enough sites for this selection. Please add more if you want to see the map.")
    with col2:
        selected_df = dp.calculate_freq_and_invalid_df(df)
        # show only the relevant columns
        # selected_df = selected_df[['Id', selected_map]]
        st.dataframe(selected_df, hide_index=True, height=500)
    st.write("---")

# third part
# Define a dictionary with column names as keys and their respective allowed ranges as values
column_ranges = {
    'Chlorine': (1.5, 3),
    'Coliforms': (0, 10),
    'Pseudomonas': (0, 0),
    'Staphylococcus': (0, 2),
    'Turbidity': (0, 1),
    'pH': (7, 8)
}


# Define a function to apply background colors based on column-specific ranges

def color_background(val, column_name):
    if type(val) != str:
        if column_name in column_ranges:
            allowed_range = column_ranges[column_name]
            if val < allowed_range[0] or val > allowed_range[1]:
                return 'background-color: #f75040'
            elif allowed_range[0] <= val <= allowed_range[1]:
                return 'background-color: '
        return 'background-color: '  # Default color for other values or unknown columns


# Apply the styling function to the entire DataFrame using the Styler
styled_df = df.style.apply(lambda x: [color_background(val, column_name) for val, column_name in zip(x, x.index)],
                           axis=1)

df_analyts = df.copy()
df_analyts['Chlorine_invalid'] = df_analyts['Chlorine'].lt(1.5) | df_analyts['Chlorine'].gt(3)
df_analyts['Coliforms_invalid'] = df_analyts['Coliforms'].gt(10)
df_analyts['pH_invalid'] = df_analyts['pH'].lt(7) | df_analyts['pH'].gt(8)
df_analyts['Pseudomonas_invalid'] = df_analyts['Pseudomonas'].gt(0)
df_analyts['Turbidity_invalid'] = df_analyts['Turbidity'].gt(1)
df_analyts['Staphylococcus_invalid'] = df_analyts['Staphylococcus'].gt(2)

df_chart = df_analyts[['Chlorine_invalid', 'Coliforms_invalid', 'Chlorine', 'Coliforms',
                       'pH_invalid', 'pH', 'Pseudomonas_invalid', 'Pseudomonas',
                       'Turbidity_invalid', 'Turbidity', 'Staphylococcus_invalid', 'Staphylococcus']].copy()

# Define the column names and corresponding conditions
columns = ['Chlorine', 'Coliforms', 'pH', 'Pseudomonas', 'Turbidity', 'Staphylococcus']
invalid_counts = []
notnull_counts = []

# Calculate counts for each column
for column in columns:
    invalid_count = df_chart[(df_chart[column + '_invalid'] == 1) & (~pd.isna(df_chart[column]))].shape[0]
    notnull_count = df_chart[column].notna().sum()
    invalid_counts.append(invalid_count)
    notnull_counts.append(notnull_count)

# Create a new DataFrame with the counts
counts_df = pd.DataFrame({
    'Analyt': columns,
    'Invalid samples': invalid_counts,
    'All samples': notnull_counts
})

# Create the grouped bar chart using Plotly Express
fig_analyt = px.bar(counts_df, x='Analyt', y=['Invalid samples', 'All samples'], barmode='overlay')
fig_analyt.update_layout(width=500)

# rows_per_page = 50
# current_page = st.session_state.get("current_page", 0)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        df_analyt = df[
            ['Id', 'Chlorine', 'Coliforms', 'pH', 'Pseudomonas', 'Turbidity',
             'Staphylococcus']].copy().round(1)
        df_analyt = df_analyt.sort_values(by=['Id'])
        df_analyt = df_analyt.fillna('Unchecked')
        num_rows = len(df_analyt)

        start_index = current_page * rows_per_page
        end_index = min(start_index + rows_per_page, num_rows)

        styled_df = df_analyt.iloc[start_index:end_index, :].style.apply(
            lambda x: [color_background(val, column_name) for val, column_name
                       in zip(x, x.index)], axis=1
        )

        st.dataframe(styled_df.format(na_rep='Unchecked', precision=1), hide_index=True)

        if current_page > 0:
            if st.button("Previous"):
                current_page -= 1
        if end_index < num_rows:
            if st.button("Next"):
                current_page += 1
        st.write("This table presents the Mikveh ID and sample dates, along with the measurement values for all analyts collected in each sample")
    with col2:
        st.plotly_chart(fig_analyt)
        st.write("This graph displays the number of invalid samples for each analyt, relative to the total number of samples they have taken.")

st.session_state["current_page"] = current_page


# histogram fig clorine/ph:
def create_fig_hist_2_bounderies(analyt, l, b1, b2, u, r1, r2):
    filtered_df = df[(df[analyt] != 0) & (~df[analyt].isna())]
    filtered_df["color"] = np.select(
        [filtered_df[analyt].lt(l), filtered_df[analyt].lt(b1),
         filtered_df[analyt].lt(b2), filtered_df[analyt].lt(u)],
        ["red", "orange", "green", "orange"],
        "red",
    )
    fig_hist_analyt = px.histogram(filtered_df, x=analyt,
                                   color='color',
                                   # marginal='histogram',
                                   barmode='overlay',
                                   color_discrete_map={
                                       "red": "red",
                                       "orange": "orange",
                                       "green": "green",
                                   })
    fig_hist_analyt.update_layout(
        xaxis=dict(range=[r1, r2]),  # Set x-axis limit to 0-6
        xaxis_tickmode='linear',
        xaxis_dtick=0.1,  # Set tick intervals as 0.2
        xaxis_tickformat='.1f',  # Set tick labels to 1 decimal place
        title='Histogram of ' + analyt + ' values samples',
        xaxis_title=analyt,
        yaxis_title='Frequency',
        bargap=0.2,  # Adjust gap between bars
        showlegend=False
    )
    return fig_hist_analyt


def create_fig_hist_1_bounderies(analyt, l, b1, r1, r2):
    filtered_df = df[(~df[analyt].isna())]
    filtered_df["color"] = np.select(
        [filtered_df[analyt].lt(l), filtered_df[analyt].lt(b1)],
        ["green", "orange"],
        "red",
    )
    fig_hist_analyt = px.histogram(filtered_df, x=analyt,
                                   color='color',
                                   # marginal='histogram',
                                   barmode='overlay',
                                   color_discrete_map={
                                       "red": "red",
                                       "orange": "orange",
                                       "green": "green",
                                   })
    fig_hist_analyt.update_layout(
        xaxis=dict(range=[r1, r2]),  # Set x-axis limit to r1-r2
        xaxis_tickmode='linear',
        xaxis_dtick=0.1,  # Set tick intervals as 0.1
        xaxis_tickformat='.1f',  # Set tick labels to 1 decimal place
        title='Histogram of ' + analyt + ' values samples',
        xaxis_title=analyt,
        yaxis_title='Frequency',
        bargap=0.1,  # Adjust gap between bars
        showlegend=False,
        barmode='overlay',  # Set the barmode to overlay
        autosize=False
    )
    return fig_hist_analyt


with st.container():
    selected_analyt = st.selectbox("Select an analyt", ['pH', 'Coliforms',
                                                        'Chlorine',
                                                        'Pseudomonas',
                                                        'Staphylococcus',
                                                        'Turbidity'])
    if (selected_analyt == 'pH'):
        st.plotly_chart(create_fig_hist_2_bounderies('pH', 7, 7.1, 7.9, 8, 5, 10))
    if (selected_analyt == 'Chlorine'):
        st.plotly_chart(
            create_fig_hist_2_bounderies('Chlorine', 1.5, 1.6, 2.9, 3, 0, 5))
    if (selected_analyt == 'Coliforms'):
        st.plotly_chart(create_fig_hist_1_bounderies('Coliforms', 10, 10.1, 0, 15))
    if (selected_analyt == 'Pseudomonas'):
        st.plotly_chart(create_fig_hist_1_bounderies('Pseudomonas', 1, 1.1, 0, 10))
    if (selected_analyt == 'Staphylococcus'):
        st.plotly_chart(
            create_fig_hist_1_bounderies('Staphylococcus', 2, 2.1, 0, 10))
    if (selected_analyt == 'Turbidity'):
        st.plotly_chart(
            create_fig_hist_1_bounderies('Turbidity', 1, 1.1, 0, 10))
    st.write("---")


def check_invalid(row):
    return 1 if row.any() else 0

# Add the 'is_invalid' column
df_analyts['is_invalid'] = df_analyts[['Chlorine_invalid', 'Coliforms_invalid',
                                       'pH_invalid','Pseudomonas_invalid',
                                       'Turbidity_invalid',
                                       'Staphylococcus_invalid']].apply(check_invalid, axis=1)

# Extract the year and month from the 'Date' column
df_analyts['Year'] = df_analyts['Date'].dt.year
df_analyts['Month'] = df_analyts['Date'].dt.month

# Group by Year and Month to get the counts for each combination
year_month_counts = df_analyts.groupby(['Year', 'Month']).agg(
    number_of_samples=('is_invalid', 'count'),
    num_of_unvalid_samples=('is_invalid', 'sum')
).reset_index()
# Creating a new 'date' column from 'Year' and 'Month'
year_month_counts['date'] = pd.to_datetime(year_month_counts[['Year', 'Month']]
                                           .assign(day=1))
fig_invalid_date = px.bar(year_month_counts, x='date',
                          y=['num_of_unvalid_samples','number_of_samples'],
                          barmode='overlay')

fig_invalid_date.update_layout(yaxis_title='number of samples')
fig_invalid_date.update_layout(width=1000)
fig_invalid_date.update_layout(xaxis_tickangle = -45)

with st.container():
    st.plotly_chart(fig_invalid_date)

district_counts = df_analyts.groupby(['district']).agg(
    number_of_samples=('is_invalid', 'count'),
    num_of_unvalid_samples=('is_invalid', 'sum')).reset_index()

fig_invalid_district = px.bar(district_counts, x='district',
                          y=['num_of_unvalid_samples', 'number_of_samples'],
                          barmode='overlay')

# Rotate x-axis labels for better visibility
fig_invalid_district.update_layout(xaxis_tickangle=-45)

# Display the figure in Streamlit
with st.container():
    st.plotly_chart(fig_invalid_district)


with st.container():
    st.write(
        "This project has been developed as part of the [Data Science for Social Good](https://cidr.huji.ac.il/en/data-science-for-social-good/) program at the Hebrew University of Jerusalem.")
    st.write("Project members: Romy Bauch, Oshri Fatkiev, Gili Kurtser-Gilead, and Omer Kidron.")
