import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ITIM", page_icon=":computer:", layout="wide")
df = pd.read_csv("files_with_date.csv")
df_freq = pd.read_csv("for_romy.csv")
df_freq = df_freq.drop(columns=['Id'])
df_freq[df_freq > 1] = 1
zeros_count = df_freq.eq(0).sum().sum()
ones_count = df_freq.eq(1).sum().sum()

# Create a pie chart
fig_freq = px.pie(
    values=[zeros_count, ones_count],
    names=['Not Sampled', 'Sampled'],
    title='Monthly sampling frequency',
)



with st.container():
    st.title("Mikveh Data Analysis for ITIM")
    st.write("[ITIM](https://www.itim.org.il/) is the leading advocacy organization working to build a Jewish and democratic Israel in which all Jews can lead full Jewish lives.")
    st.write("On this Dashboard, we analyze data from the the Health Ministry montly Mikveh samples to help ITIM find the mikvaot that are not sampled enough or have invalid data.")
    st.write("This project has been developed by the Data Science for Social Good program at the Hebrew University of Jerusalem.")
    st.write("Project Members: Romy Bauch, Oshri Fatkiev, Gili Kurtser-Gilead, and Omer Kidron")
    # upload new csv file
    uploaded_file = st.file_uploader("Upload a new csv file to update the data.")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("---")
        st.write("Data uploaded successfully!")
    st.write("---")


invalid = ((df['Chlorine'].lt(1.5) | df['Chlorine'].gt(3)) | (df['Coliforms'].gt(10)) |
        (df['pH'].lt(7) | df['pH'].gt(8)) | (df['Pseudomonas'].gt(0)) |
        df['Turbidity'].gt(1) | df['Staphylococcus'].gt(2))
df['Invalid'] = invalid
df['Date'] = pd.to_datetime(df['Date'])
# make pie chart of invalid data
fig = px.pie(df, names='Invalid', title='Invalid data by all years')
# fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

st.sidebar.title("ITIM")
years = st.sidebar.multiselect("Select specific years", df['Date'].dt.year.unique())
if len(years) > 0:
    df = df[df['Date'].dt.year.isin(years)]
    fig = px.pie(df, names='Invalid', title=f'Invalid data by {years}')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
st.sidebar.write("---")


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_freq)
    with col2:
        st.plotly_chart(fig)
    st.write("---")


with st.container():
    df_analyt = df[['Id', 'Chlorine', 'Coliforms', 'pH', 'Pseudomonas',
        'Turbidity', 'Staphylococcus']].copy().round(1)
    df_analyt = df_analyt.sort_values(by=['Id'])
    st.write(df_analyt)
    st.write("---")





