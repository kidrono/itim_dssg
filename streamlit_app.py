import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ITIM", page_icon=":computer:", layout="wide")
df = pd.read_csv("files_with_date.csv")

with st.container():
    st.title("ITIM")
    st.write("ITIM is a tool to help you manage your time. It is based on the Eisenhower Matrix, which categorizes tasks based on their urgency and importance. The matrix is divided into four quadrants, which are labeled as follows:")
    st.write("1. Important and Urgent")
    st.write("2. Important and Not Urgent")
    st.write("3. Not Important and Urgent")
    st.write("4. Not Important and Not Urgent")
    st.write("The goal of ITIM is to help you prioritize your tasks and manage your time more effectively. To get started, enter your tasks in the table below. Then, click the 'Submit' button to see your tasks organized by quadrant.")
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
fig = px.pie(df, names='Invalid')
# fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

st.sidebar.title("ITIM")
years = st.sidebar.multiselect("Select specific years", df['Date'].dt.year.unique())
if len(years) > 0:
    df = df[df['Date'].dt.year.isin(years)]
    fig = px.pie(df, names='Invalid')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
st.sidebar.write("---")


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df[['Id', 'Date', 'Invalid']])
    with col2:
        st.plotly_chart(fig)
    st.write("---")





