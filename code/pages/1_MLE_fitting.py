
import streamlit as st
from util.fetch_data_docDB_mle_fitting import fetch_mle_fitting_results

df = fetch_mle_fitting_results()
st.write(df)