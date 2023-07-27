import streamlit as st
import eda
import prediction
from streamlit_option_menu import option_menu


st.sidebar.header("HaiMeds Customer Churn Prediction")


with st.sidebar:
    st.write("Ediashta Revindra - FTDS-020")
    selected = option_menu(
        "Menu",
        [
            "Distribution",
            "Correlation Matrix",
            "Regression",
        ],
        icons=["bar-chart", "link-45deg", "code-square"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Distribution":
    eda.distribution()
elif selected == "Correlation Matrix":
    eda.corr_matrix()
elif selected == "Regression":
    prediction.predict()
