import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="Laptop Price Regression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# dataset
dataset = "https://raw.githubusercontent.com/ediashta/p2-ftds020-rmt-m1/main/churn.csv"
data = pd.read_csv(dataset)


def distribution():
    # distribution plot
    st.title("HaiMeds Customer Distribution")
    col1, col2 = st.columns(2)

    hist_plot_1 = col1.selectbox(
        "Choose Table",
        ("Age", "Last Login (Days)", "Avg. Time Spent"),
    )
    hist_plot(hist_plot_1, col1)

    hist_plot_2 = col2.selectbox(
        "Choose Table",
        ("Avg. Transaction", "Avg. Login Frequency (Days)", "Points"),
    )
    hist_plot(hist_plot_2, col2)

    col1, col2 = st.columns(2)
    bar_plot_1 = col1.selectbox(
        "Choose Table",
        ("Gender", "Region", "Membership", "Referral", "Preferred Offer", "Devices"),
    )
    bar_plot(bar_plot_1, col1)

    bar_plot_2 = col2.selectbox(
        "Choose Table",
        (
            "Internet",
            "Used Discount",
            "Offer Application Preference",
            "Past Complaint",
            "Complaint Status",
            "Feedback",
        ),
    )
    bar_plot(bar_plot_2, col2)

    st.subheader("Churn Risk Score Distribution")
    churn_score()


def corr_matrix():
    # distribution plot
    st.title("Features Correlation")
    col1, col2 = st.columns(2)

    # correlation for numerical
    fig = plt.figure(figsize=(10, 10))
    corr_matrix = data[
        [
            "age",
            "days_since_last_login",
            "avg_time_spent",
            "avg_transaction_value",
            "avg_frequency_login_days",
            "points_in_wallet",
            "churn_risk_score",
        ]
    ].corr(method="spearman")
    sns.heatmap(corr_matrix, annot=True, cmap="mako", square=True)
    col1.pyplot(fig)
    col2.write(
        "Heatmap disebelah merupakan korelasi antara data numerikal dengan Final Price sebuah laptop, data dibawah merupakan korelasi data kategorical dengan Final Price sebuah laptop"
    )
    col2.markdown("### Correlation")
    col2.markdown("* **Status** : 0.26450718170008297")
    col2.markdown("* **Brand** : 0.241996453068071")
    col2.markdown("* **Model** : 0.2519900783873629")
    col2.markdown("* **CPU** : 0.2517567086906365")
    col2.markdown("* **GPU** : 0.3422702941182396")
    col2.markdown("* **Touch** : 0.095355125133349")


def bar_plot(var, col):
    # ram storage dist
    col.write("Distribusi " + var + " terbanyak")
    var_old = var

    if var == "Gender":
        var = "gender"
    elif var == "Region":
        var = "region_category"
    elif var == "Membership":
        var = "membership_category"
    elif var == "Referral":
        var = "joined_through_referral"
    elif var == "Preferred Offer":
        var = "preferred_offer_types"
    elif var == "Devices":
        var = "medium_of_operation"
    elif var == "Internet":
        var = "internet_option"
    elif var == "Used Discount":
        var = "used_special_discount"
    elif var == "Offer Application Preference":
        var = "offer_application_preference"
    elif var == "Past Complaint":
        var = "past_complaint"
    elif var == "Complaint Status":
        var = "complaint_status"
    elif var == "Feedback":
        var = "feedback"

    fig = plt.figure(figsize=(10, 5))
    ax1 = sns.countplot(
        data=data,
        x=var,
        palette="mako",
    )
    plt.xlabel(var_old)
    ax1.bar_label(container=ax1.containers[0], labels=data[var].value_counts().values)
    col.pyplot(fig)


def hist_plot(var, col):
    # check price distribution
    col.write("Distribusi " + var)
    var_old = var

    if var == "Age":
        var = "age"
    elif var == "Last Login (Days)":
        var = "days_since_last_login"
    elif var == "Avg. Time Spent":
        var = "avg_time_spent"
    elif var == "Avg. Transaction":
        var = "avg_transaction_value"
    elif var == "Avg. Login Frequency (Days)":
        var = "avg_frequency_login_days"
    elif var == "Points":
        var = "points_in_wallet"
    else:
        var = var

    fig = plt.figure(figsize=(10, 5))

    palette = sns.color_palette("mako_r", 50)
    plt.xlabel(var_old)
    plot = sns.histplot(data=data, x=var, kde=True, bins=50, color="teal")

    for bin_, i in zip(plot.patches, palette):
        bin_.set_facecolor(i)

    col.pyplot(fig)


def churn_score():
    fig = plt.figure(figsize=(20, 5))
    plt.ylabel("Churn Risk Score")

    sorted_scores = data["churn_risk_score"].value_counts().sort_index(ascending=False)
    ax = sns.countplot(
        data=data, y="churn_risk_score", palette="mako", order=sorted_scores.index
    )
    # Get the value counts for each category of 'churn_risk_score'
    value_counts = data["churn_risk_score"].value_counts()

    # Add labels on top of each bar
    for idx, count in enumerate(value_counts):
        ax.text(count + 5, idx, str(count), va="center")

    st.pyplot(fig)


if __name__ == "__main__":
    distribution()
