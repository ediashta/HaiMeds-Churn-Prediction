import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="HaiMeds Churn Prediction",
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
    col1, col2 = st.columns([7, 5])

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
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    col1.pyplot(fig)

    feature_importance_info = """
        **Feature Importance:**

        - **gender:** 0.0
        - **region_category:** 0.0223
        - **membership_category:** 0.7859
        - **joining_date:** 0.0
        - **joined_through_referral:** 0.0355
        - **preferred_offer_types:** 0.0434
        - **medium_of_operation:** 0.0218
        - **internet_option:** 0.0025
        - **last_visit_time:** 0.0604
        - **used_special_discount:** 0.0092
        - **offer_application_preference:** 0.0179
        - **past_complaint:** 0.0072
        - **complaint_status:** 0.0054
        - **feedback:** 0.4561
        """
    col2.markdown(feature_importance_info)


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
