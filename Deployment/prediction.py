import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import load_model


# load file

with open("./column_transformer.pkl", "rb") as file_1:
    column_transformer = pickle.load(file_1)

model_functional = load_model("./functional_model.keras")


def predict():
    # form
    with st.form("key=churn_prediction"):
        st.subheader("Churn Score Prediction")

        st.markdown("**Customer Data**")

        col1, col2 = st.columns(2, gap="large")
        age = col1.number_input(label="Age", help="Customer Age", step=1, value=20)

        membership = col2.selectbox(
            label="Membership Category",
            options=(
                "No Membership",
                "Basic Membership",
                "Premium Membership",
                "Silver Membership",
                "Gold Membership",
                "Platinum Membership",
            ),
        )
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4, gap="large")
        region = col1.radio(
            label="Region",
            help="Customer Residence Region",
            options=("Town", "City", "Village"),
        )

        referral = col2.radio(
            label="Referral", help="Joined Through Referral?", options=("Yes", "No")
        )

        device = col3.radio(
            label="Device(s)",
            help="Device Used",
            options=("Smartphone", "Desktop", "Both"),
        )

        internet = col4.radio(
            label="Internet Connection", options=("Wi-Fi", "Fiber_Optic", "Mobile_Data")
        )

        st.markdown("---")

        st.markdown("**Customer Behavior**")
        col1, col2, col3, col4, col5 = st.columns(5, gap="large")

        last_login = col1.number_input(
            label="Last Login", help="Days Since Last Login", step=1, value=6
        )

        avg_time = col2.number_input(
            label="Avg. Usage Time", help="Average Usage Time (Minutes)", value=30
        )

        avg_login = col3.number_input(
            label="Avg. Login Frequency",
            help="Average Login Frequency (Days)",
            value=14,
        )

        points = col4.number_input(label="Points in Wallet", value=300)

        transaction = col5.number_input(label="Avg. Transaction", value=100, help="USD")

        st.markdown("---")

        col1, col2, col3 = st.columns(3, gap="large")

        offer_pref = col1.selectbox(
            label="Preferred Offer Type",
            options=(
                "Gift Vouchers/Coupons",
                "Credit/Debit Card Offers",
                "Without Offers",
            ),
        )

        used_disc = col2.radio(label="Used Discount Before?", options=("Yes", "No"))

        offer_app = col3.radio(
            label="Application Preference Offer?", options=("Yes", "No")
        )

        st.markdown("---")
        col1, col2, col3 = st.columns(3, gap="large")

        complaints = col1.radio(label="Past Complaint?", options=("Yes", "No"))

        complaints_status = col2.selectbox(
            label="Complaint Status",
            options=(
                "Not Appllicable",
                "Unsolved",
                "Solved",
                "Solved in Follow-up",
                "No Information Available",
            ),
        )
        feedback = col3.selectbox(
            label="Feedback Type", options=("Neutral", "Positive", "Negative")
        )
        submitted = st.form_submit_button("Predict")

    # inferencing
    data_inf = [
        {
            "age": age,
            "region_category": region,
            "membership_category": membership,
            "joined_through_referral": referral,
            "preferred_offer_types": offer_pref,
            "medium_of_operation": device,
            "internet_option": internet,
            "days_since_last_login": last_login,
            "avg_time_spent": avg_time,
            "avg_transaction_value": transaction,
            "avg_frequency_login_days": avg_login,
            "points_in_wallet": points,
            "used_special_discount": used_disc,
            "offer_application_preference": offer_app,
            "past_complaint": complaints,
            "complaint_status": complaints_status,
            "feedback": feedback,
        }
    ]

    data_inf = pd.DataFrame(data_inf)

    st.dataframe(data_inf)

    data_inf_transform = column_transformer.transform(data_inf)
    y_pred_inf = model_functional.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.65, 1, 0)

    st.write("Prediksi Churn Pelanggan Tersebut adalah :")
    if y_pred_inf[0] == 1:
        html_str = f"""
                    <style>
                    p.a {{
                    font: bold 36px Arial;
                    color: teal;
                    }}
                    </style>
                    <p class="a">Pelanggan Tidak Berpotensi Churn</p>
                    """
        st.markdown(html_str, unsafe_allow_html=True)
        st.write(
            "Dapat menekankan program loyalty agar pelanggan tetap menggunakan layanan"
        )
    else:
        html_str = f"""
                    <style>
                    p.a {{
                    font: bold 36px Arial;
                    color: red;
                    }}
                    </style>
                    <p class="a">Pelanggan Berpotensi Churn</p>
                    """
        st.markdown(html_str, unsafe_allow_html=True)
        st.write("Dapat diberikan promosi untuk menarik pelanggan kembali")


if __name__ == "__main__":
    predict()
