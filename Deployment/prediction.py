import streamlit as st
import pandas as pd
import numpy as np
import pickle


# load file

with open("./column_transformer.pkl", "rb") as file_4:
    column_transformer = pickle.load(file_4)


def predict():
    # form
    with st.form("key=laptop_predict"):
        st.subheader("Laptop Price Prediction")
        st.write("Laptop Brand & Model")
        col1, col2, col3 = st.columns(3)
        brand = col1.selectbox(
            "Laptop Brand",
            (
                "Asus",
                "Alurin",
                "MSI",
                "HP",
                "Lenovo",
                "Medion",
                "Acer",
                "Apple",
                "Razer",
                "Gigabyte",
                "Dell",
                "LG",
                "Samsung",
                "PcCom",
                "Microsoft",
                "Primux",
                "Prixton",
                "Dynabook Toshiba",
                "Thomson",
                "Denver",
                "Deep Gaming",
                "Vant",
                "Innjoo",
                "Jetwing",
                "Millenium",
                "Realme",
                "Toshiba",
            ),
        )
        model = col2.selectbox(
            "Laptop Model",
            (
                "ExpertBook",
                "Go",
                "Katana",
                "15S",
                "Crosshair",
                "ThinkPad",
                "VivoBook",
                "Akoya",
                "Victus",
                "V15",
                "Thin",
                "ROG",
                "IdeaPad",
                "Cyborg",
                "M515UA",
                "TUF",
                "Aspire",
                "Pavilion",
                "Vector",
                "Chromebook",
                "Omen",
                "ZenBook",
                "Creator",
                "MacBook Air",
                "ThinkBook",
                "250",
                "Modern",
                "255",
                "MacBook Pro",
                "Prestige",
                "Stealth",
                "Pulse",
                "Blade",
                "Legion",
                "Raider",
                "ProBook",
                "F515",
                "G5",
                "Vostro",
                "Nitro",
                "Gram",
                "E410",
                "Flex Advance",
                "Bravo",
                "Aero",
                "Yoga",
                "Galaxy Book",
                "Erazer",
                "Summit",
                "Ultra",
                "Extensa",
                "Flex",
                "EliteBook",
                "Revolt",
                "Latitude",
                "Envy",
                "Deputy",
                "Predator",
                "Surface Laptop",
                "14w",
                "Titan",
                "Ioxbook",
                "Aorus",
                "Swift",
                "Surface Go",
                "Netbook Pro",
                "Surface Pro",
                "Notebook",
                "470",
                "Spectre",
                "Alurin",
                "Satellite Pro",
                "XPS",
                "ConceptD",
                "E510",
                "Beast",
                "TravelMate",
                "Portégé",
                "Tecra",
                "Neo",
                "Electronics",
                "14S",
                "Classmate Pro",
                "17",
                "Zbook",
                "BR",
                "300w",
                "M515",
                "Nubian",
                "100e",
                "Moove",
                "V14",
                "ProArt",
                "100w",
                "V17",
                "F415EA",
                "LOQ",
                "Macbook",
                "Leopard",
                "U4",
                "P1511",
                "Enduro",
                "Precision",
                "G7",
                "Voom",
                "N1510",
                "WS63",
                "AURELION",
                "AZIR",
                "Book Prime",
                "Edge",
                "Book",
                "F415",
                "P1411",
                "A7",
                "15U70N",
                "V330",
                "Alpha",
                "Delta",
                "GL65",
                "GL75",
            ),
        )
        status = col3.radio(
            "Condition",
            ("New", "Refurbished"),
        )

        st.markdown("---")
        st.write("Specification")
        col1, col2 = st.columns(2)
        cpu = col1.selectbox(
            "CPU",
            (
                "Intel Core i5",
                "Intel Celeron",
                "Intel Core i3",
                "Intel Core i7",
                "AMD Ryzen 5",
                "AMD Ryzen 7",
                "AMD Ryzen 3",
                "Apple M1",
                "AMD Athlon",
                "Apple M2",
                "AMD Ryzen 9",
                "Intel Core i9",
                "AMD 3020e",
                "Qualcomm Snapdragon 7",
                "Intel Evo Core i7",
                "Intel Evo Core i5",
                "Intel Pentium",
                "Apple M2 Pro",
                "AMD Radeon 5",
                "Intel Evo Core i9",
                "AMD 3015e",
                "Apple M1 Pro",
                "Intel Core M3",
                "AMD Radeon 9",
                "Mediatek MT8183",
                "AMD 3015Ce",
                "Qualcomm Snapdragon 8",
                "Microsoft SQ1",
            ),
            help="Processor",
        )

        gpu = col2.selectbox(
            "GPU",
            (
                "No GPU",
                "RTX 3050",
                "RTX 4060",
                "RTX 4050",
                "RTX 3060",
                "RTX 4070",
                "RTX 2050",
                "GTX 1650",
                "RTX 3070",
                "610 M",
                "RTX 4080",
                "RX 6500M",
                "MX 550",
                "RTX 3080",
                "RTX 4090",
                "RX 7600S",
                "A 370M",
                "GTX 1660",
                "RTX A1000",
                "RTX 3000",
                "T 1200",
                "Radeon Pro 5300M",
                "A 730M",
                "Radeon Pro RX 560X",
                "RTX A5500",
                "Radeon Pro 5500M",
                "T 500",
                "T 550",
                "RTX A3000",
                "T 2000",
                "T 600",
                "T 1000",
                "Radeon RX 6600M",
                "MX 330",
                "RTX A2000",
                "MX 450",
                "RTX 2070",
                "RX 6800S",
                "RTX 2080",
                "RTX 2060",
                "GTX 1050",
                "MX 130",
                "P 500",
                "RX 6700M",
                "GTX 1070",
            ),
            help="Graphic Processor",
        )

        col1, col2, col3 = st.columns(3)
        ram = col1.select_slider(
            "RAM Size",
            options=[
                4,
                6,
                8,
                12,
                16,
                32,
                40,
                64,
                128,
            ],
            help="Gigabytes",
        )

        storage = col2.select_slider(
            "Storage Size",
            options=[0, 32, 64, 1000, 128, 240, 256, 500, 512, 2000, 3000, 4000],
            help="Gigabytes",
        )

        storage_type = col3.radio("Storage Type", ("SSD", "eMMC", "HDD"))

        col1, col2 = st.columns(2)
        screen_size = col1.select_slider(
            "Screen Size",
            options=[
                10.1,
                10.5,
                10.95,
                11.6,
                12.0,
                12.3,
                12.4,
                12.5,
                13.0,
                13.3,
                13.4,
                13.5,
                13.6,
                13.9,
                14.0,
                14.1,
                14.2,
                14.4,
                14.5,
                15.0,
                15.3,
                15.4,
                15.6,
                16.0,
                16.1,
                16.2,
                17.0,
                17.3,
                18.0,
            ],
            help="Inches",
        )

        screen_type = col2.radio("Touch Screen?", ("Yes", "No"))
        st.markdown("---")
        submitted = st.form_submit_button("Predict")

    # inferencing
    data_inf = {
        "status": status,
        "brand": brand,
        "model": model,
        "cpu": cpu,
        "ram": ram,
        "storage": storage,
        "storage type": storage_type,
        "gpu": gpu,
        "screen": screen_size,
        "touch": screen_type,
    }

    data_inf = pd.DataFrame([data_inf])

    price = model_grid_xgb.predict(data_inf)

    price_final = "{:.2f}".format(price[0])

    st.dataframe(data_inf)
    st.write("Prediksi Harga Laptop Dengan Spesifikasi Tersebut adalah")

    html_str = f"""
                <style>
                p.a {{
                font: bold 36px Arial;
                }}
                </style>
                <p class="a">USD ${price_final}</p>
                """
    st.markdown(html_str, unsafe_allow_html=True)


if __name__ == "__main__":
    predict()
