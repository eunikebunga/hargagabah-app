import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Untuk plotting di Streamlit
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Prediksi Harga Gabah Menggunakan Random Forest")

# Upload data
harga_file = st.file_uploader("Upload file Harga Gabah (CSV)", type="csv")
padi_file = st.file_uploader("Upload file Data Padi (CSV)", type="csv")

if harga_file and padi_file:
    # Load data
    df_harga = pd.read_csv(harga_file)
    df_padi = pd.read_csv(padi_file)

    # Mapping bulan
    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    # Proses GKG
    gkg_cols = [col for col in df_harga.columns if col.startswith('GKG_')]
    df_gkg = df_harga.melt(
        id_vars=['Bulan'],
        value_vars=gkg_cols,
        var_name='Tahun',
        value_name='Harga_GKG'
    )
    df_gkg['Tahun'] = df_gkg['Tahun'].str.replace('GKG_', '').astype(int)
    df_gkg['Bulan_Num'] = df_gkg['Bulan'].map(bulan_map)

    # Filter Lampung
    df_padi = df_padi[df_padi['Wilayah'] == 'Provinsi Lampung']

    tahun = [2019, 2020, 2021, 2022, 2023, 2024]
    data_padi = []
    for yr in tahun:
        data_padi.append({
            'Tahun': yr,
            'Luas_Panen': df_padi[f'Luas Panen {yr}'].values[0],
            'Produksi': df_padi[f'Produksi {yr}'].values[0],
            'Produktivitas': df_padi[f'Produktivitas {yr}'].values[0]
        })
    df_padi_prov = pd.DataFrame(data_padi)

    # Gabungkan
    df_merged = pd.merge(df_gkg, df_padi_prov, on='Tahun')
    X = df_merged[['Bulan_Num', 'Luas_Panen', 'Produksi', 'Produktivitas']]
    y = df_merged['Harga_GKG']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Grid search
    with st.spinner("🔍 Melatih model Random Forest..."):
        param_grid = {
            'n_estimators': [100],
            'max_depth': [None],
            'min_samples_split': [2],
            'max_features': ['sqrt']
        }
        model_rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=model_rf,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

    # Evaluasi
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Evaluasi Model")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R-squared:** {r2:.2f}")

    # Feature importance
    st.subheader("🔍 Pengaruh Fitur")
    feature_importance = best_model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=features, ax=ax)
    ax.set_title("Pengaruh Fitur pada Prediksi Harga GKG")
    st.pyplot(fig)

    # Prediksi manual
    st.subheader("📌 Prediksi Harga Gabah (Input Manual)")
    bulan = st.selectbox("Bulan", list(bulan_map.keys()))
    luas_panen = st.number_input("Luas Panen (ha)", value=450000.0)
    produksi = st.number_input("Produksi (ton)", value=2100000.0)
    produktivitas = st.number_input("Produktivitas (ku/ha)", value=46.0)

    input_pred = pd.DataFrame({
        'Bulan_Num': [bulan_map[bulan]],
        'Luas_Panen': [luas_panen],
        'Produksi': [produksi],
        'Produktivitas': [produktivitas]
    })

    if st.button("Prediksi"):
        hasil = best_model.predict(input_pred)[0]
        st.success(f"Prediksi Harga GKG: Rp {hasil:.2f}/kg")
