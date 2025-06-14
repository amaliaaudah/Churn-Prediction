import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import io

# Load the pre-trained model
rf_model = joblib.load('random_forest.pkl')

# Streamlit web interface
st.title('Random Forest Churn Prediction')
st.write("Upload your dataset to make churn predictions using the trained Random Forest model.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

# Function to preprocess data
def preprocess_data(df):
    # Strip spaces in 'STATUS' column
    df['STATUS'] = df['STATUS'].str.strip()

    # Rename column 'DURASI BERLANGGANAN (BULAN)' to 'durasiBulan'
    df.rename(columns={'DURASI BERLANGGANAN (BULAN)': 'durasiBulan'}, inplace=True)

    # Handle missing values for 'kapasitas'
    df['kapasitas'] = df['KAPASITAS'].fillna(df['KAPASITAS'].median())

    # Encode categorical features
    df['tregID'] = LabelEncoder().fit_transform(df['TREG'])
    df['typeID'] = LabelEncoder().fit_transform(df['TYPE'])
    df['kategoriPelangganID'] = LabelEncoder().fit_transform(df['REVENUE CATEGORY'])
    df['status'] = LabelEncoder().fit_transform(df['STATUS'])

    # Drop the original columns
    df.drop(columns=['TREG', 'TYPE', 'REVENUE CATEGORY', 'KAPASITAS', 'STATUS'], inplace=True)

    return df

# If a file is uploaded
if uploaded_file is not None:
    # Load and display the dataset
    df_new = pd.read_csv(uploaded_file, sep=';')
    st.write("Dataset Preview:")
    st.write(df_new.head())

    # Copy the dataframe for preprocessing
    df = df_new.copy()

    # Preprocess the data
    df_processed = preprocess_data(df)

    # Features for prediction
    fitur_model = ['durasiBulan', 'kapasitas', 'tregID', 'typeID', 'kategoriPelangganID']
    X_pred = df_processed[fitur_model]

    # Make predictions (binary labels) and probabilities
    y_pred = rf_model.predict(X_pred)
    y_prob = rf_model.predict_proba(X_pred)[:, 1]  # Probabilities for class '1' (churn)

    # Add prediction to original dataframe
    df_new['RF'] = y_pred
    df_new['Probability Churn'] = y_prob  # Add the churn probability column
    df_new['Berpotensi Churn?'] = df_new['RF'].map({0: 'Tidak Berpotensi Churn', 1: 'Berpotensi Churn'})

    # Show prediction results interactively
    st.write("Prediction Results:")
    st.dataframe(df_new)  # Use this for an interactive table

    # Allow the user to download the prediction result as an Excel file
    @st.cache_data
    def to_excel(df):
        """Convert the dataframe to an Excel file."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        processed_data = output.getvalue()
        return processed_data

    # Download button for the Excel file
    excel_file = to_excel(df_new)
    st.download_button(
        label="Download Predicted Data as Excel",
        data=excel_file,
        file_name="hasil_prediksi_rf.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
