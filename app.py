import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Customer Segmentation App")
st.write("This app predicts customer segment using a trained clustering model.")

# Load saved model and scaler
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.subheader("Enter Customer Details")

# User inputs
recency = st.number_input("Recency (days since last purchase)", min_value=0, value=10)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
monetary = st.number_input("Monetary (total spending)", min_value=0.0, value=500.0)

if st.button("Predict Segment"):
    # Prepare input
    customer_data = np.array([[recency, frequency, monetary]])
    
    # Scale input
    scaled_data = scaler.transform(customer_data)
    
    # Predict cluster
    cluster = kmeans.predict(scaled_data)[0]
    
    st.success(f"Predicted Customer Segment: Cluster {cluster}")

st.markdown("---")
st.write("Model: K-Means Clustering | Built with Streamlit")