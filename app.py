import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.title("Customer Segmentation")

# Load pickle files
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv("Online Retail.csv", encoding='ISO-8859-1')
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Scale & predict
rfm_scaled = scaler.transform(rfm)
rfm['Cluster'] = kmeans.predict(rfm_scaled)

st.write(rfm.head())

# Plot
fig, ax = plt.subplots()
ax.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
ax.set_xlabel("Recency")
ax.set_ylabel("Monetary")
st.pyplot(fig)