import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Streamlit App Title
st.title("Genomic Data Analysis App")

# File Upload Section
st.sidebar.header("Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File Uploaded Successfully!")
else:
    # Generate Synthetic Genomic Data if No File is Uploaded
    st.sidebar.warning("No file uploaded. Using synthetic data.")
    np.random.seed(42)
    n_samples = 4000
    n_features = 1500  # Number of SNPs
    data = np.random.randint(0, 3, size=(n_samples, n_features))
    df = pd.DataFrame(data, columns=[f'SNP_{i}' for i in range(n_features)])
    df['Target'] = np.random.choice([0, 1], size=n_samples)

# Display Data
st.subheader("Dataset Preview")
st.write(df.head())
st.write(f"Dataset Shape: {df.shape}")

# Basic EDA
st.subheader("Basic EDA")
st.write("Missing Values:", df.isnull().sum().sum())
st.write("Class Distribution:")
st.bar_chart(df['Target'].value_counts())

# SNP Distribution Visualization
st.subheader("SNP Variant Distribution")
plt.figure(figsize=(8, 4))
sns.histplot(df.iloc[:, 0], bins=3, kde=False)
plt.title("Distribution of SNP_0 Variants")
st.pyplot(plt)

# PCA Visualization
st.subheader("PCA Visualization")
pca = PCA(n_components=2)
pca_results = pca.fit_transform(df.drop(columns=['Target']))
fig, ax = plt.subplots()
sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=df['Target'], palette='coolwarm', ax=ax)
plt.title("PCA Visualization of Genomic Data")
st.pyplot(fig)

# t-SNE Visualization
st.subheader("t-SNE Visualization")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(df.drop(columns=['Target']))
fig, ax = plt.subplots()
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df['Target'], palette='coolwarm', ax=ax)
plt.title("t-SNE Visualization of Genomic Data")
st.pyplot(fig)

# Machine Learning - Random Forest Classifier
st.subheader("Machine Learning Model - Random Forest")
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
st.write("**Model Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
