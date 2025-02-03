import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/ai-azz/customer-segmentation/refs/heads/main/data/Mall_Customers.csv')

# Page configuration
st.set_page_config(page_title="Customer Clustering", page_icon="💰", layout="wide")

# Sidebar
st.sidebar.title("Customer Segmentation Dashboard")
selected_analysis = st.sidebar.selectbox("Select Analysis Type", ["EDA", "K-Means Clustering"])

# Exploratory Data Analysis (EDA)
if selected_analysis == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Gender Distribution
    st.subheader("Gender Distribution")
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution')
    st.plotly_chart(fig)
    st.write("From the visualization, we can see that female customers make up 56.0% while male customers make up 44.0% of the total customer base.")
    
    # Age Distribution
    st.subheader("Age Distribution")
    bins = [18, 25, 35, 45, 55, df['Age'].max()]
    labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    age_counts = df['Age Group'].value_counts().reset_index()
    age_counts.columns = ['Age Group', 'Count']
    fig = px.bar(age_counts, x='Age Group', y='Count', title='Customer Age Distribution', color='Age Group')
    st.plotly_chart(fig)
    st.write("The largest customer age group is 26-35 years old, while the smallest group is customers aged 55+.")
    
    # Annual Income Distribution
    st.subheader("Annual Income Distribution")
    bins_income = [0, 30, 60, 90, 120, 150]
    labels_income = ['$0-30k', '$30k-60k', '$60k-90k', '$90k-120k', '$120k-150k']
    df['Income Group'] = pd.cut(df['Annual Income (k$)'], bins=bins_income, labels=labels_income, include_lowest=True)
    income_counts = df['Income Group'].value_counts().reset_index()
    income_counts.columns = ['Income Group', 'Count']
    fig = px.bar(income_counts, x='Income Group', y='Count', title='Customer Annual Income Distribution', color='Income Group')
    st.plotly_chart(fig)
    st.write("Most customers fall within the annual income range of \$60,001–90,000, while the $120,001–150,000 category has the least number of customers.")

# K-Means Clustering
elif selected_analysis == "K-Means Clustering":
    st.title("K-Means Clustering")
    X = df.iloc[:, [3, 4]].values
    
    # Elbow Method using Plotly
    st.subheader("Elbow Method to Determine Optimal K")
    fig, ax = plt.subplots()
    kmeans = KMeans()
    visualizer = KElbowVisualizer(kmeans, k=(1, 10), ax=ax)
    visualizer.fit(X)
    visualizer.show()
    st.pyplot(fig)
    st.write("The elbow method analysis suggests that the optimal number of clusters is 4, with a total within-cluster sum of squares (WCSS) of 73,679.789. This means that dividing the data into 4 clusters provides the best balance between minimizing intra-cluster distance and maximizing inter-cluster distance.")

    # Clustering with K=4
    st.subheader("Cluster Analysis with K=4")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color=df['Cluster'].astype(str),
                     title='Cluster Visualization with Centroids', labels={'color': 'Cluster'})
    for i, centroid in enumerate(centroids):
        fig.add_scatter(x=[centroid[0]], y=[centroid[1]], mode='markers', marker=dict(size=10, symbol='x', color='red'), name=f'Centroid {i+1}')
    st.plotly_chart(fig)
    
    # Cluster Characteristics
    st.subheader("Cluster Characteristics")
    cluster_summary = []
    for i in range(4):
        cluster_data = df[df['Cluster'] == i]
        cluster_summary.append(
            f"Cluster {i+1}:\nAnnual Income Avg: {cluster_data['Annual Income (k$)'].mean():.2f} k$\n"
            f"Spending Score Avg: {cluster_data['Spending Score (1-100)'].mean():.2f}\n"
        )
    st.text("\n".join(cluster_summary))
    
    st.write("\n\nThe cluster characteristics are as follows:")
    st.write("- Cluster 1: Medium income, high spending score.")
    st.write("- Cluster 2: High income, high spending score.")
    st.write("- Cluster 3: High income, low spending score.")
    st.write("- Cluster 4: Low income, low spending score.")
    st.write("These insights help in developing better marketing strategies based on customer spending behavior.")