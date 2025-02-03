# Customer Clustering using K-Means

## 📌 Project Overview
This project focuses on customer segmentation using **K-Means Clustering**, an unsupervised machine learning technique. The dataset used in this project contains information about mall customers, including their **age, gender, annual income, and spending score**. The goal is to segment customers into different groups based on their purchasing behavior.

## 🚀 Features
- **Exploratory Data Analysis (EDA)**
  - Gender distribution analysis
  - Age distribution visualization
  - Annual income distribution
- **K-Means Clustering**
  - Elbow Method to determine the optimal number of clusters
  - Cluster visualization with centroids
  - Cluster characteristics summary
- **Interactive Dashboard with Streamlit**
  - Provides user-friendly visualization and insights
  - Enables users to explore customer segments dynamically

## 📂 Dataset
The dataset used in this project is the **Mall Customers Dataset**. It includes the following columns:
- `CustomerID`: Unique identifier for each customer
- `Gender`: Male or Female
- `Age`: Customer's age
- `Annual Income (k$)`: Annual income in thousands of dollars
- `Spending Score (1-100)`: A score assigned based on customer spending behavior

Dataset Source: [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## 🛠️ Technologies Used
- **Python**
- **Pandas** (for data manipulation)
- **Matplotlib & Seaborn** (for data visualization)
- **Scikit-learn** (for K-Means clustering)
- **Plotly** (for interactive visualizations)
- **Yellowbrick** (for Elbow Method visualization)
- **Streamlit** (for building an interactive dashboard)

## 🔧 Installation & Setup
Follow these steps to set up the project:

1. **Clone the repository**
   ```bash
   git clone https://github.com/ai-azz/customer-segmentation.git
   cd customer-segmentation
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On MacOS/Linux
   venv\Scripts\activate    # On Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**
   ```bash
   streamlit run dashboard.py
   ```

## 📊 Data Analysis & Insights
### 1️⃣ Exploratory Data Analysis (EDA)
The dataset is explored through various visualizations:
- **Gender distribution**: Pie chart representing male vs female customers.
- **Age distribution**: Bar chart showing the number of customers per age group.
- **Annual income distribution**: Histogram illustrating income ranges.

### 2️⃣ Elbow Method for Optimal Clusters
The **Elbow Method** is used to determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) against different values of **K**.

### 3️⃣ K-Means Clustering
Customers are grouped based on their **Annual Income** and **Spending Score**. The model assigns each customer to a cluster, and the centroids of these clusters are plotted for better interpretation.

## 📌 Results & Findings
- The optimal number of clusters is **4**, as determined by the Elbow Method.
- Customers are segmented into four groups based on their income and spending behavior.
- High-income customers with high spending scores form a distinct cluster, while low-income customers with lower spending scores form another.