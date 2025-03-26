import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

# Load full CSVs
aisles = pd.read_csv("aisles.csv")
departments = pd.read_csv("departments.csv")
order_products = pd.read_csv("order_products__train.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")

# Merge Data
merged = order_products.merge(orders, on="order_id")
merged = merged.merge(products, on="product_id")
merged = merged.merge(departments, on="department_id")
merged = merged.merge(aisles, on="aisle_id")

# Sidebar Navigation
st.set_page_config(page_title="Instacart Dashboard", layout="wide")
menu = st.sidebar.selectbox("Navigate", [
    "Executive Overview",
    "Customer Insights",
    "Product Analytics",
    "Department Trends",
    "Customer Segments",
    "Weekly Trends",
    "Market Basket Insights",
    "Business Recommendations"
])

# Executive Overview
if menu == "Executive Overview":
    st.title("📊 Executive Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Orders", f"{orders['order_id'].nunique():,}")
    with col2:
        st.metric("Total Customers", f"{orders['user_id'].nunique():,}")
    with col3:
        avg_basket = round(order_products.shape[0] / order_products['order_id'].nunique(), 2)
        st.metric("Average Basket Size", avg_basket)

    st.markdown("---")
    st.subheader("Orders by Department")
    dep_counts = merged["department"].value_counts().reset_index()
    dep_counts.columns = ["Department", "Orders"]
    st.plotly_chart(px.pie(dep_counts, names="Department", values="Orders", hole=0.4))

# Customer Insights
elif menu == "Customer Insights":
    st.title("👥 Customer Insights")
    st.subheader("Orders by Day of Week")
    dow_map = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    dow = orders["order_dow"].value_counts().sort_index()
    dow_df = pd.DataFrame({"Day": [dow_map[i] for i in dow.index], "Orders": dow.values})
    st.plotly_chart(px.bar(dow_df, x="Day", y="Orders"))

    st.subheader("Orders by Hour of Day")
    hour = orders["order_hour_of_day"].value_counts().sort_index()
    hour_df = pd.DataFrame({"Hour": hour.index, "Orders": hour.values})
    st.plotly_chart(px.line(hour_df, x="Hour", y="Orders"))

    st.subheader("Days Since Prior Order")
    st.plotly_chart(px.histogram(orders, x="days_since_prior_order", nbins=30))

# Product Analytics
elif menu == "Product Analytics":
    st.title("📦 Product Analytics")
    st.subheader("Top 10 Most Ordered Products")
    top_products = merged["product_name"].value_counts().head(10).reset_index()
    top_products.columns = ["Product", "Orders"]
    st.plotly_chart(px.bar(top_products, x="Product", y="Orders", text="Orders"))

    st.subheader("Most Frequently Reordered Products")
    reordered_df = merged[merged["reordered"] == 1]
    top_reordered = reordered_df["product_name"].value_counts().head(10).reset_index()
    top_reordered.columns = ["Product", "Reorders"]
    st.plotly_chart(px.bar(top_reordered, x="Product", y="Reorders", text="Reorders"))

# Department Trends
elif menu == "Department Trends":
    st.title("🏪 Department Trends")
    st.subheader("Product Distribution by Department and Aisle")
    grouped = merged.groupby(["department", "aisle"]).size().reset_index(name="Count")
    st.plotly_chart(px.treemap(grouped, path=["department", "aisle"], values="Count"))

    st.subheader("Department Popularity by Hour")
    dep_hour = merged.groupby(["department", "order_hour_of_day"]).size().reset_index(name="Orders")
    st.plotly_chart(px.line(dep_hour, x="order_hour_of_day", y="Orders", color="department"))

# Customer Segments
elif menu == "Customer Segments":
    st.title("🔍 Customer Segmentation (Recency & Frequency)")

    # Safely generate order_date using order_number
    orders = orders.sort_values(by=['user_id', 'order_number'])
    orders['order_date'] = pd.to_datetime('2017-01-01') + pd.to_timedelta(orders['order_number'], unit='D')

    orders = orders.sort_values(by=['user_id', 'order_date'])
    orders['gap'] = orders.groupby('user_id')['order_date'].diff().dt.days

    freq_df = orders.groupby('user_id')['order_id'].nunique().reset_index(name='total_orders')
    recency_df = orders.groupby('user_id')['gap'].min().reset_index(name='recency')
    customer_data = pd.merge(freq_df, recency_df, on='user_id').fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(customer_data[['total_orders', 'recency']])
    kmeans = KMeans(n_clusters=4, random_state=42).fit(scaled)
    customer_data['cluster'] = kmeans.labels_

    st.plotly_chart(px.scatter(customer_data, x='total_orders', y='recency', color='cluster', title='Customer Clusters'))

# Weekly Trends
elif menu == "Weekly Trends":
    st.title("📈 Weekly Product Trends")

    # Safely reuse or generate order_date using order_number
    orders = orders.sort_values(by=['user_id', 'order_number'])
    orders['order_date'] = pd.to_datetime('2017-01-01') + pd.to_timedelta(orders['order_number'], unit='D')

    order_products_weekly = order_products.merge(orders[['order_id', 'order_date']], on='order_id')
    order_products_weekly['week'] = order_products_weekly['order_date'].dt.to_period('W')

    top_items = order_products_weekly.merge(products, on='product_id')
    weekly = top_items.groupby(['week', 'product_name']).size().reset_index(name='quantity')
    pivot = weekly.pivot(index='week', columns='product_name', values='quantity').fillna(0)
    trend = pivot.diff().sum().sort_values(ascending=False).head(10)

    st.bar_chart(trend)

# Market Basket Insights
elif menu == "Market Basket Insights":
    st.title("🛒 Market Basket Analysis")
    st.subheader("Frequent Itemsets and Association Rules")

    # Filter frequent products and users
    product_counts = merged['product_name'].value_counts(normalize=True) * 100
    popular_items = product_counts[product_counts.cumsum() < 50].index
    merged_filtered = merged[merged['product_name'].isin(popular_items)]

    user_counts = merged_filtered['user_id'].value_counts().head(2500).index
    merged_filtered = merged_filtered[merged_filtered['user_id'].isin(user_counts)]

    basket = merged_filtered.groupby(['order_id', 'product_name'])['product_id'].count().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)

    frequent = apriori(basket_sets, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.1)

    rules['antecedents_'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    pivot = rules[rules['antecedents_'] != rules['consequents_']].pivot(index='antecedents_', columns='consequents_', values='lift')

    st.dataframe(rules[['antecedents_', 'consequents_', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(15))

    st.subheader("Lift Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# Business Recommendations
elif menu == "Business Recommendations":
    st.title("📈 Strategic Recommendations")
    st.markdown("""
    ### Key Observations:
    - 🕒 Peak order times are between 10 AM and 4 PM
    - 🧺 Basket size hovers around 5 items — ideal for bundling offers
    - 🥦 Fresh produce and dairy dominate orders — key promo categories
    - 🔁 High reordering rates suggest opportunity for subscription/repeat-item suggestions

    ### Suggested Actions:
    - Launch time-based discounts for slow hours
    - Personalize top 5 reorders per customer
    - Push combo offers for popular aisles
    - Expand cross-sell on "Buy again" lists
    """)

st.markdown("---")
st.caption("Designed with ❤️ by Instacart Analytics Dashboard | Powered by Streamlit")
