# -----------------------------------------
# E-Commerce Customer Segmentation (RFM)
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# 1 Load Dataset
# -----------------------------------------

print("Loading dataset...")

df = pd.read_csv("data/ecommerce_sales.csv", encoding="latin1")

print(df.head())

# -----------------------------------------
# 2 Data Cleaning
# -----------------------------------------

print("Cleaning data...")

# Remove missing customer IDs
df.dropna(subset=['CustomerID'], inplace=True)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)

# Create revenue column
df['Revenue'] = df['Quantity'] * df['UnitPrice']

print("Data cleaned successfully")

# -----------------------------------------
# 3 Basic Sales Analysis
# -----------------------------------------

print("\nTotal Revenue:")

total_revenue = df['Revenue'].sum()

print(total_revenue)

print("\nTop Selling Products:")

top_products = (
    df.groupby('Description')['Quantity']
    .sum()
    .sort_values(ascending=False)
)

print(top_products.head(10))

# -----------------------------------------
# 4 Monthly Sales Trend
# -----------------------------------------

df['Month'] = df['InvoiceDate'].dt.to_period('M')

monthly_sales = df.groupby('Month')['Revenue'].sum()

print("\nMonthly Sales Trend")

print(monthly_sales.head())

plt.figure()

monthly_sales.plot()

plt.title("Monthly Revenue Trend")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.show()

# -----------------------------------------
# 5 RFM Analysis
# -----------------------------------------

print("\nPerforming RFM Analysis...")

today = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({

    'InvoiceDate': lambda x: (today - x.max()).days,
    'InvoiceNo': 'count',
    'Revenue': 'sum'

})

rfm.columns = ['Recency','Frequency','Monetary']

print(rfm.head())

# -----------------------------------------
# 6 RFM Scoring
# -----------------------------------------

rfm['R_score'] = pd.qcut(rfm['Recency'],4,labels=[4,3,2,1])

rfm['F_score'] = pd.qcut(
    rfm['Frequency'].rank(method='first'),
    4,
    labels=[1,2,3,4]
)

rfm['M_score'] = pd.qcut(rfm['Monetary'],4,labels=[1,2,3,4])

rfm['RFM_score'] = rfm[['R_score','F_score','M_score']].astype(str).sum(axis=1)

# -----------------------------------------
# 7 Customer Segmentation
# -----------------------------------------

def segment(row):

    r = int(row['R_score'])
    f = int(row['F_score'])
    m = int(row['M_score'])

    if r == 4 and f == 4 and m == 4:
        return 'Best Customers'

    elif f >= 3:
        return 'Loyal Customers'

    elif r <= 2:
        return 'At Risk'

    else:
        return 'New Customers'


rfm['Segment'] = rfm.apply(segment, axis=1)

print("\nCustomer Segments")

print(rfm['Segment'].value_counts())

# -----------------------------------------
# 8 Visualization
# -----------------------------------------

segment_counts = rfm['Segment'].value_counts()

plt.figure()

segment_counts.plot(kind='bar')

plt.title("Customer Segments")

plt.xlabel("Segment")

plt.ylabel("Number of Customers")

plt.show()

# -----------------------------------------
# 9 Save Output
# -----------------------------------------

print("\nSaving results...")

rfm.to_csv("outputs/customer_segments.csv")

print("Customer segmentation file saved")

print("\nProject Completed")
