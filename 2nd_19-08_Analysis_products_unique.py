#Load libraries
import pandas as pd
import matplotlib.pyplot as plt

#Load data
data = pd.read_csv("./data.csv")

# Group by the number of unique product IDs and calculate the observation count
unique_product_counts = data['product_id'].value_counts().reset_index()
unique_product_counts.columns = ['product_id', 'observation_count']
unique_product_counts.sort_values(by='observation_count', ascending=False, inplace=True)

# Calculate the cumulative percentage of observations
unique_product_counts['cumulative_percentage'] = (unique_product_counts['observation_count'].cumsum() / data.shape[0]) * 100

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(unique_product_counts.index, unique_product_counts['cumulative_percentage'], marker='o')
plt.title('Cumulative Percentage of Observations vs. Unique Product Counts')
plt.xlabel('Unique Product Counts')
plt.ylabel('Cumulative Percentage of Observations (%)')
plt.grid(True)
plt.show()

#We find that at 4537 unique products, we have 80% of cumulative observations
target = unique_product_counts[unique_product_counts["cumulative_percentage"] <= 80].shape[0]
cutoff = unique_product_counts[unique_product_counts["cumulative_percentage"] <= 80]

#Load products and assign products
products_mapping = pd.read_csv("./products.csv")
unique_product_counts = unique_product_counts.merge((products_mapping), on='product_id', how='left')
unique_product_counts['observation_percentage'] = (unique_product_counts['observation_count'] / data.shape[0]) * 100

#Exclude a number of products
excluded_product_ids = unique_product_counts.head(6)['product_id']
cutoff = cutoff["product_id"]
unique_products_select = unique_product_counts[unique_product_counts['product_id'].isin(cutoff)]
data = data[data['product_id'].isin(cutoff)]
data = data[~data['product_id'].isin(excluded_product_ids)]
del cutoff, excluded_product_ids, products_mapping, target, unique_product_counts, unique_products_select

