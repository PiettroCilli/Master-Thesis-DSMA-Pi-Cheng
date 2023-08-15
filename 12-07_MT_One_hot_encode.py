#Loading libraries
import pandas as pd

#Loading data
data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_transformed.csv")

#Setting display options
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

#Look at general information
data.nunique()
data.info()
data.describe(include = 'all')

##Transforming the data as a variation of one hot encoding
#Selecting only the necessary columns
data = data[['user_id', 'product_name', 'sticky', 'avg_days_since_prior_order', 'u_reordered_ratio', 'u_total_orders', 'order_size_avg']]

# Pivot the data
pivot_data = data.pivot(index='user_id', columns='product_name', values='sticky')

# Fill NaN values with 0 (representing that the user has not bought the product)
pivot_data = pivot_data.fillna(0)

# Reset the index to make user_id a column again
pivot_data = pivot_data.reset_index()

# Get the other user-specific data (we can take the first non-null value because these features are the same for each product for a user)
user_data = data.groupby('user_id').first().reset_index()[['user_id', 'avg_days_since_prior_order', 'u_reordered_ratio', 'u_total_orders', 'order_size_avg']]

# Merge the user_data with the pivot_data
final_data = pd.merge(user_data, pivot_data, on='user_id')

#Opslaan van CSV file
final_data.to_csv("data_small_one-hot.csv", index = False)



 
