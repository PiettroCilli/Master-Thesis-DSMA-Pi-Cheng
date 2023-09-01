#Loading libraries
import pandas as pd
import gc         

#Loading data
data = pd.read_csv("./data_big_transformed.csv")

#Setting display options
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

#Take a subset of full merged data
data1 = data.loc[data.user_id.isin(data.user_id.drop_duplicates().sample(frac=0.25, random_state=16))] 
data = data1
del data1
gc.collect()

#Look at general information
data.nunique(axis=0)
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
del data, pivot_data, user_data

#Opslaan van CSV file
final_data.to_csv("data_big_one-hot.csv", index = False)
data = final_data
del final_data

data.iloc[1:25, [0, 1, 7, 8, 9, 13]] 
data.iloc[1:25, 3000:3020] 
