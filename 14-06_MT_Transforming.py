#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import gc                

# #Load full complete merged data
# data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data.csv")
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


#Take a subset of full merged data
# data1 = data.loc[data.user_id.isin(data.user_id.drop_duplicates().sample(frac=0.003, random_state=16))] 
# data = data1
# del data1
# data.to_csv("data_small_original.csv", index = False)


#Load subset of full merged original dataset
data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_original.csv")
#Set options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)

##Investigate data##

#General look
data.info()
data.describe(include = 'all') #49677 unique products originally
data.iloc[6:18, [0, 1, 7, 8, 9, 13]] 

#Amount of unique orders, persons and products
len(data['order_id'].unique())
len(data['product_id'].unique())
len(data['user_id'].unique())

# Sort values by user_id and order_number
data = data.sort_values(['user_id', 'product_id', 'order_number'])

##Creating predictors variables##

# #Transform numbers to weekdays
# weekday_map = {0:'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
# data['order_dow'] = data['order_dow'].map(weekday_map)
# del weekday_map

#Create spread by taking the difference in order spread
data['order_diff'] = data.groupby(['user_id', 'product_id'])['order_number'].diff().fillna(0) - 1
spread_data = data.groupby(['user_id', 'product_id'])['order_diff'].sum().reset_index()
spread_data.rename(columns={'order_diff': 'spread'}, inplace=True)
spread_data['spread'] = spread_data['spread'].apply(lambda x: 0 if x < 0 else x)

#Mean days since last order 
user = data.groupby('user_id')[['days_since_prior_order']].mean()
user.columns=['avg_days_since_prior_order']

#Total amount of orders per user, removed 934 customers
user1 = data.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user1 = user1[user1['u_total_orders'] >= 3]
user1 = user1.reset_index()

#Reorder percentage for each product 
u_reorder = data.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio')
u_reorder = u_reorder.reset_index()

#The amount of products per order per user
order_size= data.groupby(['user_id', 'order_id'])['product_id'].count()
order_size = order_size.to_frame()
order_size.columns=['size']

#The average amount of products ordered per user 
results= order_size.groupby('user_id').mean()
results.columns= ['order_size_avg']

#Merging the predictor variables
user = user.merge(u_reorder, on='user_id')
del u_reorder
user = user.merge(user1, on='user_id', how='inner')
del user1
user = user.merge(results, on='user_id', how='inner')
del results 
del order_size
gc.collect()


#Frequency of products bought per product
prd = data.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd = prd.reset_index()

#Products probability of being reordered
p_reorder = data.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')
p_reorder = p_reorder.reset_index()

#Quick check if everything went well actually
# =============================================================================
# check = prd.merge(p_reorder, on="product_id")
# check.describe()
# plt.plot(check["p_total_purchases"], check["p_reorder_ratio"])
# plt.xlabel("Amount of times a product is bought") # Text for X-Axis
# plt.ylabel("Reorder ratio") # Text for Y-Axis
# plt.xlim(0, 10000)
# plt.ylim(0, 1)
# plt.show()
# =============================================================================

#Average position a product is added in the basket
avg_pos = data.groupby('product_id')[['add_to_cart_order']].mean()
avg_pos.columns=['mean_add_to_cart_order']

#Merging 
prd = prd.merge(p_reorder, on='product_id', how='left')
del p_reorder
gc.collect()


#Merging
prd = prd.merge(avg_pos, on='product_id', how='left')
del avg_pos
gc.collect()


##Create extra variables for stickyness##

#How many product each user bought
uxp = data.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp = uxp.reset_index()


#How many times a customer bought a product (almost same as code above)
times = data.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']


#The total amount of orders for each user
total_orders = data.groupby('user_id')['order_number'].max().to_frame('total_orders')


#The order number a customer bought a product for the first time
first_order_no = data.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()


#Merging
span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1

#Creating dataframe sticky
sticky = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
sticky = sticky.merge(spread_data, on=['user_id', 'product_id'], how = 'left')


#Create the variable sticky
sticky['sticky'] = 0.5*(sticky.Times_Bought_N / sticky.total_orders) + 0.5*(sticky.Times_Bought_N/(sticky.Order_Range_D + sticky.spread))

##Start deleting unnessary objects

#Drop all other uncessary objects
sticky = sticky.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D', 'spread'], axis=1)

#Remove other objects we don't need
del [first_order_no, span, times, total_orders, spread_data]


##Continue merging the datasets

#Merging products bought and sticky
uxp = uxp.merge(sticky, on=['user_id', 'product_id'], how='left')
del sticky
temp = uxp.merge(user, on='user_id', how='left')
temp = temp.merge(prd, on='product_id', how='left')


# #Continue the days a product was most often bought on
# dow_product = data.groupby(['user_id', 'product_id'])['order_dow'].agg(lambda x: x.value_counts().idxmax())
# dow_product  = dow_product.reset_index()
# dow_product

# #Merging
# temp = temp.merge(dow_product, on=['user_id', 'product_id'], how='left')

#Deleting some frames I don't need anymore
del uxp, user, prd

##Changes to product, department and aisles
#department and aisle
aisles = pd.read_csv("./aisles.csv") 
departments = pd.read_csv("./departments.csv") 
products = pd.read_csv("./products.csv")

#Merge aisle and department
temp = temp.merge(products, on=['product_id'], how='left')

#Changing product_id to product_name
temp["product_id"] = temp["product_name"]
del temp['product_name'] 
temp.rename(columns = {'product_id':'product_name'}, inplace = True)

#Adding aisle and department names
temp = temp.merge(aisles, on=['aisle_id'], how='left')
temp = temp.merge(departments, on=['department_id'], how='left')

#Deleting more variables in dataframe
del temp['aisle_id'] 
del temp['department_id']

#Delete unnecessary data frames
del aisles, data, departments, products
gc.collect()

#change name of dataframe to data
data = temp
del temp

data.info()

#Saving data to CSV as new data
data.to_csv("data_big_transformed.csv", index = False)
data.nunique()


