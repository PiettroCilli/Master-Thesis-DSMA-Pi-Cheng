#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
                    
#Loading datasets
aisles = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/aisles.csv") 
departments = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/departments.csv") 
orders_prior = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/order_products__prior.csv") 
orders = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/orders.csv") 
products = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/products.csv") 

#Setting indices as 0
orders.set_index('order_id', inplace=True)
products.set_index('product_id', inplace=True)
aisles.set_index('aisle_id', inplace=True)
departments.set_index('department_id', inplace=True)

# Setting the NaNs to 0. i.e. sets the days since order to 0 for 1st orders
orders = orders.fillna(0) 

#Transform characters to categorical variables
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')

#Merging the datasets
merged = pd.merge(orders, orders_prior, on = "order_id" )
merged2 = pd.merge(merged, products, on = "product_id" )
merged3 = pd.merge(merged2, aisles, on = "aisle_id" )
merged4 = pd.merge(merged3, departments, on = "department_id" )

#Naming the new merged dataset
data = merged4

##Delete objects to free space
del(aisles, departments, merged, merged2, merged3, merged4, orders, orders_prior, products)

#Save as CSV file
data.to_csv("data.csv", index = False)

#Take subset of all customers
#data = data.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.1, random_state=25))] 




# =============================================================================
# #Creating plots to check things
# plt.plot(check["p_total_purchases"], check["p_reorder_ratio"])
# plt.xlabel("Amount of times a product is bought") # Text for X-Axis
# plt.ylabel("Reorder ratio") # Text for Y-Axis
# plt.xlim(0, 10000)
# plt.ylim(0.75, 1)
# plt.show()
# 
# plt.plot(check["p_total_purchases"], check["p_reorder_ratio"])
# plt.xlabel("Amount of times a product is bought") # Text for X-Axis
# plt.ylabel("Reorder ratio") # Text for Y-Axis
# plt.xlim(0, 1000)
# plt.ylim(0.75, 1)
# plt.show()
# 
# plt.plot(check["p_total_purchases"], check["p_reorder_ratio"])
# plt.xlabel("Amount of times a product is bought") # Text for X-Axis
# plt.ylabel("Reorder ratio") # Text for Y-Axis
# plt.xlim(0, 250)
# plt.ylim(0.75, 1)
# plt.show()
# 
# =============================================================================






