import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers, backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Layer, InputSpec
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns

#Import data
data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_one-hot.csv")
save_dir = './results'
data = data.iloc[:, 5:]

#Model architecture
input_dim = data.shape[1]
autoencoder = Sequential([
    Dense(int(input_dim), activation='sigmoid', activity_regularizer=regularizers.l1(1e-5),
          name='encoder'),
    Dense(15, activation='sigmoid', activity_regularizer=regularizers.l1(1e-5), 
          name='latent_layer'),
    Dense(int(input_dim), activation='sigmoid', name = "decoder"),
])

#Load weights of this model
autoencoder.load_weights(save_dir + '/ae_weights')
autoencoder.build((None, input_dim))
autoencoder.summary()

#Get latent layer form encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_layer').output)

#Create the clustering layer
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "alpha": self.alpha,
            # "weights" is not included since it's a trainable weight and handled by Keras internally
        })
        return config

#Create the target distribution function
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

#1 Set up DEC model
n_clusters = 3  # adjust as necessary
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

#2 Load weights of DEC model
model.load_weights('./results/DEC_model_final.h5')

#3 Calculate predictions
q = model.predict(data, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p
y_pred = q.argmax(1)
np.unique(y_pred, return_counts=True)
np.mean(q, axis = 0)
np.mean(p, axis = 0)

##Evaluate cluster quality with metrics
data['cluster'] = y_pred

#Silhouette
silhouette_avg = silhouette_score(data, y_pred)
print("The average silhouette_score is :", silhouette_avg)

#Davies bouldin
db_score = davies_bouldin_score(data, y_pred)
print("The Davies-Bouldin score is:", db_score)
#data.to_csv("data_small_analysis.csv", index = False)

## Analyse bases in segmentation
data1 = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_one-hot.csv")
data1 = data1.iloc[:, 1:]
data1['cluster'] = y_pred
distinct_colors = ['#E63946', '#F1FAEE', '#A8DADC']

# Summary for bases
selected_columns = ['avg_days_since_prior_order', 'u_reordered_ratio', 'u_total_orders', 'order_size_avg', 'cluster']
cluster_summary = data1[selected_columns].groupby('cluster').agg(['mean']).transpose()

# Plot the selected columns
cluster_summary.plot(kind='bar', figsize=(15,7), rot=0)
plt.title('Cluster Mean Values for Selected Columns')
plt.ylabel('Mean Value')
plt.xlabel('Cluster')
plt.show()

#del cluster_summary, clustering_layer, db_score, distinct_colors, input_dim, n_clusters, p, q, selected_columns, silhouette_avg, data1

## Load aisles, departments & products & start analysis
aisles = pd.read_csv('C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/aisles.csv')
departments = pd.read_csv('C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/departments.csv')
products = pd.read_csv('C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/products.csv')
merged_products = pd.merge(products, aisles, on='aisle_id', how='left')
merged_products = pd.merge(merged_products, departments, on='department_id', how='left')

# Create a mapping from product_name to aisle and department
product_to_aisle = dict(zip(merged_products['product_name'], merged_products['aisle']))
product_to_department = dict(zip(merged_products['product_name'], merged_products['department']))

# Extract the product columns from the original data (ignoring 'cluster' column for now)
product_columns = [col for col in data.columns if col != 'cluster']

# Create mappings for aisles and departments based on the product columns in the original data
aisles_in_data = {product: product_to_aisle.get(product, 'Unknown') for product in product_columns}
departments_in_data = {product: product_to_department.get(product, 'Unknown') for product in product_columns}

# Count the number of products in each aisle and department
aisle_counts = pd.Series(list(aisles_in_data.values())).value_counts()
department_counts = pd.Series(list(departments_in_data.values())).value_counts()
aisle_counts.head()
department_counts.head()

# Set the style for the plots
sns.set(style="whitegrid")

# Group the data by cluster and compute the mean for each product
cluster_means = data.groupby('cluster').mean()

# Plot the top 10 products for each cluster
for cluster in cluster_means.index:
    plt.figure(figsize=(12, 6))
    top_10_products = cluster_means.loc[cluster].sort_values(ascending=False).head(10)
    sns.barplot(x=top_10_products.index, y=top_10_products.values, palette='coolwarm')
    plt.title(f'Top 10 Products in Cluster {cluster}')
    plt.ylabel('Mean Value')
    plt.xlabel('Product')
    plt.xticks(rotation=90)
    plt.show()

# Create a DataFrame to hold the aisle information
data_aisles = data.copy()
for product, aisle in aisles_in_data.items():
    if product in data.columns:
        data_aisles[aisle] = data_aisles.get(aisle, 0) + data_aisles[product]

# Drop the original product columns to keep only aisles and 'cluster'
data_aisles.drop(columns=product_columns, inplace=True)

# Group the data by cluster and compute the mean for each aisle
cluster_aisle_means = data_aisles.groupby('cluster').mean()

# Plot the top 10 aisles for each cluster
for cluster in cluster_aisle_means.index:
    plt.figure(figsize=(12, 6))
    top_10_aisles = cluster_aisle_means.loc[cluster].sort_values(ascending=False).head(10)
    sns.barplot(x=top_10_aisles.index, y=top_10_aisles.values, palette='coolwarm')
    plt.title(f'Top 10 Aisles in Cluster {cluster}')
    plt.ylabel('Mean Value')
    plt.xlabel('Aisle')
    plt.xticks(rotation=90)
    plt.show()

# Recreate the DataFrame to hold the department information
data_departments = data.copy()
for product, department in departments_in_data.items():
    if product in data.columns:
        data_departments[department] = data_departments.get(department, 0) + data_departments[product]

# Drop the original product columns to keep only departments and 'cluster'
data_departments.drop(columns=product_columns, inplace=True)

# Group the data by cluster and compute the mean for each department
cluster_department_means = data_departments.groupby('cluster').mean()

# Plot the top 10 departments for each cluster
for cluster in cluster_department_means.index:
    plt.figure(figsize=(12, 6))
    top_10_departments = cluster_department_means.loc[cluster].sort_values(ascending=False).head(10)
    sns.barplot(x=top_10_departments.index, y=top_10_departments.values, palette='coolwarm')
    plt.title(f'Top 10 Departments in Cluster {cluster}')
    plt.ylabel('Mean Value')
    plt.xlabel('Department')
    plt.xticks(rotation=90)
    plt.show()

# Identify the top 5 and bottom 5 products for each cluster based on mean values
top_bottom_products_by_cluster = {}

for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster'] == cluster]
    mean_values = cluster_data[product_columns].mean().sort_values(ascending=False)
    top_5_products = mean_values.head(5).index.tolist()
    bottom_5_products = mean_values.tail(5).index.tolist()
    top_bottom_products_by_cluster[cluster] = {'Top 5 Products': top_5_products, 'Bottom 5 Products': bottom_5_products}

top_bottom_products_by_cluster

# Identify the top 5 and bottom 5 aisles for each cluster based on mean values
top_bottom_aisles_by_cluster = {}

for cluster in data_aisles['cluster'].unique():
    cluster_data = data_aisles[data_aisles['cluster'] == cluster]
    mean_values = cluster_data.drop(columns=['cluster']).mean().sort_values(ascending=False)
    top_5_aisles = mean_values.head(5).index.tolist()
    bottom_5_aisles = mean_values.tail(5).index.tolist()
    top_bottom_aisles_by_cluster[cluster] = {'Top 5 Aisles': top_5_aisles, 'Bottom 5 Aisles': bottom_5_aisles}

top_bottom_aisles_by_cluster

# Identify the top 5 and bottom 5 departments for each cluster based on mean values
top_bottom_departments_by_cluster = {}

for cluster in data_departments['cluster'].unique():
    cluster_data = data_departments[data_departments['cluster'] == cluster]
    mean_values = cluster_data.drop(columns=['cluster']).mean().sort_values(ascending=False)
    top_5_departments = mean_values.head(5).index.tolist()
    bottom_5_departments = mean_values.tail(5).index.tolist()
    top_bottom_departments_by_cluster[cluster] = {'Top 5 Departments': top_5_departments, 'Bottom 5 Departments': bottom_5_departments}

top_bottom_departments_by_cluster


# Step 1: Obtain the latent dimensions (encoded representations)
# Replace 'encoder_model' with the name of your encoder model and 'data' with your input data
data.pop(data.columns[-1])
latent_dimensions = encoder.predict(data)

# Step 2: Apply t-SNE to the latent dimensions
tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(latent_dimensions)

# Step 3: Create a DataFrame for plotting
df_latent_tsne = pd.DataFrame(latent_tsne, columns=['Dimension 1', 'Dimension 2'])
df_latent_tsne['Cluster'] = y_pred  # y_pred contains the cluster labels

# Step 4: Plot the t-SNE output
plt.figure(figsize=(10, 8))
for cluster in np.unique(y_pred):
    mask = df_latent_tsne['Cluster'] == cluster
    plt.scatter(df_latent_tsne[mask]['Dimension 1'], df_latent_tsne[mask]['Dimension 2'], label=f'Cluster {cluster}', alpha=0.6)
plt.title('t-SNE visualization of clusters in latent space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
