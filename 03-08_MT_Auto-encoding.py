import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils import plot_model

#Import data
data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_one-hot.csv")

#Standardize variables and remove unwanted variables
scaler = MinMaxScaler()
columns_to_normalize = ["avg_days_since_prior_order", "u_reordered_ratio", "u_total_orders", "order_size_avg"]
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
del columns_to_normalize, scaler
data = data.drop('user_id', axis=1)
#data = data.iloc[:, 4:]


#Setting up neural network architecture, including specifying the amount of latent dimensions
input_dim = data.shape[1]
autoencoder = Sequential([
    Dense(int(input_dim), activation='sigmoid', activity_regularizer=regularizers.l1(1e-5),
          name='encoder'),
    Dense(15, activation='sigmoid', activity_regularizer=regularizers.l1(1e-5), 
          name='latent_layer'),
    Dense(int(input_dim), activation='sigmoid', name = "decoder"),
])

#Specify what loss function to use and train the model
autoencoder.compile(optimizer='adam', loss='mse')
save_dir = './results'
autoencoder.fit(data, data, epochs=50, batch_size=round(data.shape[0]/10), verbose=0)

#Save weights and load weights
autoencoder.save_weights(save_dir + '/ae_weights')
autoencoder.load_weights(save_dir + '/ae_weights')
autoencoder.summary()

# get the latent layer
latent_layer = autoencoder.get_layer('latent_layer').output
encoder = Model(inputs=autoencoder.input, outputs=latent_layer)

#Plotting architecture of the auto-encoding network
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
plot_model(encoder, to_file='autoencoder.png', show_shapes=True)
encoder.output

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

#Create the encoded data
pred = encoder.predict(data)
pred1 = pd.DataFrame(pred)
pred1.describe()

#Elbow plot to choose the number of clusters to use in DEC
scores_2 = []
range_values = range(1, 15)
for i in range_values:
  kmeans = KMeans(n_clusters= i)
  kmeans.fit(pred)
  scores_2.append(kmeans.inertia_)
plt.figure(figsize=(10,10))
plt.plot(scores_2, 'bx-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('scores') 
plt.show()

#Define number of clusters and implement clustering layer in model
n_clusters = 3  # adjust as necessary
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

#Plot architecture of the model so far (from input to latent layer to clusters)
plot_model(model, to_file='model.png', show_shapes=True)

#Use Kullback-Leibler divergence
model.compile(optimizer=SGD(0.0001, 0), loss='kld')

#Step 1: Initialize cluster centers using k-means
kmeans = KMeans(n_clusters=n_clusters, n_init=3)
y_pred = kmeans.fit_predict(encoder.predict(data))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
np.unique(y_pred, return_counts=True)

#Step 2: 
loss = 0
index = 0
maxiter = 2000
update_interval = 100
index_array = np.arange(data.shape[0])
tol = 0.00001

#Step 3: start training
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(data, verbose=0)
        p = target_distribution(q)  # Update the auxiliary target distribution p

        # Evaluate the clustering performance
        y_pred_last = np.copy(y_pred)
        y_pred = q.argmax(1)
        
        # Check if reached tolerance threshold
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * round(data.shape[0]/10): min((index+1) * round(data.shape[0]/10), data.shape[0])]
    loss = model.train_on_batch(x=data.iloc[idx], y=p[idx])
    index = index + 1 if (index + 1) * round(data.shape[0]/10) <= data.shape[0] else 0
model.save_weights(save_dir + '/DEC_model_final.h5')
model.load_weights(save_dir + '/DEC_model_final.h5')

#Look at the predicted clusters for each observation
q = model.predict(data, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p
y_pred = q.argmax(1)
np.unique(y_pred, return_counts=True)
np.mean(q, axis = 0)
np.mean(p, axis = 0)
