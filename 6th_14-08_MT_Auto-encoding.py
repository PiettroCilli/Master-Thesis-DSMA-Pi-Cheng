#Import modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers, optimizers, backend as K, callbacks
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Layer, InputSpec
from tensorflow.keras.utils import plot_model
import tensorflow as tf

#Load data
data = pd.read_csv('/content/drive/MyDrive/Merged Instacart/data_half_one-hot.csv')
data = data.iloc[:, 5:]


#Setting up neural network architecture, including specifying the amount of latent dimensions
input_dim = data.shape[1]
autoencoder = Sequential([
    Dense(int(input_dim), activation='sigmoid', activity_regularizer=regularizers.l1(1e-7),
          name='encoder'),
    Dense(150, activation='sigmoid', activity_regularizer=regularizers.l1(1e-7),
          name='latent_layer'),
    Dense(int(input_dim), activation='sigmoid', name = "decoder"),
])

#Specify what loss function to use and train the model
autoencoder.compile(optimizer='adam', loss='mse')
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
history = autoencoder.fit(data, data, epochs=200, batch_size=256, verbose=1, validation_split=0.2, callbacks=[early_stopping])
train_mse = history.history['loss'][-1]  # Last training MSE
val_mse = history.history['val_loss'][-1]  # Last validation MSE
ld=150
r=1e-06
print(f"LD: {ld}, Reg: {r}, Train_mse: {train_mse}, Val_mse: {val_mse}")

#If you already have the weights saved, use this code to build the model
#autoencoder.load_weights('/content/drive/MyDrive/Merged Instacart/models_final/ae_weights')
#input_dim = data.shape[1]
#autoencoder.build((None, input_dim))
#autoencoder.summary()
# get the latent layer
#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_layer').output)

#Save the weights, load them etc.
autoencoder.save_weights('/content/drive/MyDrive/Merged Instacart/models_final/ae_weights_1e-06')
autoencoder.load_weights('/content/drive/MyDrive/Merged Instacart/models_final/ae_weights_1e-06')
autoencoder.summary()

# get the latent layer & create some plots to confirm architecture
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_layer').output)
Plotting architecture of the auto-encoding network
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
plot_model(encoder, to_file='encoder.png', show_shapes=True)


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
  kmeans = KMeans(n_clusters= i, n_init=10)
  kmeans.fit(pred)
  scores_2.append(kmeans.inertia_)
plt.figure(figsize=(7,7))
plt.plot(scores_2, 'bx-')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within cluster sum of squares')
plt.show()

#Building DEC model
n_clusters = 3  # adjust as necessary
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

#Use Kullback-Leibler divergence
#model.compile(optimizer=optimizers.Adam(0.0001), loss='kld')
model.compile(optimizer=optimizers.SGD(0.0001, 0.1), loss='kld')

#Step 1: Initialize cluster centers using k-means
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
y_pred = kmeans.fit_predict(encoder.predict(data))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
np.unique(y_pred, return_counts=True)

#Step 2:
loss = 0
index = 0
maxiter = 5000
update_interval = 25
index_array = np.arange(data.shape[0])
tol = 0.00004



# Convert data to numpy for faster slicing
data_np = data.values

# Pre-allocate q array
q = np.zeros((len(data_np), n_clusters))  # Assuming q has shape (data_length, n_clusters)

# Step 3: start training
batch_size = 256
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        # Optimized Predict in batches
        for i in range(0, len(data_np), batch_size):
          q[i:i+batch_size] = model.predict(data_np[i:i+batch_size], verbose=0)

        p = target_distribution(q)  # Update the auxiliary target distribution p

        # Evaluate the clustering performance
        y_pred_last = np.copy(y_pred)
        y_pred = q.argmax(1)

    # Check the tolerance threshold every 100 iterations
    if ite % 100 == 0 and ite > 0:
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        if delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break

    idx = index_array[index * batch_size: min((index+1) * batch_size, len(data_np))]
    loss = model.train_on_batch(x=data_np[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= len(data_np) else 0

    # Monitor Progress
    if ite % 100 == 0:  # Print progress every 100 iterations
        print(f"Iteration {ite}/{maxiter}, Loss: {loss:.5f}")
#Save the model
model.save('/content/drive/MyDrive/Merged Instacart/models_final/DEC_SGD0.001_1e-07.h5')
loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/Merged Instacart/models_final/DEC_SGD0.001_1e-07.h5', custom_objects={'ClusteringLayer': ClusteringLayer})

#Look at the predicted clusters for each observation
q = model.predict(data, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p
y_pred = q.argmax(1)
print(np.unique(y_pred, return_counts=True))
print(np.mean(q, axis = 0))
print(np.mean(p, axis = 0))

#Compute silhouette just to have a look
from sklearn.metrics import silhouette_score
data['cluster'] = y_pred

#Silhouette
silhouette_avg = silhouette_score(data, y_pred)
print("The average silhouette_score is :", silhouette_avg)
