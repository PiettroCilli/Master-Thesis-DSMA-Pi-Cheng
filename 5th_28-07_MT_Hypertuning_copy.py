#Mount on G Drive
from google.colab import drive
drive.mount('/content/drive')

#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers, callbacks
from tensorflow.keras.backend import clear_session; clear_session()

#Load data from G drive
data = pd.read_csv('/content/drive/MyDrive/Merged Instacart/data_half_one-hot.csv')
data = data.iloc[:, 5:]

#Parameters to train
grid = [(ld, r) for ld in [150] for r in [1e-08, 1e-09]]
input_dim = data.shape[1]
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

#Save results in
mse_results = []

#Running the hypertuning process
for ld, r in grid:
    autoencoder = Sequential([
        Dense(input_dim, activation='sigmoid', activity_regularizer=regularizers.l1(r), name='encoder'),
        Dense(ld, activation='sigmoid', activity_regularizer=regularizers.l1(r), name='latent_layer'),
        Dense(input_dim, activation='sigmoid', name='decoder')
    ])

    #Autoencoder training
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(data, data, epochs=200, batch_size=256, verbose=0, callbacks=[early_stopping], validation_split=0.2)

    #Calculate MSEs
    train_mse = history.history['loss'][-1]  # Last training MSE
    val_mse = history.history['val_loss'][-1]  # Last validation MSE

    #Create latent dimensions
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_layer').output)
    data_encoded = encoder.predict(data)

    #Encoded data
    #filename = f'/content/drive/MyDrive/Merged Instacart/models_final/data_150_{r}'
    #np.save(filename, data_encoded)

    #Calculate  MSEs
    mse_results.append((ld, r, train_mse, val_mse))
    print(f"LD: {ld}, Reg: {r}, Train_mse: {mse_results[-1][2]:.10f}, Val_mse: {mse_results[-1][3]:.10f}")

    # Clear Keras session to free up resources
    clear_session()

print(mse_results)

# Load the latent representations from the provided files
filepaths = [
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_0.001.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_0.0001.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_1e-05.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_1e-06.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_1e-07.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_1e-08.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_1e-09.npy'
]

# Names for the latent representations based on regularization strengths
names = ["r=1e-03", "r=1e-04", "r=1e-05", "r=1e-06", "r=1e-07", "r=1e-08", "r=1e-09"]

# Load the representations into a dictionary
latent_data = {name: np.load(path) for name, path in zip(names, filepaths)}

# Load the latent representations from the provided files
filepaths = [
    '/content/drive/MyDrive/Merged Instacart/models_final/data_150_1e-07.npy',
    '/content/drive/MyDrive/Merged Instacart/models_final/data_alt_1e-07.npy'
]

# Names for the latent representations based on regularization strengths
names = ["r=1e-05", "r=1e-05_alt"]

# Load the representations into a dictionary
latent_data = {name: np.load(path) for name, path in zip(names, filepaths)}

#Create plots of distribution of values across latent spaces
plt.figure(figsize=(18, 14))
for i, (name, data) in enumerate(latent_data.items(), 1):
    # Flatten the encoded representation
    flattened_encoded = data.flatten()

    # Plot histogram
    plt.subplot(3, 3, i)
    plt.hist(flattened_encoded, bins=100, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Values for {name}", fontsize=16, pad=20)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 14))

#Create plots for activation across features
for i, (name, data) in enumerate(latent_data.items(), 1):
    # Calculate mean activations for the encoded representation
    mean_activations = np.mean(data, axis=0)

    # Plot histogram
    plt.subplot(3, 3, i)
    plt.hist(mean_activations, bins=50, color='lightcoral', edgecolor='black')
    plt.title(f"Mean Activation Distribution for {name}", pad=20, fontsize=16)
    plt.xlabel("Mean Activation Value", fontsize=14)
    plt.ylabel("Number of Features (Neurons)", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Dynamic threshold based on the maximum value of each latent representation
threshold_percentage = 0.05
dynamic_sparsity_percentages = {}

for name, data in latent_data.items():
    dynamic_threshold = threshold_percentage * np.max(data)
    sparsity_percentage = np.mean(np.abs(data) < dynamic_threshold) * 100
    dynamic_sparsity_percentages[name] = sparsity_percentage

dynamic_sparsity_percentages

# Variance analysis
variances = {}
for name, data in latent_data.items():
    var = np.var(data)
    variances[name] = var
print(variances)
print(np.var(data))
