#Load libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

#Import data
data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_one-hot.csv")

#Standardize variables
scaler = MinMaxScaler()
columns_to_normalize = ["avg_days_since_prior_order", "u_reordered_ratio", "u_total_orders", "order_size_avg"]
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
del columns_to_normalize, scaler

#drop columns and create backup
#data1 = data
#data = data.drop(columns = ["avg_days_since_prior_order", "u_reordered_ratio", "u_total_orders", "order_size_avg"])
# data.to_csv("data_ex_one-hot.csv", index = False)

# Make data ready
data = data.drop('user_id', axis=1)

# Split the data into training, validation, and test sets
X_trainval, X_test = train_test_split(data, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_trainval, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Grid search over latent dimensions
grid =  [(ld, r) for ld in [5, 10, 15] for r in [1e-4, 1e-5, 1e-6]]  # adjust as necessary
results = []
input_dim = data.shape[1]
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

for ld, r in grid:
    # Define the autoencoder based on your architecture
    autoencoder = models.Sequential([
        layers.Dense(input_dim, activation='sigmoid', activity_regularizer=regularizers.l1(r), name='encoder'),
        layers.Dense(ld, activation='sigmoid', activity_regularizer=regularizers.l1(r), name='latent_layer'),
        layers.Dense(input_dim, activation='sigmoid', name='decoder')
    ])

    # Compile and train
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, 
                    epochs=200,  # set higher epochs since early stopping will interrupt when necessary
                    batch_size=round(X_train.shape[0]/10), 
                    verbose=0,
                    validation_data=(X_val, X_val),
                    callbacks=[early_stopping])

    # Evaluate
    X_train_reconstructed = autoencoder.predict(X_train)
    X_val_reconstructed = autoencoder.predict(X_val)
    train_mse = mean_squared_error(X_train, X_train_reconstructed)
    val_mse = mean_squared_error(X_val, X_val_reconstructed)

    print(f"LD: {ld}, Reg: {r}, Train MSE: {train_mse:.5f}, Val MSE: {val_mse:.5f}")
    results.append((ld, r, train_mse, val_mse))

# Unpack the results
ld, r, training_mses, val_mses = zip(*results)
unique_ld = sorted(set(ld))
unique_r = sorted(set(r))

# Create unique lists of the latent dimensions and sparsity parameters for the plots
latent_dims = list(set(ld))
sparsity_parameters = list(set(r))

# Plotting
for reg in unique_r:
    filter = np.array(r) == reg
    plt.plot(np.array(ld)[filter], np.array(training_mses)[filter], 'o-', label=f'Train Loss {reg}')
    plt.plot(np.array(ld)[filter], np.array(val_mses)[filter], 's-', label=f'Val Loss {reg}')

plt.title('Model Loss vs. Latent Dimensions')
plt.ylabel('MSE')
plt.xlabel('Latent Dimensions')
plt.legend(title='Regularization', loc='upper right')
plt.xticks(unique_ld)
plt.grid(True)
plt.show()

# Further exploration if needed
X_val_reconstructed_df = pd.DataFrame(X_val_reconstructed)
print(X_val_reconstructed_df.info())
print(X_val_reconstructed_df.describe())
#Latent dimensions seem not to matter so much and sparsity seems to be fine at 1e-05


line_styles = ['-', '--', '-.', ':']
for index, reg in enumerate(unique_r):
    filter = np.array(r) == reg
    plt.plot(np.array(ld)[filter], np.array(training_mses)[filter], line_styles[index], linewidth=2, alpha=0.8, 
             marker='o', label=f'Train Loss {reg}')
    plt.plot(np.array(ld)[filter], np.array(val_mses)[filter], line_styles[index], linewidth=2, alpha=0.8, 
             marker='s', label=f'Val Loss {reg}')

plt.title('Model Loss vs. Latent Dimensions and sparsity parameters')
plt.ylabel('MSE')
plt.xlabel('Latent Dimensions')
plt.legend(title='Regularization', loc='upper right')
plt.xticks(unique_ld)
plt.grid(True)
plt.tight_layout()  # Adjusts the plot to make sure everything fits
plt.show()




