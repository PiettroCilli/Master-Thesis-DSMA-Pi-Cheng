#Import data
data = pd.read_csv("C:/Users/Administrator/Documents/Msc Data Science & Marketing Analytics/Master thesis/Data sets/Merged Instacart/data_small_one-hot.csv")

#Load model
model.load_weights(save_dir + '/DEC_model_final.h5')

#Look at the predicted clusters for each observation
q = model.predict(data, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p
y_pred = q.argmax(1)
np.unique(y_pred, return_counts=True)
np.mean(q, axis = 0)
np.mean(p, axis = 0)

# Merge the clusters (predictions) with the original data
clustered_data = data.copy()
clustered_data['Cluster'] = y_pred
distinct_colors = ['#E63946', '#F1FAEE', '#A8DADC']

# Plot the updated bar chart for descriptive statistics
clustered_data.groupby('Cluster').mean().plot(kind='bar', figsize=(15,7), color=distinct_colors)
plt.title('Average values per cluster')
plt.ylabel('Average value')
plt.xlabel('Variable')
plt.legend(title='Cluster', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(axis='y')
plt.tight_layout()

# Plot the updated t-SNE with the new color scheme
plt.figure(figsize=(10, 8))
for i, color in enumerate(distinct_colors):
    plt.scatter(tsne_results[y_pred == i, 0], tsne_results[y_pred == i, 1], c=color, label=f'Cluster {i}', alpha=0.6)
plt.title('t-SNE visualization of clusters')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()