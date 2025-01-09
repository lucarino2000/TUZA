import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import random
from create_dataset import create_dataset

def plot_clusters(clustered_dataset):
    cluster_means = (
    clustered_dataset.groupby(['Latent Risk', 'Cluster'])['Normalised Fees']
    .mean()
    .unstack()
    )

    palette = sns.color_palette("viridis", n_colors=len(clustered_dataset['Cluster'].unique()))
    cluster_colors = dict(zip(cluster_means.columns, palette))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=clustered_dataset, x='Latent Risk', y='Normalised Fees', hue='Cluster', palette=cluster_colors)

    for cluster in cluster_means.columns:
        plt.plot(
            cluster_means.index,
            cluster_means[cluster],
            label=f'Mean - {cluster}',
            marker='o',
            linestyle='--',
            color=cluster_colors[cluster],
        )

    plt.title('Latent Risk vs Normalised Fees with GMM Clustering and Cluster Mean Trends')
    plt.xlabel('Latent Risk')
    plt.ylabel('Normalised Fees')
    plt.legend(title='Cluster and Mean')
    plt.show()

if __name__ == "__main__":

    # Create our data_frame from the data.csv file
    dataset, testset, data_frame = create_dataset()

    # remove outlier so it does not interfere with gmm
    dataset = dataset[~((dataset['Latent Risk'] == 1.8) & (dataset['Normalised Fees'] > 0.03))]

    results = []
    fitted_models = {}
    for risk_value in dataset['Latent Risk'].unique():

        # Fit a 3 point GMM to each subset based on discrete latent risk
        subset = dataset[dataset['Latent Risk'] == risk_value]['Normalised Fees'].values.reshape(-1, 1)

        gmm = GaussianMixture(n_components=3, random_state=13, covariance_type='tied')
        gmm.fit(subset)
        clusters = gmm.predict(subset)

        # Sort the clusters into their labels
        cluster_means = {i: subset[clusters == i].mean() for i in range(3)}
        sorted_clusters = sorted(cluster_means, key=cluster_means.get)
        
        cluster_mapping = {
            sorted_clusters[0]: "Competitive",
            sorted_clusters[1]: "Neutral",
            sorted_clusters[2]: "Un-Competitive",
        }

        fitted_models[risk_value] = {'gmm': gmm, 'cluster_mapping': cluster_mapping}
        labeled_clusters = [cluster_mapping[cluster] for cluster in clusters]

        for fee, cluster_label in zip(subset.flatten(), labeled_clusters):
            results.append({'Latent Risk': risk_value, 'Normalised Fees': fee, 'Cluster': cluster_label})

    # Create a dataframe to view the results
    clustered_dataset = pd.DataFrame(results)

    test_results = []
    for _, row in testset.iterrows():
        latent_risk = row['Latent Risk']
        normalised_fee = row['Normalised Fees']

        if latent_risk in fitted_models:
            model_data = fitted_models[latent_risk]
            gmm = model_data['gmm']
            cluster_mapping = model_data['cluster_mapping']

            predicted_cluster = gmm.predict([[normalised_fee]])[0]
            predicted_label = cluster_mapping[predicted_cluster]
        else:
            predicted_label = None

        test_results.append(predicted_label)

    # Plot the clustered dataset
    plot_clusters(clustered_dataset)

    # Save the test set as a labeled csv to compare to the other model
    testset['Price Label'] = test_results
    testset = testset[['Normalised Fees', 'Price Label']]
    testset['Original Row Index'] = testset.index
    testset = testset[['Original Row Index', 'Normalised Fees', 'Price Label']]

    testset.to_csv("testset_clustering_predictions.csv", index=False)