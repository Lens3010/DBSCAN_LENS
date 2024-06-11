dataset = StandardScaler().fit_transform(dataset)

numero_dimensioni = dataset.shape[1]


k = numero_dimensioni + 1
nbrs = NearestNeighbors(n_neighbors=k).fit(dataset)

distances, indices = nbrs.kneighbors(dataset)

distances = np.sort(distances[:,k-1], axis=0)
plt.plot(distances)
plt.xlabel('Points')
plt.ylabel('Distances')
plt.show()

dataset = StandardScaler().fit_transform(dataset)

# eps = 3 Numero che si vede dall'Elbow Method
min_samples = numero_dimensioni *2 


clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
labels = clustering.labels_


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

#Plotto
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = dataset[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = dataset[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()
