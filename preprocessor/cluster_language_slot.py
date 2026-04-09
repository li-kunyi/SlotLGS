import os
import glob
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


def load_all_features(folder):
    feat_files = glob.glob(os.path.join(folder, "*_feats.npy"))
    
    if len(feat_files) == 0:
        raise ValueError("No *_feats.npy files found!")

    all_feats = []
    for f in feat_files:
        feats = np.load(f)  # shape: (N, 512)
        all_feats.append(feats)

    all_feats = np.concatenate(all_feats, axis=0)
    print(f"Loaded {len(feat_files)} files, total features: {all_feats.shape}")

    return all_feats


def cluster_features(X, eps=0.1, min_samples=8):
    print("Clustering with DBSCAN...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(X)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Cluster num:", num_clusters)

    return labels, num_clusters


def compute_cluster_centers(X, labels):
    cluster_feats = []

    for label in set(labels):
        if label == -1:
            continue  # skip noise
        
        cluster_points = X[labels == label]
        center = cluster_points.mean(axis=0)  # (512,)
        cluster_feats.append(center)

    cluster_feats = np.stack(cluster_feats, axis=0)
    print("Cluster feature shape:", cluster_feats.shape)

    return cluster_feats


def clustering(folder):
    # 1. load
    X = load_all_features(folder)

    # 2. normalize
    X = normalize(X, axis=1)

    # 3. clustering
    labels, num_clusters = cluster_features(X)

    # 4. cluster centers
    cluster_feats = compute_cluster_centers(X, labels)

    # 5. save
    save_path = os.path.join(folder, "cluster_feats.npy")
    np.save(save_path, cluster_feats)

    print(f"Saved cluster features to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()

    clustering(args.dataset_path)