import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import load_data


def main():
    print("Loading data...")
    X, y = load_data()

    # Add station_prefix as a new feature
    # Read raw data to get station_code
    wallet_df = pd.read_csv("cust_wallet_detail_train.csv")
    wallet_df = wallet_df.iloc[X.index]  # Align with X

    X_with_prefix = X.copy()
    X_with_prefix["station_prefix"] = (
        wallet_df["station_code"].astype(str).str[:3].values
    )

    # Encode prefix as numeric
    prefix_map = {p: i for i, p in enumerate(X_with_prefix["station_prefix"].unique())}
    X_with_prefix["station_prefix_encoded"] = X_with_prefix["station_prefix"].map(
        prefix_map
    )

    print(f"Station prefixes: {list(prefix_map.keys())[:10]}...")

    # Use station_prefix and tran_amt
    features = ["tran_amt", "station_prefix_encoded"]
    X_feat = X_with_prefix[features].copy()

    X_sample, _, y_sample, _ = train_test_split(
        X_feat.values, y.values, train_size=5000, random_state=42, stratify=y.values
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e"]
    for label, color, name in [
        (0, colors[0], "No Coupon"),
        (1, colors[1], "Used Coupon"),
    ]:
        mask = y_sample == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.4, s=15, label=name)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: Transaction Amount + Station Prefix")
    ax.legend(title="Target")

    plt.tight_layout()
    plt.savefig("images/pca_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved pca_visualization.png")


if __name__ == "__main__":
    main()
