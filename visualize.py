import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from data_loader import load_data


def main():
    print("Loading data...")
    X, y = load_data()

    df = X.copy()
    df["coupon_used"] = y

    print("Running LDA...")
    X_sample, _, y_sample, _ = train_test_split(
        X.values, y.values, train_size=5000, random_state=42, stratify=y.values
    )
    feature_names = X.columns.tolist()

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_sample, y_sample)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)

    coupon_used_mask = y_sample == 1
    coupon_not_used_mask = y_sample == 0

    tran_amt_idx = feature_names.index("tran_amt") if "tran_amt" in feature_names else 1
    hour_idx = feature_names.index("hour") if "hour" in feature_names else 7

    xy_used = np.vstack(
        [X_sample[coupon_used_mask, tran_amt_idx], X_sample[coupon_used_mask, hour_idx]]
    ).T
    kde_used = KernelDensity(bandwidth=5.0)
    kde_used.fit(xy_used)

    x_min, x_max = X_sample[:, tran_amt_idx].min(), X_sample[:, tran_amt_idx].max()
    y_min, y_max = X_sample[:, hour_idx].min(), X_sample[:, hour_idx].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
    z_used = np.exp(kde_used.score_samples(xy_grid)).reshape(xx.shape)

    xy_not_used = np.vstack(
        [
            X_sample[coupon_not_used_mask, tran_amt_idx],
            X_sample[coupon_not_used_mask, hour_idx],
        ]
    ).T
    kde_not_used = KernelDensity(bandwidth=5.0)
    kde_not_used.fit(xy_not_used)
    z_not_used = np.exp(kde_not_used.score_samples(xy_grid)).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].contourf(xx, yy, z_not_used, levels=15, cmap="Blues", alpha=0.4)
    axes[0].contourf(xx, yy, z_used, levels=15, cmap="Oranges", alpha=0.4)
    axes[0].contour(
        xx, yy, z_not_used, levels=5, colors="blue", linewidths=0.8, linestyles="dashed"
    )
    axes[0].contour(
        xx,
        yy,
        z_used,
        levels=5,
        colors="darkorange",
        linewidths=0.8,
        linestyles="dashed",
    )
    axes[0].set_xlabel("Transaction Amount")
    axes[0].set_ylabel("Hour of Day")
    axes[0].set_title("Density: Coupon Used vs Not Used")
    axes[0].legend(
        handles=[
            matplotlib.lines.Line2D(
                [0], [0], color="darkorange", linestyle="dashed", label="Coupon Used"
            ),
            matplotlib.lines.Line2D(
                [0], [0], color="blue", linestyle="dashed", label="Coupon Not Used"
            ),
        ],
        loc="upper right",
    )

    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y_sample,
        palette={0: "#1f77b4", 1: "#ff7f0e"},
        alpha=0.6,
        ax=axes[1],
    )
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].set_title("t-SNE Visualization")
    axes[1].legend(title="Target", labels=["No Coupon", "Used Coupon"])

    plt.tight_layout()
    plt.savefig("images/dimensionality_reduction.png", dpi=150, bbox_inches="tight")
    print("Saved dimensionality_reduction.png")

    print("Generating feature importance...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_sample, y_sample)
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=True
    )
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_xlabel("Feature Importance")
    ax2.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=150, bbox_inches="tight")
    print("Saved feature_importance.png")

    print("Generating correlation matrix...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "coupon_used"]
    corr_matrix = df[numeric_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax_corr,
    )
    ax_corr.set_title("Feature Correlation Matrix", fontsize=10)
    ax_corr.set_xticklabels(
        ax_corr.get_xticklabels(), fontsize=6, rotation=45, ha="right"
    )
    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), fontsize=6, rotation=0)
    plt.tight_layout()
    plt.savefig("images/correlation_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved correlation_matrix.png")

    print("Generating LDA scatter plot...")
    X_2d = df[["tran_amt", "station_code"]].fillna(0).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    lda_2d = LDA(n_components=1)
    lda_2d.fit(X_2d_scaled, y)

    X_sample_2d, _, y_sample_2d, _ = train_test_split(
        X_2d_scaled, y.values, train_size=5000, random_state=42, stratify=y.values
    )

    from matplotlib.colors import ListedColormap

    colors = ListedColormap(["#1f77b4", "#ff7f0e"])
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    axes3[0].scatter(
        X_sample_2d[:, 0],
        X_sample_2d[:, 1],
        c=y_sample_2d,
        alpha=0.5,
        s=20,
        cmap=colors,
    )
    axes3[0].set_xlabel("Transaction Amount (standardized)")
    axes3[0].set_ylabel("Station Code (standardized)")
    axes3[0].set_title("Data Distribution: Transaction vs Station Code")
    y_min, y_max = X_sample_2d[:, 1].min(), X_sample_2d[:, 1].max()
    axes3[0].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    axes3[0].scatter([], [], c="#1f77b4", label="No Coupon")
    axes3[0].scatter([], [], c="#ff7f0e", label="Used Coupon")
    axes3[0].legend(title="Target")

    lda_coef = lda_2d.coef_[0]
    axes3[1].bar(["Transaction Amount", "Discount Amount"], lda_coef, color="steelblue")
    axes3[1].set_ylabel("LDA Coefficient")
    axes3[1].set_title("LDA Coefficients")

    plt.tight_layout()
    plt.savefig("images/lda_tran_amt_discounts_amt.png", dpi=150, bbox_inches="tight")
    print("Saved lda_tran_amt_discounts_amt.png")


if __name__ == "__main__":
    main()
