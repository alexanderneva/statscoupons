import pandas as pd
import numpy as np


def extract_time_features(df):
    df = df.copy()
    df["sale_time"] = pd.to_datetime(
        df["sale_time"], format="%Y/%m/%d %H:%M", errors="coerce"
    )
    df["hour"] = df["sale_time"].dt.hour
    df["day_of_week"] = df["sale_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    return df


def load_data():
    print("Loading data files...")

    wallet_df = pd.read_csv("cust_wallet_detail_train.csv")
    coupon_used_df = pd.read_csv("cust_coupon_detail_used_train.csv")
    coupon_send_df = pd.read_csv("cust_coupon_detail_send_train.csv")

    print(f"Wallet data shape: {wallet_df.shape}")
    print(f"Coupon used data shape: {coupon_used_df.shape}")
    print(f"Coupon send data shape: {coupon_send_df.shape}")

    coupon_used_df = coupon_used_df.rename(columns={"transtime": "sale_time"})
    coupon_send_df = coupon_send_df.rename(columns={"voucherstarttime": "sale_time"})

    wallet_df["coupon_used"] = (wallet_df["coupon_amt"] > 0).astype(int)

    coupon_used_agg = (
        coupon_used_df.groupby("membercode")
        .agg({"vouchercode": "count", "couponusemoney": "sum"})
        .rename(
            columns={
                "vouchercode": "coupon_used_count",
                "couponusemoney": "total_coupon_used_amt",
            }
        )
    )

    coupon_send_agg = (
        coupon_send_df.groupby("membercode")
        .agg({"voucherrulecode": "count", "cashvalue": "sum"})
        .rename(
            columns={
                "voucherrulecode": "coupon_send_count",
                "cashvalue": "total_coupon_send_amt",
            }
        )
    )

    df = wallet_df.merge(coupon_used_agg, on="membercode", how="left")
    df = df.merge(coupon_send_agg, on="membercode", how="left")

    df["coupon_used_count"] = df["coupon_used_count"].fillna(0)
    df["total_coupon_used_amt"] = df["total_coupon_used_amt"].fillna(0)
    df["coupon_send_count"] = df["coupon_send_count"].fillna(0)
    df["total_coupon_send_amt"] = df["total_coupon_send_amt"].fillna(0)

    print(f"\nMerged data shape: {df.shape}")

    print(f"\nTarget distribution (coupon_used):")
    print(wallet_df["coupon_used"].value_counts())

    df = extract_time_features(df)

    feature_cols = [
        "station_code",
        "tran_amt",
        "receivable_amt",
        "discounts_amt",
        "point_amt",
        "attributionorgcode",
        "transactionorgcode",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_morning",
        "is_afternoon",
        "is_evening",
        "is_night",
        "coupon_used_count",
        "total_coupon_used_amt",
        "coupon_send_count",
        "total_coupon_send_amt",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]

    df = df[available_cols + ["coupon_used"]].copy()

    df["benefit_ratio"] = (df["discounts_amt"] + df["point_amt"]) / (df["tran_amt"] + 1)
    df["discount_ratio"] = df["discounts_amt"] / (df["receivable_amt"] + 1)
    df["savings_pct"] = (df["discounts_amt"] + df["point_amt"]) / (
        df["receivable_amt"] + 1
    )
    df["tran_to_receivable_ratio"] = df["tran_amt"] / (df["receivable_amt"] + 1)

    engineered_features = [
        "benefit_ratio",
        "discount_ratio",
        "savings_pct",
        "tran_to_receivable_ratio",
    ]

    OPTIMAL_FEATURES = [
        "station_code",
        "attributionorgcode",
        "hour",
        "tran_amt",
        "receivable_amt",
        "tran_to_receivable_ratio",
        "day_of_week",
        "total_coupon_send_amt",
        "coupon_used_count",
        "transactionorgcode",
        "total_coupon_used_amt",
        "coupon_send_count",
        "savings_pct",
        "benefit_ratio",
        "discount_ratio",
    ]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    outlier_cols = ["tran_amt", "receivable_amt", "discounts_amt", "point_amt"]
    Q1 = df[outlier_cols].quantile(0.05)
    Q3 = df[outlier_cols].quantile(0.95)
    mask = ((df[outlier_cols] >= Q1) & (df[outlier_cols] <= Q3)).all(axis=1)
    df = df[mask]
    print(f"\nRemoved outliers: kept {len(df)} of {len(df) / 0.9:.0f} records")

    feature_cols_extended = [f for f in OPTIMAL_FEATURES if f in df.columns]
    print(
        f"\nUsing optimized features ({len(feature_cols_extended)}): {feature_cols_extended}"
    )

    X = df[feature_cols_extended]
    y = df["coupon_used"]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y
