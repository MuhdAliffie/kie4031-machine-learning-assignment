from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
    r2_score,
    PrecisionRecallDisplay
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "SystemCodeNumber",
    "Capacity",
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "cluster",
]

CLUSTER_FEATURES = [
    "Capacity",
    "Occupancy",
    "AvailableSpots",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
]

# --- 1. DATA PREPARATION ---

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
    df = df.dropna(subset=["LastUpdated", "SystemCodeNumber", "Capacity", "Occupancy"])

    df["AvailableSpots"] = df["Capacity"] - df["Occupancy"]
    df = df[df["AvailableSpots"] >= 0]

    # IMPROVEMENT: Define "Full" as having 5% or less capacity remaining
    FULL_THRESHOLD = 0.05
    df["is_full"] = (df["AvailableSpots"] <= (df["Capacity"] * FULL_THRESHOLD)).astype(int)
    
    # IMPROVEMENT: Calculate percentage available for fairer regression
    df["PercentAvailable"] = df["AvailableSpots"] / df["Capacity"]

    df["hour"] = df["LastUpdated"].dt.hour
    df["minute"] = df["LastUpdated"].dt.minute
    df["day_of_week"] = df["LastUpdated"].dt.dayofweek
    df["month"] = df["LastUpdated"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


def run_clustering(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, KMeans, StandardScaler, float, pd.DataFrame]:
    cluster_df = df[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(x_scaled)

    clustered_df = df.copy()
    clustered_df["cluster"] = clusters

    score = silhouette_score(x_scaled, clusters)
    cluster_summary = clustered_df.groupby("cluster")[
        ["Occupancy", "AvailableSpots", "hour", "day_of_week", "is_weekend"]
    ].mean()

    print(f"\nClustering completed with {n_clusters} clusters")
    print(f"Silhouette Score: {score:.3f}")
    print("\n--- Cluster Summary ---")
    print(cluster_summary)

    return clustered_df, kmeans, scaler, score, cluster_summary


def build_preprocessor() -> ColumnTransformer:
    categorical_features = ["SystemCodeNumber"]
    numeric_features = [
        "Capacity", "hour", "minute", "day_of_week", "month",
        "is_weekend", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        "cluster",
    ]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

# --- 2. CLASSIFICATION PIPELINE ---

def build_classification_pipeline() -> ImbPipeline:
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    return ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", clf),
    ])


def train_and_evaluate_classification(df: pd.DataFrame, clf_pipeline: ImbPipeline):
    X = df[FEATURE_COLS]
    y_clf = df["is_full"]

    ordered_idx = df.sort_values("LastUpdated").index
    split = int(len(ordered_idx) * 0.8)
    train_idx, test_idx = ordered_idx[:split], ordered_idx[split:]

    print("Training classification model (is_full at <= 5% capacity)...")
    clf_pipeline.fit(X.loc[train_idx], y_clf.loc[train_idx])

    pred_full = clf_pipeline.predict(X.loc[test_idx])
    pred_prob_full = clf_pipeline.predict_proba(X.loc[test_idx])[:, 1]

    metrics = {"accuracy": accuracy_score(y_clf.loc[test_idx], pred_full)}

    preview = X.loc[test_idx].copy()
    preview["actual_is_full"] = y_clf.loc[test_idx]
    preview["predicted_is_full"] = pred_full
    preview["predicted_prob_full"] = pred_prob_full

    return clf_pipeline, metrics, preview


def save_classification_plots(preview: pd.DataFrame, metrics: dict[str, float], plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Classification Report (is_full) ---")
    print(classification_report(preview["actual_is_full"], preview["predicted_is_full"]))

    # Confusion Matrix
    cm = confusion_matrix(preview["actual_is_full"], preview["predicted_is_full"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Classification: Confusion Matrix (Full vs Not Full)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    
    # IMPROVEMENT: Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        preview["actual_is_full"], 
        preview["predicted_prob_full"], 
        name="Random Forest"
    )
    plt.title("Precision-Recall Curve (Focusing on 'Full' Lots)")
    plt.savefig(plots_dir / "precision_recall_curve.png", dpi=160)
    plt.close()


def save_text_report(report_path: Path, title: str, sections: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = [title, "=" * len(title), ""]
    content.extend(sections)
    report_path.write_text("\n".join(content), encoding="utf-8")

# --- 3. REGRESSION PIPELINE ---

def build_regression_pipeline() -> SkPipeline:
    preprocessor = build_preprocessor()
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)
    return SkPipeline([("pre", preprocessor), ("model", model)])


def train_and_evaluate_regression(df: pd.DataFrame, model: SkPipeline):
    X = df[FEATURE_COLS]
    
    # IMPROVEMENT: Train on Percentage, not raw spots
    y = df["PercentAvailable"] 

    ordered_idx = df.sort_values("LastUpdated").index
    split = int(len(ordered_idx) * 0.8)
    train_idx, test_idx = ordered_idx[:split], ordered_idx[split:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print("Training regression model (Percentage Based)...")
    model.fit(X_train, y_train)
    preds_percent = model.predict(X_test)

    # Convert percentages back to actual spots for evaluation metrics
    actual_spots = y_test * X_test["Capacity"]
    predicted_spots = preds_percent * X_test["Capacity"]

    metrics = {
        "mae": mean_absolute_error(actual_spots, predicted_spots),
        "rmse": float(np.sqrt(mean_squared_error(actual_spots, predicted_spots))),
        "r2": r2_score(actual_spots, predicted_spots),
    }

    preview = X_test.copy()
    preview["actual_available"] = actual_spots
    preview["predicted_available"] = predicted_spots

    return model, metrics, preview


def save_regression_plots(preview: pd.DataFrame, metrics: dict[str, float], plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    actual = preview["actual_available"].to_numpy()
    predicted = preview["predicted_available"].to_numpy()
    preview["absolute_error"] = np.abs(actual - predicted)

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predicted, alpha=0.35, edgecolors="none")
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    ax.set_title("Regression: Actual vs Predicted Available Spots")
    ax.set_xlabel("Actual available spots")
    ax.set_ylabel("Predicted available spots")
    ax.text(
        0.05, 0.95, f"MAE: {metrics['mae']:.2f}\nRMSE: {metrics['rmse']:.2f}\nR2: {metrics['r2']:.3f}",
        transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "regression_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # IMPROVEMENT: Average Error by Hour
    error_by_hour = preview.groupby("hour")["absolute_error"].mean()
    plt.figure(figsize=(10, 5))
    error_by_hour.plot(kind="bar", color="#4C78A8")
    plt.title("Regression: Average Prediction Error by Hour of Day")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Mean Absolute Error (Spots)")
    plt.xticks(rotation=0)
    plt.savefig(plots_dir / "regression_error_by_hour.png", dpi=160)
    plt.close()

    # IMPROVEMENT: Error vs Capacity (Normalized)
    preview["percentage_error"] = (preview["absolute_error"] / preview["Capacity"]) * 100
    plt.figure(figsize=(8, 6))
    plt.scatter(preview["Capacity"], preview["percentage_error"], alpha=0.3, color="darkorange")
    plt.title("Regression: Error Percentage vs. Lot Capacity")
    plt.xlabel("Lot Capacity")
    plt.ylabel("Error (% of Capacity)")
    plt.axhline(10, color="red", linestyle="--", label="10% Error Threshold")
    plt.legend()
    plt.savefig(plots_dir / "regression_error_vs_capacity.png", dpi=160)
    plt.close()


def save_clustering_plots(clustered_df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    cluster_df = clustered_df[CLUSTER_FEATURES].copy()
    x_scaled = scaler.transform(cluster_df)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clustered_df["cluster"], alpha=0.5, cmap="tab10")
    plt.title("Parking Pattern Clusters (PCA View)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(plots_dir / "cluster_pca.png", dpi=160)
    plt.close()

    cluster_counts = clustered_df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    cluster_counts.plot(kind="bar", color="#5C89B8")
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_size_distribution.png", dpi=160)
    plt.close()


def save_clustering_report(
    clustered_df: pd.DataFrame,
    score: float,
    cluster_summary: pd.DataFrame,
    plots_dir: Path,
) -> None:
    report_sections = [
        f"Clustering completed with {clustered_df['cluster'].nunique()} clusters",
        f"Silhouette Score: {score:.3f}",
        "",
        "--- Cluster Summary ---",
        cluster_summary.to_string(),
        "",
        f"Cluster counts: {clustered_df['cluster'].value_counts().sort_index().to_dict()}",
    ]
    save_text_report(plots_dir / "clustering_report.txt", "Clustering Report", report_sections)

# --- 4. FEATURE IMPORTANCE (Shared) ---

def plot_feature_importance(pipeline, title: str, filepath: Path):
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_
    
    preprocessor = pipeline.named_steps["pre"]
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["SystemCodeNumber"])
    num_features = ["Capacity", "hour", "minute", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "minute_sin", "minute_cos", "cluster"]
    all_features = list(cat_features) + num_features
    
    indices = np.argsort(importances)[-15:] # Plot top 15
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align="center", color="#5C89B8")
    plt.yticks(range(len(indices)), [all_features[i] for i in indices])
    plt.title(title)
    plt.xlabel("Relative Feature Importance")
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()

# --- 5. EXECUTION ROUTINES ---

def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\n--- Dataset Summary ---")
    print(f"Total rows (records): {len(df)}")
    print(f"Total unique parking sites: {df['SystemCodeNumber'].nunique()}")
    print(f"Lots defined as 'Full' (<= 5% capacity): {df['is_full'].sum()}")


def build_classification_report_text(df: pd.DataFrame, metrics: dict[str, float], preview: pd.DataFrame) -> str:
    return "\n".join(
        [
            "Classification Report",
            "====================",
            "",
            f"Dataset rows: {len(df)}",
            f"Unique sites: {df['SystemCodeNumber'].nunique()}",
            f"Full lots (<= 5% capacity): {df['is_full'].sum()}",
            "",
            f"Accuracy: {metrics['accuracy']:.2%}",
            "",
            classification_report(preview["actual_is_full"], preview["predicted_is_full"]),
            "",
            "Sample predictions:",
            preview[
                ["SystemCodeNumber", "Capacity", "actual_is_full", "predicted_is_full", "predicted_prob_full"]
            ].head(10).to_string(index=False),
        ]
    )


def build_regression_report_text(df: pd.DataFrame, metrics: dict[str, float], preview: pd.DataFrame) -> str:
    return "\n".join(
        [
            "Regression Report",
            "=================",
            "",
            f"Dataset rows: {len(df)}",
            f"Unique sites: {df['SystemCodeNumber'].nunique()}",
            f"Full lots (<= 5% capacity): {df['is_full'].sum()}",
            "",
            f"MAE: {metrics['mae']:.3f}",
            f"RMSE: {metrics['rmse']:.3f}",
            f"R2: {metrics['r2']:.3f}",
            "",
            "Sample predictions:",
            preview[
                ["SystemCodeNumber", "Capacity", "hour", "minute", "day_of_week", "actual_available", "predicted_available"]
            ].head(10).to_string(index=False),
        ]
    )


def build_clustering_report_text(clustered_df: pd.DataFrame, score: float, cluster_summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "Clustering Report",
            "=================",
            "",
            f"Dataset rows: {len(clustered_df)}",
            f"Unique sites: {clustered_df['SystemCodeNumber'].nunique()}",
            f"Cluster count: {clustered_df['cluster'].nunique()}",
            f"Silhouette Score: {score:.3f}",
            "",
            "--- Cluster Summary ---",
            cluster_summary.to_string(),
            "",
            "Cluster counts:",
            clustered_df["cluster"].value_counts().sort_index().to_string(),
        ]
    )

def run_classification(args: argparse.Namespace) -> None:
    df = load_and_prepare_data(args.data)
    clustered_df, _, _, _, _ = run_clustering(df, n_clusters=4)
    df = clustered_df
    print_dataset_summary(df)

    clf_pipeline = build_classification_pipeline()
    clf_pipeline, metrics, preview = train_and_evaluate_classification(df, clf_pipeline)

    print("\nClassification model performance:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")

    joblib.dump(clf_pipeline, args.output_model_clf)
    save_classification_plots(preview, metrics, args.plots_dir)
    plot_feature_importance(clf_pipeline, "Classification: Top Features", args.plots_dir / "feature_importance.png")
    save_text_report(args.plots_dir / "classification_report.txt", "Classification Report", build_classification_report_text(df, metrics, preview).splitlines())
    
    print(f"\nSaved models and plots to: {args.plots_dir}")

def run_regression(args: argparse.Namespace) -> None:
    df = load_and_prepare_data(args.data)
    clustered_df, _, _, _, _ = run_clustering(df, n_clusters=4)
    df = clustered_df
    print_dataset_summary(df)

    model = build_regression_pipeline()
    model, metrics, preview = train_and_evaluate_regression(df, model)

    print("\nRegression model performance:")
    print(f"  MAE: {metrics['mae']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  R2: {metrics['r2']:.3f}")

    joblib.dump(model, args.output_model)
    save_regression_plots(preview, metrics, args.plots_dir)
    plot_feature_importance(model, "Regression: Top Features", args.plots_dir / "feature_importance.png")
    save_text_report(args.plots_dir / "regression_report.txt", "Regression Report", build_regression_report_text(df, metrics, preview).splitlines())
    
    print(f"\nSaved models and plots to: {args.plots_dir}")


def run_clustering_only(args: argparse.Namespace) -> None:
    df = load_and_prepare_data(args.data)
    clustered_df, kmeans, scaler, score, cluster_summary = run_clustering(df, n_clusters=4)

    print_dataset_summary(clustered_df)

    save_clustering_plots(clustered_df, kmeans, scaler, args.plots_dir)
    save_text_report(
        args.plots_dir / "clustering_report.txt",
        "Clustering Report",
        build_clustering_report_text(clustered_df, score, cluster_summary).splitlines(),
    )

    joblib.dump(kmeans, args.output_model_cluster)
    print(f"\nSaved clustering model and plots to: {args.plots_dir}")

# --- 6. CLI SETUP ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main script for parking classification and regression")
    subparsers = parser.add_subparsers(dest="mode")

    clf_parser = subparsers.add_parser("classification", help="Classification only (full vs not full)")
    clf_parser.add_argument("--data", type=Path, default=Path("dataset.csv"))
    clf_parser.add_argument("--plots-dir", type=Path, default=Path("plots/classification-plots"))
    clf_parser.add_argument("--output-model-clf", type=Path, default=Path("parking_clf_model.joblib"))

    reg_parser = subparsers.add_parser("regression", help="Single-model regression analysis")
    reg_parser.add_argument("--data", type=Path, default=Path("dataset.csv"))
    reg_parser.add_argument("--plots-dir", type=Path, default=Path("plots/regression/plots"))
    reg_parser.add_argument("--output-model", type=Path, default=Path("parking_reg_model.joblib"))

    cluster_parser = subparsers.add_parser("clustering", help="Clustering only")
    cluster_parser.add_argument("--data", type=Path, default=Path("dataset.csv"))
    cluster_parser.add_argument("--plots-dir", type=Path, default=Path("plots/clustering-plots"))
    cluster_parser.add_argument("--output-model-cluster", type=Path, default=Path("parking_cluster_model.joblib"))

    both_parser = subparsers.add_parser("both", help="Run classification and regression one after another")
    both_parser.add_argument("--data", type=Path, default=Path("dataset.csv"))
    both_parser.add_argument("--classification-plots-dir", type=Path, default=Path("plots/classification-plots"))
    both_parser.add_argument("--regression-plots-dir", type=Path, default=Path("plots/regression/plots"))
    both_parser.add_argument("--output-model-clf", type=Path, default=Path("parking_clf_model.joblib"))
    both_parser.add_argument("--output-model-reg", type=Path, default=Path("parking_reg_model.joblib"))

    all_parser = subparsers.add_parser("all", help="Run clustering, classification, and regression")
    all_parser.add_argument("--data", type=Path, default=Path("dataset.csv"))
    all_parser.add_argument("--clustering-plots-dir", type=Path, default=Path("plots/clustering-plots"))
    all_parser.add_argument("--classification-plots-dir", type=Path, default=Path("plots/classification-plots"))
    all_parser.add_argument("--regression-plots-dir", type=Path, default=Path("plots/regression-plots"))
    all_parser.add_argument("--output-model-cluster", type=Path, default=Path("parking_cluster_model.joblib"))
    all_parser.add_argument("--output-model-clf", type=Path, default=Path("parking_clf_model.joblib"))
    all_parser.add_argument("--output-model-reg", type=Path, default=Path("parking_reg_model.joblib"))

    if len(sys.argv) == 1:
        return parser.parse_args(["all"])

    if sys.argv[1] in {"classification", "regression", "clustering", "both", "all", "-h", "--help"}:
        return parser.parse_args()

    return parser.parse_args(["all", *sys.argv[1:]])

def main() -> None:
    args = parse_args()
    if args.mode == "classification":
        run_classification(args)
    elif args.mode == "regression":
        run_regression(args)
    elif args.mode == "clustering":
        run_clustering_only(args)
    elif args.mode == "both":
        clf_args = argparse.Namespace(data=args.data, plots_dir=args.classification_plots_dir, output_model_clf=args.output_model_clf)
        reg_args = argparse.Namespace(data=args.data, plots_dir=args.regression_plots_dir, output_model=args.output_model_reg)
        run_classification(clf_args)
        print("\n" + "=" * 60 + "\n")
        run_regression(reg_args)
    elif args.mode == "all":
        cluster_args = argparse.Namespace(data=args.data, plots_dir=args.clustering_plots_dir, output_model_cluster=args.output_model_cluster)
        clf_args = argparse.Namespace(data=args.data, plots_dir=args.classification_plots_dir, output_model_clf=args.output_model_clf)
        reg_args = argparse.Namespace(data=args.data, plots_dir=args.regression_plots_dir, output_model=args.output_model_reg)
        run_clustering_only(cluster_args)
        print("\n" + "=" * 60 + "\n")
        run_classification(clf_args)
        print("\n" + "=" * 60 + "\n")
        run_regression(reg_args)

if __name__ == "__main__":
    main()