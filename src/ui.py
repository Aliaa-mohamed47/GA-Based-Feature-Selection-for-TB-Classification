import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.ga_core import run_ga
from src.pso import run_pso
from src.data_loader import load_data, preprocess

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="GA vs PSO Feature Selection", layout="wide")
st.title("🧬 GA vs PSO Feature Selection Dashboard")

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_resource
def get_data():
    X, y, names = load_data()
    return preprocess(X, y) + (names,)

X_train, X_test, y_train, y_test, feature_names = get_data()
num_features_total = X_train.shape[1]

# =====================================================
# SIDEBAR SETTINGS
# =====================================================
st.sidebar.header("⚙️ Algorithm Settings")
algorithm = st.sidebar.selectbox("Choose Algorithm", ["GA", "PSO"])

if algorithm == "GA":
    st.sidebar.subheader("🧬 GA Parameters")
    selection_method = st.sidebar.selectbox("Selection Method", ["tournament", "roulette", "rank"])
    crossover_method = st.sidebar.selectbox("Crossover Method", ["single_point", "two_point", "uniform"])
    mutation_method = st.sidebar.selectbox("Mutation Method", ["bit_flip", "swap", "inversion"])
    pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
    gens = st.sidebar.slider("Generations", 10, 200, 50)
else:
    st.sidebar.subheader("🐝 PSO Parameters")
    inertia = st.sidebar.slider("Inertia (w)", 0.1, 1.0, 0.7)
    c1 = st.sidebar.slider("Cognitive (c1)", 0.1, 2.5, 1.5)
    c2 = st.sidebar.slider("Social (c2)", 0.1, 2.5, 1.5)

# =====================================================
# RUN EXPERIMENT
# =====================================================
if st.button("▶️ Run Experiment"):
    with st.spinner(f"Running {algorithm}..."):
        if algorithm == "GA":
            result = run_ga(
                selection_method=selection_method,
                crossover_method=crossover_method,
                mutation_method=mutation_method,
                verbose=False
            )
        else:
            result = run_pso(X_train, X_test, y_train, y_test, verbose=False)

    # Display Metrics
    st.success(f"{algorithm} Completed")
    c1, c2, c3 = st.columns(3)
    
    best_acc = result["best_accuracy"]
    num_feat = result["num_features"]
    reduction = ((num_features_total - num_feat) / num_features_total) * 100

    c1.metric("Accuracy", f"{best_acc:.4f}")
    c2.metric("Selected Features", f"{num_feat}")
    c3.metric("Reduction %", f"{reduction:.2f}%")

    # Convergence Plot
    st.subheader("📈 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(result["history_best"], label="Best Fitness", color='blue')
    ax.set_xlabel("Generation/Iteration")
    ax.set_ylabel("Fitness Score")
    st.pyplot(fig)

# =====================================================
# REPORT & STABILITY ANALYSIS (REAL DATA)
# =====================================================
st.write("---")
st.header("📊 Performance & Stability Report")

# Use your specific path
csv_path = Path("results/summary_report.csv")

if csv_path.exists():
    # Load and Clean CSV
    raw_df = pd.read_csv(csv_path, header=None)
    raw_df = raw_df.dropna(how="all")
    
    # Locate header row
    header_idx = raw_df[raw_df.iloc[:, 0].str.contains("algorithm", case=False, na=False)].index
    if not header_idx.empty:
        header_row = header_idx[0]
        df = raw_df.iloc[header_row+1:].copy()
        df.columns = ["algorithm", "config", "accuracy", "std", "reduction", "runtime"]
        
        # Convert numeric
        for col in ["accuracy", "std", "reduction", "runtime"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['accuracy'])

        # --- REAL BOX PLOT ---
        st.subheader("📦 Algorithm Stability (Accuracy Spread)")
        fig_box, ax_box = plt.subplots(figsize=(10, 5))
        
        # This creates boxes based on the actual recorded accuracies in your report
        df.boxplot(column='accuracy', by='algorithm', ax=ax_box, patch_artist=True)
        plt.suptitle("") # Clear automatic title
        ax_box.set_title("Distribution of Accuracy Across All Experiments")
        ax_box.set_ylabel("Accuracy")
        st.pyplot(fig_box)

        # --- SUMMARY TABLE ---
        st.subheader("📋 Detailed Comparison Table")
        st.dataframe(df, use_container_width=True)
        
        # Download
        st.download_button("📥 Download Clean Report", df.to_csv(index=False), "report.csv", "text/csv")
    else:
        st.error("Could not find the 'algorithm' header in the CSV.")
else:
    st.warning("Summary report not found at the specified path.")





## =====================================================
# COMPARISON SECTION (GA vs PSO)
# =====================================================

st.write("---")
st.header("⚔️ GA vs PSO Comparison")

csv_path = Path("results/summary_report.csv")

if not csv_path.exists():
    st.warning("Run GA or PSO first to generate comparison data")

else:
    # ================= LOAD CSV SAFELY =================
    df = pd.read_csv(csv_path, header=None)

    # remove empty rows
    df = df.dropna(how="all")

    # find real header row (where "algorithm" exists)
    header_row = df[df.iloc[:, 0] == "algorithm"].index[0]

    # set header
    df.columns = df.iloc[header_row]

    # keep only actual data
    df = df[(header_row + 1):].reset_index(drop=True)

    # rename columns clearly
    df.columns = [
        "algorithm",
        "config_label",
        "accuracy",
        "accuracy_std",
        "reduction_pct",
        "runtime_sec"
    ]

    # convert numeric columns
    for col in ["accuracy", "accuracy_std", "reduction_pct", "runtime_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # ================= SUMMARY =================
    summary = df.groupby("algorithm").agg({
        "accuracy": "mean",
        "reduction_pct": "mean",
        "runtime_sec": "mean"
    }).reset_index()

    st.dataframe(summary, use_container_width=True)

    # ================= ACCURACY =================
    st.subheader("📊 Accuracy Comparison")

    fig1, ax1 = plt.subplots()
    ax1.bar(summary["algorithm"], summary["accuracy"])
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Mean Accuracy")
    st.pyplot(fig1)

    # ================= REDUCTION =================
    st.subheader("📉 Feature Reduction Comparison")

    fig2, ax2 = plt.subplots()
    ax2.bar(summary["algorithm"], summary["reduction_pct"])
    ax2.set_ylabel("Reduction %")
    ax2.set_title("Feature Reduction")
    st.pyplot(fig2)

    # ================= RUNTIME =================
    st.subheader("⏱ Runtime Comparison")

    fig3, ax3 = plt.subplots()
    ax3.bar(summary["algorithm"], summary["runtime_sec"])
    ax3.set_ylabel("Seconds")
    ax3.set_title("Runtime")
    st.pyplot(fig3)
# =====================================================
# EDA ANALYSIS (IMAGE DISPLAY)
# =====================================================

st.write("---")
st.header("🔬 EDA Analysis")

eda_image_path = Path("results/plots/eda_correlation.png")

if eda_image_path.exists():
    st.image(str(eda_image_path), caption="EDA Correlation Heatmap", width=700)
else:
    st.warning("EDA image not found. Please check the file path.")