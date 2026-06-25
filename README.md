# 🧬 GA-Based Feature Selection for TB Classification

An optimization framework that implements and compares evolutionary algorithms to perform feature selection on a Tuberculosis (TB) chest X-ray image dataset, evaluated using a K-Nearest Neighbors (KNN) classifier.

## 📌 Project Overview
Instead of using all extracted image features, this project evolves optimal binary masks using **Genetic Algorithms (GA)** and **Particle Swarm Optimization (PSO)** to select the most discriminative subset of features—maximizing classification accuracy while minimizing dimensionality.

### 🧠 Core Methodology & Pipeline
1. **Feature Extraction:** OpenCV extracts a 36-dimensional handcrafted feature vector per image (8 HOG texture features, 24 HSV color histogram bins, 4 basic color statistics).
2. **Multi-Objective Optimization:** Evaluates feature subsets using a custom Fitness Function:
   $$\text{Fitness} = 0.95 \times \text{CV\_Accuracy} + 0.05 \times (1 - \text{Feature\_Ratio})$$
3. **Evaluation:** Guided by 3-Fold Cross-Validation using a KNN ($K=3$) classifier to prevent test-set leakage.
4. **Reproducibility:** Validated across 30 independent trials with fixed random seeds.

## 📊 Key Experimental Results

| Algorithm Configuration | Accuracy (Mean ± STD) | Feature Reduction Rate |
| :--- | :---: | :---: |
| **GA** (Tournament Selection + Bit-Flip) | **93.57% ± 1.96%** | **~73%** |
| **GA** (Roulette Wheel + Uniform) | 93.36% ± 0.48% | ~72% |
| **PSO** (Standard Binary BPSO) | 93.14% ± 0.96% | ~63% |

*The Tournament-based Genetic Algorithm outperformed the standard BPSO baseline, achieving the highest feature reduction rate while maintaining superior classification performance.*

## ⚙️ Tech Stack & Dependencies
* **Core Machine Learning:** `scikit-learn`, `numpy`, `pandas`
* **Computer Vision / Preprocessing:** `opencv-python`
* **Optimization Frameworks:** Pure Python implementation of GA (Roulette, Tournament, Rank selection; Multi-type Crossover/Mutation) & Binary PSO.
* **Visualization & Analytics:** `matplotlib`, `seaborn`, `streamlit` (Interactive Analytics Dashboard)

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/<your-username>/ai420-ga-feature-selection.git
cd ai420-ga-feature-selection
pip install -r requirements.txt
```

### Running Experiments
```bash
python experiments/run_paper5.py
```

### Launching the Dashboard
```bash
streamlit run src/ui.py
```
## 👥 Team Members
**Aliaa Mohamed** • **A'laa Kamal** • **Waad Sadek** • **Sama Said** • **Malak Hany** • **Malak Ragab**
