# Predicting Life Satisfaction Using Machine Learning and Explainable AI

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Highlights](#highlights)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Dataset](#dataset)
6. [Notebook Structure](#notebook-structure)
7. [Results and Insights](#results-and-insights)
8. [Explainable AI](#explainable-ai)
9. [Ablation Studies](#ablation-studies)
10. [Citation](#citation)
11. [Contact](#contact)
12. [License](#license)

---

## Introduction
This repository accompanies the research article *"Predicting Life Satisfaction Using Machine Learning and Explainable AI"*, published in **Heliyon**. The project demonstrates how advanced machine learning and explainable AI (XAI) techniques can predict life satisfaction with high accuracy. The dataset, sourced from the SHILD survey in Denmark, provides critical insights into factors affecting well-being. The study also explores the use of large language models (LLMs) for predicting life satisfaction, achieving significant results.

**Publication Link:** [Heliyon Article](https://www.sciencedirect.com/science/article/pii/S2405844024071895)

---

## Repository Structure
```
├── Figures/                   # Contains visualizations used in the notebooks
├── LICENSE                    # License information
├── README.md                  # Repository documentation
├── Predicting_Life_Satisfaction.ipynb  # Main Jupyter Notebook
```

---

## Highlights
- Achieved **93.8% accuracy** and **73% macro F1-score** for predicting life satisfaction.
- Used **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify 27 key determinants of life satisfaction.
- Employed **Explainable AI** techniques to ensure interpretability and transparency of predictions.
- Explored **Large Language Models (LLMs)** like BERT, BioBERT, and ClinicalBERT to predict life satisfaction using natural language sentences.
- Conducted **ablation studies** on data resampling and feature selection techniques to optimize model performance.

---

## Getting Started

### Prerequisites
- **Python 3.6+**: Download [here](https://www.python.org/downloads/)
- **Jupyter Notebook**: Install via pip:
  ```bash
  pip install notebook
  ```

### Installation
Clone the repository and navigate to the directory:
```bash
git clone https://github.com/alifelham/Predicting-Life-Satisfaction-Using-Machine-Learning.git
cd Predicting-Life-Satisfaction-Using-Machine-Learning
```

Install required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn missingno imbalanced-learn scikit-plot xgboost lightgbm
```

---

## Dataset
The dataset is sourced from the **SHILD (Survey of Health Impairment and Living Conditions in Denmark)**. It is publicly available under a **CC0 1.0 Universal Public Domain Dedication license**.

**Dataset Link:** [SHILD Dataset](https://doi.org/10.5061/dryad.qd2nj)

---

## Notebook Structure
1. **Data Importing and Preprocessing**: Handles missing values, categorical encoding, and outlier management.
2. **Exploratory Data Analysis**: Visualizations and data summaries.
3. **Model Building**: Implements ML models such as Random Forest, XGBoost, and LightGBM.
4. **Model Evaluation**: Uses metrics like accuracy, F1-score, precision, recall, and AUC-ROC.
5. **Results Visualization**: Displays model performance and insights.
6. **Explainable AI**: Interprets predictions using XAI techniques.
7. **Age Group Analysis**: Examines primary determinants across different age brackets.

---

## Results and Insights
### Key Performance Metrics:
| Model               | Accuracy (%) | F1-Score (%) | Precision (%) | Recall (%) |
|---------------------|--------------|--------------|---------------|------------|
| Random Forest       | 93.8         | 70.6         | 72.0          | 69.3       |
| Gradient Boosting   | 92.2         | 70.3         | 67.9          | 73.7       |
| XGBoost             | 93.0         | 68.5         | 68.7          | 68.2       |
| Ensemble (Best)     | **93.6**     | **73.0**     | 71.9          | 74.3       |

### Insights:
- Health condition is the most critical determinant across all age groups.
- Dual data resampling (SMOTE + undersampling) improves both accuracy and F1-score.
- RFECV-based feature selection outperforms PCA-based approaches.

---

## Explainable AI
Explainable AI was employed to ensure model transparency. The framework explains how each input feature contributes to the prediction, providing actionable insights for stakeholders like policymakers and healthcare professionals.

**Example Visualization:**
*Add visualizations here for model explanations or prediction thresholds.*

---

## Ablation Studies
### Data Resampling:
The dual strategy of oversampling and undersampling led to significant improvements in model performance, achieving a balanced precision-recall tradeoff.

### Feature Selection:
RFECV selected 27 key features, surpassing PCA in both accuracy and interpretability.

---

## Citation
If you use this repository, please cite the following:
```bibtex
@article{alifelham2024lifesatisfaction,
    title={Predicting life satisfaction using machine learning and explainable AI},
    author={Alif Elham Khan, Mohammad Junayed Hasan, Humayra Anjum, Nabeel Mohammed, Sifat Momen},
    journal={Heliyon},
    year={2024},
    doi={10.1016/j.heliyon.2024.e31158}
}
```

---

## Contact
For questions or collaboration, contact:
- **Alif Elham Khan**: [alif.khan1@northsouth.edu](mailto:alif.khan1@northsouth.edu)
- **Mohammad Junayed Hasan**: [mohammad.hasan5@northsouth.edu](mailto:mohammad.hasan5@northsouth.edu)
- **Humayra Anjum**: [humayra.anjum@northsouth.edu](mailto:humayra.anjum@northsouth.edu)

---

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
