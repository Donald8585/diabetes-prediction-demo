---
title: Diabetes Risk Prediction
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🏥 Diabetes Risk Prediction System

Interactive ML demo for diabetes risk assessment using patient health data. Trained on 700,000 patients achieving 68% accuracy and 72% ROC-AUC.

🔗 **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/Donald8585/diabetes-prediction)  
🔗 **Kaggle Notebook**: [Training Analysis](https://www.kaggle.com/code/sword4949/notebook33ee53cd5b)

## 🎯 Features

- **Real-time risk assessment** with XGBoost classifier
- **26 patient features** including demographics, lifestyle, and medical history
- **Interactive Gradio interface** with 4 sample patient profiles
- **Risk stratification** (Low/Moderate/High)
- **Feature importance analysis** showing family history dominates prediction

## 📊 Model Performance

### Dataset
- **Source**: Kaggle Playground Series S5E12
- **Size**: 700,000 patients
- **Features**: 26 total (15 numerical, 6 categorical, 5 binary)
- **Target**: Binary classification (diabetes yes/no)
- **Class Distribution**: 62% diabetic, 38% non-diabetic

### Model Comparison

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 64% | 0.62 |
| Random Forest | 66% | 0.69 |
| **XGBoost** | **68%** | **0.72** |

XGBoost was selected for superior handling of feature interactions despite weak individual correlations.

### Feature Importance

Top 5 predictive features:
1. **Family History of Diabetes** (82.0%)
2. Age (3.2%)
3. Physical Activity (3.0%)
4. Triglycerides (1.0%)
5. BMI (0.8%)

## 🎮 Demo Features

### Patient Profiles

**4 Pre-loaded Cases:**
- **Healthy 30-Year-Old** - Low risk baseline
- **High Risk - Family History** - Multiple risk factors including genetic predisposition
- **Active Senior** - Older age but healthy lifestyle
- **Sedentary Lifestyle** - Poor lifestyle choices without genetic factors

### Input Categories

**Demographics:**
- Age, gender, ethnicity
- Education level, income, employment status

**Lifestyle:**
- Alcohol consumption, physical activity
- Diet quality score, sleep hours, screen time
- Smoking status

**Medical Measurements:**
- BMI, waist-to-hip ratio
- Blood pressure (systolic/diastolic)
- Heart rate
- Cholesterol (total, HDL, LDL), triglycerides

**Medical History:**
- Family history of diabetes
- Hypertension history
- Cardiovascular history

## 🛠️ Tech Stack

- **ML Framework**: XGBoost, Scikit-learn
- **UI**: Gradio
- **Deployment**: HuggingFace Spaces (CPU)
- **Language**: Python 3.10+

## 📈 Training Details

**Data Preprocessing:**
- One-hot encoding for categorical variables
- No missing values (clean dataset)
- Feature scaling not required for tree-based models

**Model Configuration:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.2,
    min_child_weight=3,
    random_state=42
)
```

**Hyperparameter Tuning:**
- GridSearchCV with 3-fold cross-validation
- Optimized for ROC-AUC score
- 162 total combinations evaluated

## 🔬 Key Insights

1. **Family history dominates** - 82% feature importance confirms genetic component
2. **Weak individual correlations** (max 0.21) indicate multifactorial disease
3. **Age and lifestyle matter** - Physical inactivity is protective (-0.17 correlation)
4. **Class imbalance challenges** - Attempted SMOTE and class weights (didn't improve)

## 📝 Limitations

- Model trained on synthetic Kaggle competition data
- 68% accuracy means 32% error rate - not production-ready for clinical use
- Class imbalance (62% diabetic) affects minority class recall
- PCA features would improve performance but reduce interpretability

## 🚀 Future Improvements

- Try deep learning (neural networks) for better feature interactions
- Ensemble multiple models for improved robustness
- Add SHAP values for individual prediction explanations
- Collect real-world medical data for validation

## 👤 Author

**Alfred So (So Chit Wai)**
- GitHub: [@Donald8585](https://github.com/Donald8585)
- LinkedIn: [alfred-so](https://linkedin.com/in/alfred-so)
- Kaggle: [sword4949](https://kaggle.com/sword4949)

## 📝 License

MIT License

---

⭐ **Built with XGBoost, Gradio, and 700K patient records** ⭐
