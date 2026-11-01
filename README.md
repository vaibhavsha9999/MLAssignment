# ML Assignment 2 - Classification Models and Streamlit App

## a. Problem statement
This project focuses on building and comparing multiple classification models on a public medical diagnosis dataset, then exposing the evaluation workflow through a Streamlit application. The objective is to train several models on the same dataset, compare their performance using standard classification metrics, and deploy an interactive frontend that loads saved models and evaluates uploaded test data without retraining.

## b. Dataset description
The selected dataset is the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository. It is a binary classification dataset with 569 instances and 30 numeric features. The target variable indicates whether a tumor is malignant or benign. The dataset is well suited for this assignment because it satisfies the minimum requirement of more than 500 instances and more than 12 features.

## c. GitHub Repository Link
https://github.com/vaibhavsha9999/MLAssignment

## d. Models used
The following five classification models were implemented on the same dataset:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Naive Bayes (Gaussian)
- Support Vector Machine

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.982456 | 0.995370 | 0.986111 | 0.986111 | 0.986111 | 0.962302 |
| Decision Tree | 0.912281 | 0.915675 | 0.955882 | 0.902778 | 0.928571 | 0.817412 |
| KNN | 0.956140 | 0.978836 | 0.958904 | 0.972222 | 0.965517 | 0.905447 |
| Naive Bayes | 0.938596 | 0.987765 | 0.945205 | 0.958333 | 0.951724 | 0.867553 |
| Support Vector Machine | 0.982456 | 0.995040 | 0.986111 | 0.986111 | 0.986111 | 0.962302 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves excellent performance with very high accuracy, AUC, and balanced precision-recall. It is one of the strongest models for this dataset. |
| Decision Tree | Performs well but is less stable than the linear and kernel-based models, with a noticeably lower AUC and MCC. |
| KNN | Delivers strong results and remains competitive, though it falls slightly behind the top-performing models on accuracy and AUC. |
| Naive Bayes | Shows reliable classification ability with high AUC, but performs slightly below Logistic Regression and SVM on overall metrics. |
| Support Vector Machine | Matches Logistic Regression closely in accuracy and remains highly effective on this dataset, with strong generalization. |

### Overall, Winner for the dataset?
Overall, the winner for this dataset is Logistic Regression because it has the best AUC among the top-performing models and matches the best accuracy and F1 performance. Support Vector Machine is extremely close and is tied on several metrics, but Logistic Regression provides the best overall balance across the evaluation metrics.

## e. Deployment and Streamlit app
The application is implemented in app.py and loads saved trained models from the model directory. It allows users to upload a CSV file containing the required features and target column, then evaluates the selected model without retraining on the uploaded data.

Live Streamlit App Link: Deployment pending after community cloud setup.

## f. Repository structure
- app.py: Streamlit interface for model selection and evaluation
- requirements.txt: project dependencies
- README.md: project documentation
- artifacts/test_data.csv: held-out test set used for evaluation
- artifacts/train_data.csv: training split used for model training
- model/: saved trained models for all five classifiers
- src/: preprocessing and training scripts

## g. Execution notes
1. Install dependencies using requirements.txt.
2. Run the data preparation script to generate train and test splits.
3. Train all five models using the training script.
4. Evaluate models using the evaluation script.
5. Launch the Streamlit app and upload the test CSV to compare model performance.

## h. Final checklist
- GitHub repository contains the full source code
- requirements.txt is included
- README.md is present
- Test data is included
- All five trained models are saved in the model directory
- Streamlit app loads saved models and evaluates uploaded CSV data without retraining
