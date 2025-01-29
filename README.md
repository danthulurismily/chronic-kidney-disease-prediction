# chronic-kidney-disease-prediction
# Chronic Kidney Disease Prediction using KNN and Decision Tree

This project aims to predict chronic kidney disease (CKD) using two machine learning algorithms: K-Nearest Neighbors (KNN) and Decision Tree. The project uses a dataset from the UCI Machine Learning Repository, which contains various clinical features related to kidney function.  The notebook performs data preprocessing, exploratory data analysis (EDA), model training, and evaluation to compare the performance of both algorithms.

## Project Overview

Chronic Kidney Disease (CKD) is a serious health condition where the kidneys progressively lose their ability to filter waste and excess fluids from the blood. Early detection and treatment can significantly slow the progression of the disease and improve patient outcomes. This project leverages machine learning to assist in the early prediction of CKD based on clinical data.

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository and is named "chronic_kidney_disease.csv". It contains 400 instances and 25 attributes, including both numerical and categorical features. Some key attributes include:

* *age:* Age of the patient
* *bp:* Blood pressure
* *sg:* Specific gravity of urine
* *al:* Albumin levels
* *su:* Sugar levels
* *rbc:* Red blood cell count
  *...* (Other clinical features)
* *class:*  The target variable, indicating the presence or absence of CKD (ckd/notckd)


## Methodology

The project follows these key steps:

1. *Data Preprocessing:*
   - Handling missing values (represented as '?') using KNN imputation.
   - Scaling numerical features using StandardScaler.
   - Encoding the target variable ('class') using LabelEncoder.

2. *Exploratory Data Analysis (EDA):*
   - Visualizing the distribution of the target variable using a countplot to understand class imbalance.

3. *Model Training and Evaluation:*
   - Splitting the data into training and testing sets (50/50 split).
   - Training a Decision Tree Classifier and a KNN Classifier.
   - Using GridSearchCV to find the best hyperparameters for each model based on accuracy.
   - Evaluating the performance of both models using metrics such as precision, accuracy, recall, F1-score, and a confusion matrix.


## Results

The project demonstrates that the KNN classifier, with appropriate hyperparameter tuning, achieves slightly better performance than the Decision Tree classifier on this specific dataset and with the chosen evaluation metric (accuracy).  However, both models achieve high accuracy (around 97%), indicating their potential for CKD prediction. The specific results, including the best hyperparameters and evaluation metrics, are presented in the notebook's output.



## Getting Started

To run this project, you will need:

1. *Python:* Install Python 3.7 or higher.
2. *Libraries:* Install the required libraries using pip install -r requirements.txt.  The requirements.txt file is included in the repository and lists the following packages:
pandas
numpy
scikit-learn
seaborn
matplotlib
3. *Dataset:* Download the "chronic_kidney_disease.csv" dataset from the UCI Machine Learning Repository and place it in the same directory as the Jupyter notebook.

4. *Jupyter Notebook:* Open the "Kidney_Disease_Prediction.ipynb" notebook and run the cells.




## Further Improvements

* *Feature Engineering:* Explore creating new features from the existing ones to potentially improve model performance.
* *Class Imbalance:* Address the slight class imbalance in the dataset using techniques like SMOTE or oversampling.
* *Model Comparison:* Compare the performance with other classification algorithms like SVM, Logistic Regression, Random Forest, etc.
* *Deployment:* Develop a web application or other interface to deploy the trained model for practical use.
* *Explainability:* Use techniques like SHAP values or LIME to explain the model's predictions and gain insights into feature importance.


This enhanced README provides a comprehensive overview of the project, including background information, dataset details, methodology, results, instructions for running the code, and suggestions for further improvements.  It is now ready for GitHub!
