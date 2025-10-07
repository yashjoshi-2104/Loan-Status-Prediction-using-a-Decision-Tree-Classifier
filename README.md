# Loan-Status-Prediction-using-a-Decision-Tree-Classifier
This project focuses on building a machine learning model to predict the status of a loan. Using a dataset of loan information, a Decision Tree Classifier is trained, evaluated, and optimized to classify loans into one of three categories: 'Fully Paid', 'Charged Off', or 'Current'. 

## Dataset

Dimensions: The initial dataset contains 39,717 rows and 111 columns.

Description: The dataset includes a wide range of features for each loan, such as the loan amount, interest rate, borrower's employment information, payment history, and current status.

## Project Workflow
The project follows a standard machine learning pipeline:

Data Loading & Initial Exploration: The loan.csv dataset is loaded into a pandas DataFrame. The shape and initial rows are inspected to understand its structure.

Preprocessing & Feature Engineering:

Target Variable Identification: The loan_status column is identified as the target variable. It is factorized into numerical labels:

0: Charged Off

1: Current

2: Fully Paid

## Model Training & Evaluation:

The preprocessed data is split into training (80%) and testing (20%) sets using a stratified split to maintain the distribution of the target classes.

A baseline Decision Tree Classifier (with max_depth=6) is trained.

The model's performance is evaluated using accuracy, a classification report (precision, recall, F1-score), and a confusion matrix.

## Results

Model Metric	Baseline Model (max_depth=5)	Tuned Model

Accuracy	98.58%	99.48%

Best Parameters	-	{'criterion': 'entropy', 'min_samples_leaf': 5}

## Feature Importance
The top features identified by the baseline model are:

recoveries (Importance: 0.63)

out_prncp_inv (Importance: 0.20)

total_rec_prncp (Importance: 0.08)

funded_amnt (Importance: 0.07)

last_pymnt_amnt (Importance: 0.007)

## Decision Tree Visualization
A visualization of the tuned decision tree (trimmed to a depth of 3 for readability) was generated to understand the model's decision-making process. The root node splits on the recoveries feature, confirming its high predictive power.


