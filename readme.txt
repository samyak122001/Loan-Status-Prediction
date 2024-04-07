Loan Status Prediction

This repository contains a machine learning project that predicts the loan status of borrowers based on various features. The project aims to build a predictive model that can classify whether a loan will be approved or not.


Dataset

The project uses two datasets:

train_loan_data.csv: The training dataset used to train and validate the models.
test_loan_data.csv: The test dataset used to make predictions using the best trained model.

Project Structure

The repository has the following structure:


Copy code
├── loan_prediction.py
├── train_loan_data.csv
├── test_loan_data.csv
├── test_predictions.csv
└── README.md

loan_prediction.py: The main Python script that contains the code for data preprocessing, model training, evaluation, and prediction.
train_loan_data.csv: The training dataset in CSV format.
test_loan_data.csv: The test dataset in CSV format.
test_predictions.csv: The file containing the predicted loan statuses for the test dataset.

Dependencies
The project requires the following dependencies:

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn

You can install the dependencies using pip:


Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
To run the loan status prediction project, follow these steps:


Navigate to the project directory:


cd loan-status-prediction
Make sure you have the required dependencies installed.
Place the train_loan_data.csv and test_loan_data.csv files in the project directory.

Run the loan_prediction.py script:


python loan_prediction.py
The script will perform the following steps:
Load and preprocess the training and test datasets.
Perform exploratory data analysis (EDA) on the training data.
Train and evaluate different machine learning models.
Select the best-performing model based on the F1-score.
Perform hyperparameter tuning on the best model.
Make predictions on the test dataset using the best model.
Save the predictions to the test_predictions.csv file.
The predicted loan statuses for the test dataset will be saved in the test_predictions.csv file.

Results
The project trains and evaluates multiple machine learning models, including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The best-performing model is selected based on the weighted F1-score.

The best model's performance and hyperparameters are printed in the console output. The predictions for the test dataset are saved in the test_predictions.csv file.

Contributing

Contributions to the project are welcome. If you find any issues or have suggestions for improvement, please create an issue or submit a pull request.



