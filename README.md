# TitanicSurvival
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset contains information about passengers, such as age, gender, class, and other characteristics. The goal is to build a model that accurately classifies whether a passenger survived or not.
Table of Contents

    Overview
    Dataset
    Data Preprocessing
    Modeling Techniques
    Evaluation Metrics
    Results
    Installation
    Usage
    License

Overview

The Titanic Survival Prediction project utilizes machine learning models to classify passengers into two categories:

    0: Did not survive
    1: Survived

The models are trained and evaluated using popular machine learning libraries. The predictions are based on features such as age, gender, passenger class, fare, and family size.
Dataset

The dataset used is the famous Titanic Dataset, available from Kaggle's Titanic Competition.
Key Columns in the Dataset:

    Survived: Target variable (0 = No, 1 = Yes)
    Pclass: Passenger class (1 = First, 2 = Second, 3 = Third)
    Sex: Gender of the passenger
    Age: Age of the passenger
    SibSp: Number of siblings/spouses aboard
    Parch: Number of parents/children aboard
    Fare: Ticket fare
    Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Data Preprocessing

To prepare the dataset for modeling:

    Handle Missing Values:
    Fill or drop rows with missing data for features like Age and Embarked.

    Feature Engineering:
        Create new features, such as family size or a cabin indicator.
        Encode categorical features (Sex, Embarked) into numerical values.

    Scaling and Normalization:
    Normalize numerical features like Age and Fare for better model performance.

    Train-Test Split:
    Split the dataset into training and testing sets (e.g., 80%-20%).

Modeling Techniques

The following machine learning models were implemented for this project:

    Logistic Regression
    Random Forest
    K-Nearest Neighbors (KNN)
    Support Vector Machine (SVM)
    Gradient Boosting (optional)

The models were evaluated on their ability to accurately classify survival outcomes.
Evaluation Metrics

The models were evaluated using the following metrics:

    Accuracy: Proportion of correctly predicted instances.
    Precision: Correct positive predictions as a proportion of total predicted positives.
    Recall: Correct positive predictions as a proportion of actual positives.
    F1-Score: Harmonic mean of precision and recall.

Results
Example Model Performances (Hypothetical Results):
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.80	0.78	0.75	0.76
Random Forest	0.85	0.82	0.80	0.81
KNN	0.78	0.75	0.73	0.74
SVM	0.82	0.80	0.78	0.79
Installation

To run the project, you need to install the required dependencies. Use the following command:

pip install -r requirements.txt

The requirements.txt should include:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn

Usage

    Load the Dataset:
    Place the Titanic dataset in the project directory or provide its path in the code.

    Run the Code:
    Execute the script to preprocess the data, train the models, and evaluate their performance.

    Predict Survival:
    Use the trained model to predict survival for new data points:

    new_passenger = [3, 'male', 22, 1, 0, 7.25, 'S']
    print(predict_survival(new_passenger))

    Visualize Results:
    Use the built-in plots to visualize the feature importance and performance metrics.

License

This project is licensed under the MIT License - see the LICENSE file for details.
