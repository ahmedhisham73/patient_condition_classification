# patient_condition_classification
# Drug Sentiment Classification
# Table of Contents
# Overview
# Project Details
# Installation
# Usage
# Contributing
# Overview
This repository contains the code for a drug sentiment classification project. The goal of the project is to predict drug conditions based on the review. We trained a multi-class classification model using an ensemble of machine learning models and used Flask for model deployment.

# Project Details
The project involved several steps:

# Data preprocessing, including text normalization, lemmatization, and vectorization.
Training multiple machine learning models, including Logistic Regression, Multinomial Naive Bayes, and Decision Tree Classifier.
Creating an ensemble model using voting for better prediction.
Deploying the trained model using Flask.
Installation
Before running the project, make sure to install the necessary dependencies. You can do this by running the following command:


pip install -r requirements.txt
# Usage
To start the Flask application, run the following command:

bash

python app.py
Then, open a web browser and navigate to localhost:5000. You can use the web interface to input a drug review and get a prediction for the drug condition.

# Contributing
Contributions to this project are welcome. Please open an issue to discuss your proposed changes, or submit a pull request directly.

