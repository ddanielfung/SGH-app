# SGH Application: Diabetes Prediction using Machine Learning Models

This repository contains the code for developing and optimizing three machine learning models - logistic regression, decision tree, and random forest - to predict the likelihood of diabetes based on glycohemoglobin levels (gh) using Python in a Jupyter notebook.

## Dataset
The dataset used for this project is the NHANES (National Health and Nutrition Examination Survey) glycohemoglobin data. You can download the dataset from [this link](https://hbiostat.org/data) by scrolling down to NHANES glycohemoglobin data and downloading the `nhgh.tsv` file.

## Data Dictionary
The data dictionary for the dataset can be found [here](https://hbiostat.org/data/repo/nhgh). It provides detailed information about the variables present in the dataset.

## Instructions
1. Clone this repository to your local machine.
2. Download the dataset from the provided link and place it in the repository directory.
3. Ensure you have Python and Jupyter notebook installed on your machine.
4. Open the Jupyter notebook `Classification.ipynb` to view the code and execute the cells for the modelling.
5. Open the Jupyter notebook `SQLite3.ipynb` to view the code and execute the cells for the database normalisation and interface.
6. Open the Jupyter notebook `Visualisation.ipynb` to view the code and execute the cells for the visualisation.


## Dependencies
Make sure you have the following Python libraries installed:
- Pandas
- NumPy
- Scikit-learn

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Approach
1. Data Preprocessing: Loading the dataset, handling missing values, encoding categorical variables, and splitting the data into training and testing sets.
2. Model Development: Implementing logistic regression, decision tree, and random forest models using scikit-learn.
3. Model Optimization: Fine-tuning hyperparameters for a model to improve performance.
4. Model Comparison: Brief comparison of the performance of the fine-tuned models and interpreting the results.

