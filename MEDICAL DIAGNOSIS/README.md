# Medical Diagnosis Prediction Model

This is a beginner-friendly machine learning project that predicts diabetes diagnosis using the Pima Indians Diabetes Dataset. The model uses Random Forest Classifier, which is both powerful and easy to understand.

## Prerequisites

- Python 3.7 or higher
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Pima Indians Diabetes Dataset, which includes the following features:
- Number of pregnancies
- Glucose concentration
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age

The target variable indicates whether the patient has diabetes (1) or not (0).

## Running the Model

To run the prediction model, simply execute:
```bash
python diabetes_prediction.py
```

The script will:
1. Load and preprocess the data
2. Train the Random Forest model
3. Evaluate the model's performance
4. Generate a feature importance plot
5. Show an example prediction

## Output

The script provides:
- Basic dataset information
- Model accuracy and detailed classification report
- A feature importance plot (saved as 'feature_importance.png')
- An example prediction for a sample patient

## Understanding the Code

The code is heavily commented to help beginners understand:
- Data loading and preprocessing
- Feature scaling
- Model training
- Evaluation metrics
- Making predictions

## Model Details

The model uses Random Forest Classifier with:
- 100 decision trees
- Standardized features
- 80-20 train-test split
- Random state for reproducibility

## Note

This is a simplified model for educational purposes. In a real medical setting, more sophisticated models and extensive validation would be required. 