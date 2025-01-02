# Online Payment Fraud Detection Using Machine Learning  

This project implements a machine learning pipeline to detect fraudulent online payment transactions. The dataset is analyzed, preprocessed, and modeled to ensure effective fraud detection using various algorithms. This repository contains the complete code and explanations for each step of the process, from data exploration to model evaluation.  

## Features  
- **Exploratory Data Analysis (EDA):**  
  - Performed data visualization to understand distributions, correlations, and patterns.  
  - Visualized key features such as transaction types, amounts, and step timings using bar plots, count plots, and correlation heatmaps.  

- **Data Preprocessing:**  
  - Addressed class imbalance by oversampling the minority class (fraudulent transactions) to match the majority class.  
  - Encoded categorical variables using Label Encoding for compatibility with machine learning models.  
  - Split the data into training and testing sets to evaluate model performance.

- **Machine Learning Models:**  
  - Trained and evaluated multiple models, including Logistic Regression, Random Forest, and XGBoost.  
  - Used ROC-AUC scores, accuracy, and confusion matrices to compare model performance.  

## Workflow  
1. **Data Exploration and Visualization:**  
   - Loaded and inspected the dataset (`onlinefraud.csv`).  
   - Identified categorical, integer, and float variables for preprocessing.  
   - Plotted transaction distributions and correlations to highlight fraudulent transaction patterns.  

2. **Data Preprocessing:**  
   - Handled class imbalance using random oversampling for fraudulent transactions.  
   - Encoded categorical variables with `LabelEncoder`.  
   - Created a balanced dataset and shuffled the data for unbiased training.  

3. **Model Training and Evaluation:**  
   - Implemented Logistic Regression, Random Forest, and XGBoost classifiers.  
   - Evaluated models using metrics like ROC-AUC, accuracy, and confusion matrices.  
   - Visualized model performance through confusion matrices and comparison of training vs. testing metrics.  

## Dependencies  
The following libraries are used in this project:  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `xgboost`  

Install the dependencies using:  
```bash  
pip install numpy pandas matplotlib seaborn scikit-learn xgboost  
```  

## How to Run the Code  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/online-payment-fraud-detection.git  
   ```  
2. Navigate to the project directory:  
   ```bash  
   cd online-payment-fraud-detection  
   ```  
3. Run the Jupyter Notebook or Python script containing the code. Ensure the dataset (`onlinefraud.csv`) is in the same directory.  

## Results  
- **Best Model:** Random Forest achieved the highest accuracy and ROC-AUC score in detecting fraudulent transactions.  
- **Performance Metrics:**  
  - Training Accuracy: ~100%  
  - Testing Accuracy: ~100%  
  - ROC-AUC Score: ~1.00  

## Visualizations  
- **Class Distributions Before and After Oversampling:** Showcases the improvement in class balance.  
- **Feature Correlations:** Highlights relationships between numerical features and target variables.  
- **Confusion Matrices:** Demonstrates model effectiveness in distinguishing fraud from non-fraud cases.  

## Dataset  
The dataset (`onlinefraud.csv`) contains transaction details, including `type`, `amount`, `isFraud`, and other relevant features.  

## Acknowledgments    
- Dataset source (Kaggle) - https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection.  

