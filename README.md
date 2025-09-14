# Heart Attack Prediction using Machine Learning  

## Overview  
This project predicts the likelihood of a heart attack based on patient health data using multiple **classification algorithms**. The dataset contains patient records with features such as age, cholesterol, blood pressure, and other medical indicators. The target variable is **binary (0 = No Heart Attack, 1 = Heart Attack)**.  

## Dataset  
- File: `heartattack.csv`  
- Target Column: `target` (0 or 1)  
- Features: medical attributes such as age, sex, cholesterol, resting blood pressure, maximum heart rate, etc.  


## Requirements  
Install the following Python packages before running the notebook:  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Models Used  
The following classification models were trained and compared:  

1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Support Vector Classifier (SVC)  
4. Decision Tree Classifier  

## Workflow  
1. Load and explore the dataset.  
2. Preprocess the data (handle missing values, normalization if needed).  
3. Split dataset into **training** and **testing** sets.  
4. Train multiple machine learning models.  
5. Evaluate models using:  
   - Accuracy  
   - Recall  
6. Compare model performances to find the best classifier.  

## Results  

### Comparison by Accuracy
| Model                   | Accuracy | Recall  |
|--------------------------|----------|---------|
| Logistic Regression      | **0.8852** | **0.9062** |
| Decision Tree Classifier | 0.7869   | 0.7188  |
| SVC                      | 0.7049   | 0.8750  |
| KNeighbors Classifier    | 0.6885   | 0.7500  |

### Comparison by Recall
| Model                   | Accuracy | Recall  |
|--------------------------|----------|---------|
| Logistic Regression      | **0.8852** | **0.9062** |
| SVC                      | 0.7049   | 0.8750  |
| KNeighbors Classifier    | 0.6885   | 0.7500  |
| Decision Tree Classifier | 0.7869   | 0.7188  |

### Final Report  
- **Best model based on Accuracy**: Logistic Regression (0.8852)  
- **Best model based on Recall**: Logistic Regression (0.9062)  

## How to Run  
1. Open `heartattack.ipynb` in Jupyter Notebook or VSCode.  
2. Run all cells step by step.  
3. View the final evaluation metrics to confirm the best-performing model.  

## Future Improvements  
- Add more classification models (Random Forest, Gradient Boosting, XGBoost, LightGBM, etc.).  
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).  
- Feature selection to improve accuracy.  
- Deploy the best model using Flask / FastAPI for real-world predictions.  
