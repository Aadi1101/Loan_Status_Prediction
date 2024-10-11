# Loan Status Prediction

## Project Overview
This project aims to predict the **loan approval status** based on various factors using multiple machine learning algorithms. The data contains attributes such as applicant income, coapplicant income, loan amount, and credit history, among others. The solution uses a **modular architecture**, where each component such as data ingestion, transformation, and model training is handled separately.

## Data Columns
The dataset contains the following columns:
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status` (Target Variable)

## Machine Learning Models Used
We have employed several machine learning algorithms for this task. Below is a list of models used:

```python
models = { 
    "Decision Tree": DecisionTreeClassifier(), 
    "AdaboostClassifier": AdaBoostClassifier(), 
    "Gradient Boosting Classifier": GradientBoostingClassifier(verbose=1), 
    "Random Forest Classifier": RandomForestClassifier(verbose=1), 
    "Support Vector Machine": SVC(verbose=True), 
    "K Nearest Neighbours": KNeighborsClassifier(), 
    "Naive Bayes": GaussianNB(), 
    "Catboost Classifier": CatBoostClassifier(verbose=1), 
    "Logistic Regression": LogisticRegression(verbose=1), 
    "XGBoost Classifier": XGBClassifier() 
}
```

### Hyperparameter Tuning
We used **GridSearchCV** for cross-validation and hyperparameter tuning. Below are the parameters for each model:

```python
params = {
    'Logistic Regression':{
        'penalty':['elasticnet','l1','l2']
    },
    'Decision Tree':{
        'max_depth':[10,20,30],
        'min_samples_split':[2,5,10]
    },
    'AdaboostClassifier':{
        'n_estimators':[100,150,200],
        'learning_rate':[0.1,0.01,0.001]
    },
    'Gradient Boosting Classifier':{
        'n_estimators':[100,150,200],
        'max_depth':[10,20,30],
        'learning_rate':[0.1,0.01,0.001]
    },
    'Random Forest Classifier':{
        'n_estimators':[450],
        'max_features':['log2'],
        'max_depth':[340],
        'min_samples_split':[3],
        'min_samples_leaf':[8,10,12],
        'criterion':['gini']
    },
    'Support Vector Machine':{
        'kernel':['linear','poly','sigmoid','rbf'],
        'gamma':['scale','auto']
    },
    'K Nearest Neighbours':{
        'metric':['euclidean']
    },
    'Naive Bayes':{

    },
    'Catboost Classifier':{
        'learning_rate':[0.1,0.01,0.001],
        'depth':[10,20,30],
        'iterations':[100,150,200],
        'l2_leaf_reg':[2,3,4]
    },
    "XGBoost Classifier":{}
}
```

## Docker
To pull the Docker image for this project, use the following command:
```bash
docker pull gogetama/loan_status_prediction
```

## Future Enhancements
1. **Model Interpretability**: Implement SHAP values to better interpret the predictions of the models.
2. **More Features**: Adding more financial and socio-economic features for better prediction accuracy.
3. **Model Optimization**: Further tune hyperparameters using more sophisticated methods like Random Search or Bayesian Optimization.
4. **UI Improvements**: Enhance the frontend for a better user experience.
5. **Model Deployment**: Deploy the model to a cloud service like AWS or Google Cloud for scalability.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Aadi1101/Loan_Status_Prediction/blob/main/LICENSE) file for more details.
