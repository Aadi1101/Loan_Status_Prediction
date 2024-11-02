import os,sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from src.logger import logging
from src.exception import CustomException
import dill,json
import numpy as np
import random
from catboost import CatBoostClassifier


def save_model(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,param,n_features,epsilon,alpha,gamma):
    try:
        report = {}

        # Initialize the Q-table for reinforcement learning (state-action table)
        q_table = np.zeros((2**n_features, n_features))

        # Iterate through each model
        for model_name, model in models.items():
            logging.info(f"Evaluation initiated for {model_name}.")
            para = param[model_name]
            best_model_accuracy = 0  # Track the best model accuracy per model
            best_features = []  # Track the best feature set per model

            for episode in range(10):
                # Start with no features selected
                feature_selection = [0] * n_features
                state = get_state(feature_selection)  # Get initial state based on feature selection
                logging.info(f"Episode {episode + 1}/{10} started for {model_name}. Initial state: {state}.")

                for step in range(5):  # Set a maximum number of steps per episode
                    # Choose an action and toggle feature selection
                    action = choose_action(state, epsilon, n_features, q_table)
                    feature_selection[action] = 1 - feature_selection[action]  # Toggle feature selection
                    next_state = get_state(feature_selection)
                    logging.info(f"Step {step + 1}: Action {action} chosen, toggling feature {action}. Updated feature selection: {feature_selection}.")

                    # Collect selected features based on the current feature selection
                    selected_features = [i for i in range(n_features) if feature_selection[i] == 1]

                    if not selected_features:
                        logging.info("No features selected, skipping model training for this step.")
                        test_model_accuracy = 0  # If no features selected, set accuracy to 0
                    else:
                        # Use GridSearchCV to find the best hyperparameters for selected features
                        gs = GridSearchCV(model, para, cv=5, verbose=1, n_jobs=-1)
                        logging.info(f"GridSearchCV initiated for {model_name} with selected features: {selected_features}.")
                        gs.fit(x_train[:, selected_features], y_train)
                        best_params = gs.best_params_
                        updated_model = model.__class__(**best_params)
                        # Update model with the best parameters and train on selected features
                        if isinstance(model, CatBoostClassifier):
                            updated_model = CatBoostClassifier(**best_params)
                        else:
                            updated_model = model.set_params(**best_params)
                        logging.info(f"Training {model_name} with best params on selected features.")
                        updated_model.fit(x_train[:, selected_features], y_train)

                        # Predict and evaluate model performance on test data
                        print("utils selected features: ",selected_features)
                        y_test_pred = updated_model.predict(x_test[:, selected_features])
                        test_model_accuracy = accuracy_score(y_test, y_test_pred)

                    # Update Q-table based on the test accuracy as reward
                    update_q_value(state, action, test_model_accuracy, next_state, q_table, alpha, gamma)
                    logging.info(f"Q-value updated for state {state}, action {action} with reward {test_model_accuracy}.")
                    state = next_state  # Move to the next state

                    # Update best accuracy and feature set if this configuration performs better
                    if test_model_accuracy:
                        best_model_accuracy = test_model_accuracy
                        best_features = selected_features
                    print(best_features)

                # Decay epsilon after each episode to reduce exploration over time
                epsilon = max(0.01, epsilon * 0.99)
                logging.info(f"Epsilon decayed to {epsilon} after episode {episode + 1} for {model_name}.")

            # Store the best accuracy and feature set for each model in the report
            report[model_name] = {
                "best_accuracy": best_model_accuracy,
                "best_features": best_features,
                "best_params": best_params
            }
            logging.info(f"Completed evaluation for {model_name} with best accuracy {best_model_accuracy}. Best params: {best_params}, Best features: {best_features}.")

        return report
    except Exception as e:
        raise CustomException(e, sys)

def save_json_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def get_state(feature_selection):
    try:
        return sum([int(f) * (2**i) for i,f in enumerate(feature_selection)])
    except Exception as e:
        raise CustomException(e,sys)

def choose_action(state,epsilon,n_features,q_table):
    try:
        if random.uniform(0,1) < epsilon:
            return random.randint(0,n_features-1)
        return np.argmax(q_table[state])
    except Exception as e:
        raise CustomException(e,sys)

def update_q_value(state,action,reward,next_state,q_table,alpha,gamma):
    try:
        best_next_action = np.argmax(q_table[next_state])
        q_table[state,action] = (1-alpha) * q_table[state,action] + alpha * (reward + gamma * q_table[next_state,best_next_action])
    except Exception as e:
        raise CustomException(e,sys)