#from flaml import AutoML
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle, os
# from flaml import AutoML
# def apply_automl(X_train_transformed, X_test_transformed, y_train, y_test):
#     """
#     Train models using FLAML automl
#     """
#     # Initialize FLAML AutoML instance
#     automl = AutoML()

#     # Fit the AutoML model on the training data
#     automl.fit(X_train_transformed, y_train, task='classification', time_budget=120)  # Set time budget in seconds

#     # Predict on the test data
#     y_pred = automl.predict(X_test_transformed)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)

#     print(f"Accuracy: {accuracy:.4f}")
#     print("Classification Report:")
#     print(report)

#     # Saving Automl object
#     with open(os.getcwd() + '/artifacts/models/best_estimator.pkl', 'wb') as f:
#         pickle.dump(automl, f)

#     return automl

def apply_automl(X_train_transformed, X_test_transformed, y_train, y_test):
    """
    Train models using TPOT AutoML
    """
    # Initialize TPOTClassifier instance
    automl = TPOTClassifier(verbosity=2, generations=5, population_size=50, random_state=42)

    # Fit the TPOT model on the training data
    automl.fit(X_train_transformed, y_train)

    # Predict on the test data
    y_pred = automl.predict(X_test_transformed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save the best pipeline as a Python script
    automl.export(os.path.join(os.getcwd(), 'artifacts/models/best_pipeline.py'))

    # Save the trained automl object for future predictions
    with open(os.path.join(os.getcwd(), 'artifacts/models/best_estimator_tpot.pkl'), 'wb') as f:
        pickle.dump(automl.fitted_pipeline_, f)

    return automl
