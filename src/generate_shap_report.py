import shap, pandas as pd, pickle
import matplotlib.pyplot as plt, os 
# def calculate_shap_values(automl_object, X_train_transformed, X_test_transformed):
#     """
#     Calculates shap values and generates shap reports
#     """
#     # Shap Explainer
#     explainer = shap.Explainer(automl_object.model.estimator, 
#                             X_train_transformed)  
#     shap_values = explainer(X_test_transformed)

#     # Save SHAP summary plot
#     plt.figure(figsize=(10, 7))
#     shap.summary_plot(shap_values, X_test_transformed, feature_names=X_test_transformed.columns)
#     plt.savefig(os.getcwd() + '/artifacts/reports/shap/shap_summary_plot.png')
#     plt.close()

#     # Save SHAP decision plot
#     plt.figure(figsize=(10, 7))
#     shap.decision_plot(explainer.expected_value, shap_values.values, X_test_transformed)
#     plt.savefig(os.getcwd() + '/artifacts/reports/shap/shap_decision_plot.png')
#     plt.close()

#     # Save SHAP values to a CSV file
#     shap_values_df = pd.DataFrame(shap_values.values, columns=X_test_transformed.columns)
#     shap_values_df.to_csv(os.getcwd() + '/artifacts/reports/shap/shap_values.csv', index=False)

#     # Optionally, save the explanations to a pickle file
#     with open(os.getcwd() + '/artifacts/models/shap_values.pkl', 'wb') as f:
#         pickle.dump(shap_values, f)

def calculate_shap_values(tpot_object, X_train_transformed, X_test_transformed):
    """
    Calculates shap values and generates shap reports for the TPOT model
    """
    # Extract the fitted pipeline from the TPOT object
    trained_model = tpot_object.fitted_pipeline_.predict_proba  # Extract the trained pipeline

    # Shap Explainer
    explainer = shap.Explainer(trained_model, X_train_transformed)  # Use the trained model
    shap_values = explainer(X_test_transformed)

    # Save SHAP summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=X_test_transformed.columns)
    plt.savefig(os.getcwd() + '/artifacts/reports/shap/shap_summary_plot.png')
    plt.close()
    
    try:
        # Save SHAP decision plot
        plt.figure(figsize=(10, 7))
        shap.decision_plot(explainer.expected_value, shap_values.values, X_test_transformed)
        plt.savefig(os.getcwd() + '/artifacts/reports/shap/shap_decision_plot.png')
        plt.close()
        
        # Save SHAP values to a CSV file
        shap_values_df = pd.DataFrame(shap_values.values, columns=X_test_transformed.columns)
        shap_values_df.to_csv(os.getcwd() + '/artifacts/reports/shap/shap_values.csv', index=False)
    except Exception as e:
        print("Decision plot is not valid for this pipeline. Skipping.")

    # Optionally, save the explanations to a pickle file
    with open(os.getcwd() + '/artifacts/models/shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)