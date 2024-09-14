from get_and_clean_data import get_raw_data, clean_df_headers, generate_directories
import os 
from dataprep.eda import create_report
from engineer_features import (engineer_features_with_featuretools, 
                               treat_missing_values, apply_label_encoding, 
                               apply_one_hot_encoding, split_data, scale_features,
                               auto_feature_engineering_for_classification)
from flaml_training import apply_automl
from generate_shap_report import calculate_shap_values

def main():
    """
    Entrypoint of the application. Parent function for the workflow
    """
    # Generate Directories
    generate_directories()

    # Reading raw data
    df_train, df_test = get_raw_data()

    # Cleaning headers 
    df_train_cleaned = clean_df_headers(df_train)
    df_test_cleaned = clean_df_headers(df_test)

    # Saving cleaned data
    df_train_cleaned.to_csv(os.getcwd() + '/datasets/processed/cleaned_train.csv', index=False)
    df_test_cleaned.to_csv(os.getcwd() + '/datasets/processed/cleaned_test.csv', index=False)

    # Creating EDA report on the train data
    create_report(df_train_cleaned).save(os.getcwd() + "/artifacts/reports/eda/dataprep_report.html")

    # Using Feature Tools for doing feature engineering
    df_train_cleaned = engineer_features_with_featuretools(df_train_cleaned)

    # Null value treatment
    df_train_cleaned = treat_missing_values(df_train_cleaned)

    # Applying label encoding
    df_train_cleaned, categorical_features = apply_label_encoding(df_train_cleaned)

    # Applying one hot encoding
    df_train_cleaned = apply_one_hot_encoding(df_train_cleaned)

    # Performing train test split
    X_train, X_test, y_train, y_test = split_data(df_train_cleaned)

    # Scaling features
    X_train, X_test = scale_features(X_train, X_test, categorical_features)

    # Engineering features with autofeat for Classification
    X_train_transformed, X_test_transformed = auto_feature_engineering_for_classification(X_train, 
                                                                                          y_train, 
                                                                                          X_test)
    
    # Applying automl and fetching the best estimator
    automl_object = apply_automl(X_train_transformed, X_test_transformed, y_train, y_test)

    # Generating SHAP report
    calculate_shap_values(automl_object, X_train_transformed, X_test_transformed)

if __name__ == "__main__":
    main()