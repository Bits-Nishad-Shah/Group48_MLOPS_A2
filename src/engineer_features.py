from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from autofeat import AutoFeatClassifier
import featuretools as ft, pandas as pd

def engineer_features_with_featuretools(df_clean):
    """
    Applies feature engineering with Feature Tools
    """
    # Feature engineering with FeatureTools
    es = ft.EntitySet(id="titanic")

    # Add the dataframe to the EntitySet, specifying the index column (primary key)
    es = es.add_dataframe(dataframe_name="passengers", dataframe=df_clean, index="passengerid")

    ## Perform Deep Feature Synthesis without suffix
    feature_matrix, feature_defs = ft.dfs(entityset=es, 
                                        target_dataframe_name="passengers")

    # Drop the overlapping columns from feature_matrix
    feature_matrix_cleaned = feature_matrix.drop(columns=df_clean.columns, errors='ignore')

    # Now join the generated features with the original dataframe, adding a suffix to avoid overlap
    df_clean = df_clean.join(feature_matrix_cleaned, rsuffix='_gen')

    return df_clean

def treat_missing_values(df_clean):
    """
    Null value treatment
    """
    # Fill missing values in 'cabin' using forward fill (preceding value)
    df_clean['cabin'] = df_clean['cabin'].fillna(method='ffill').fillna(method='bfill')
    df_clean['embarked'] = df_clean['embarked'].fillna(method='ffill').fillna(method='bfill')

    # Fill missing values in 'age' using the mean of the column
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())

    return df_clean

def apply_label_encoding(df_clean):
    """
    Label encoding for categorical features
    """
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Identify categorical columns
    categorical_columns = df_clean.select_dtypes(include=['string']).columns

    # Apply LabelEncoder to each categorical column
    for column in categorical_columns:
        label_encoder.fit(df_clean[column].astype(str))
        df_clean[column + '_encoded'] = label_encoder.transform(df_clean[column].astype(str))
        
    # Drop the original categorical columns if no longer needed
    df_clean = df_clean.drop(columns=categorical_columns)

    return df_clean, categorical_columns

def apply_one_hot_encoding(df_clean):
    """
    Applying one hot encoding
    """
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    # Fit the encoder on the training data
    encoder.fit(df_clean[['sex', 'embarked']])

    # Transform the training data
    encoded_cols = encoder.transform(df_clean[['sex', 'embarked']])
    encoded_df = pd.DataFrame(encoded_cols, 
                              columns=encoder.get_feature_names_out(['sex', 'embarked']), 
                              dtype=int)

    # Drop the original 'sex' and 'embarked' columns from the training dataframe
    df_clean = df_clean.drop(columns=['sex', 'embarked'])

    # Concatenate the original dataframe with the new encoded columns
    df_clean = pd.concat([df_clean, encoded_df], axis=1)

    return df_clean 

def split_data(df_clean):
    """
    Splits data into train and test. The test data will be used for validation.
    """
    # Seperating X and y
    X = df_clean.drop(columns=['survived'])
    y = df_clean['survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, categorical_columns):
    """
    Scaling numerical features
    """
    # Scaling features to prevent numerical underflow
    columns_to_scale = [colm for colm in X_train.columns if colm not in categorical_columns and colm not in ['sex', 'embarked']]
    scaler = StandardScaler()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    return X_train, X_test

def auto_feature_engineering_for_classification(X_train, y_train, X_test):
    """
    Using autofeat for classification
    """
    # Auto feature generation with AutoFeat
    autofeat_classifier = AutoFeatClassifier()
    autofeat_classifier.fit(X_train, y_train)
    X_train_transformed = autofeat_classifier.transform(X_train)
    X_test_transformed = autofeat_classifier.transform(X_test)
    return X_train_transformed, X_test_transformed
