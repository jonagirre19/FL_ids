import pandas as pd
from sklearn import preprocessing

def get_data():
    train_data = _load_data("fl_ids/data/UNSW_NB15_training-set.csv")
    test_data = _load_data("fl_ids/data/UNSW_NB15_testing-set.csv")

    train_data = _preprocess_data(train_data)
    test_data = _preprocess_data(test_data)

    X_train, Y_train = _separate_features_and_labels(train_data)
    X_test, Y_test = _separate_features_and_labels(test_data)

    return X_train, Y_train, X_test, Y_test

def _load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["id", "attack_cat"])
    return data

def _preprocess_data(data):
    
    data = data.copy()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'string']).columns
    
    for column in categorical_cols:
        le = preprocessing.LabelEncoder()
        data[column] = le.fit_transform(data[column])

    data = data.apply(pd.to_numeric, errors='coerce')

    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    
    return pd.DataFrame(data_scaled, columns=data.columns)

def _separate_features_and_labels(data):
    Y = data.label.to_numpy()
    X = data.drop(columns="label").to_numpy()
    return X, Y