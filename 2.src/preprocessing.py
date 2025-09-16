def load_data(path, filename):
    '''
    Function to load the data from a CSV file
    '''
    import pandas as pd
    return pd.read_csv(path + filename)



def drop_zip_columns(df):
    """
    Drops 'Unnamed: 0', ZIP_NN, 'Sub_product' and 'Sub_issue' columns 
    """
    cols_to_drop = [col for col in ['Unnamed: 0','ZIP_NN', 'Sub_product', 'Sub_issue'] if col in df.columns]
    return df.drop(columns=cols_to_drop)

def create_train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

def balance_data(X_train, y_train, over_strategy, under_strategy, k_neighbors=3, random_state=42):
    """
    Function to balance the data. A hybrid method is applied. 
    SMOTE and NearMiss for over and undersampling data.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss

    smote = SMOTE(sampling_strategy=over_strategy, random_state=random_state, k_neighbors=k_neighbors)
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    undersampler = NearMiss(sampling_strategy=under_strategy)
    X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_over, y_train_over)

    return X_train_balanced, y_train_balanced


def scale_data(X_train_balanced, X_test):
    '''
     To scale the training and test data using MinMaxScaler.
    '''
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scal = scaler.fit_transform(X_train_balanced)
    X_test_scal = scaler.transform(X_test)
    return X_train_scal, X_test_scal

