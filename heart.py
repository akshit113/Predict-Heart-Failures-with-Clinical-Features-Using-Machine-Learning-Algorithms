import numpy as np
from pandas import read_csv, DataFrame, concat, get_dummies
from seaborn import distplot
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def clean_age(df):
    old_values = df['age'].values.tolist()
    new_age_list = []
    new_age = 0
    for age in old_values:
        if age in range(40, 50):
            new_age = 1
        elif age in range(50, 60):
            new_age = 2
        elif age in range(60, 70):
            new_age = 3
        elif age in range(70, 80):
            new_age = 4
        elif age in range(80, 90):
            new_age = 5
        else:
            new_age = 6
        print(f'''{age}: {new_age}''')
        new_age_list.append(new_age)
    age_df = DataFrame(new_age_list, columns=['Age_Buckets'])
    df = concat([age_df, df], axis=1)
    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """
    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    return df


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    print(f'''Training Set: {str(len(train))}''')
    print(f'''Validation Set: {str(len(validate))}''')
    print(f'''Test Set: {str(len(test))}''')
    return train, validate, test


def clean_data(df):
    df = clean_age(df)
    df = one_hot_encode(df, ['Age_Buckets', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'])
    return df


def normalize_columns(df, colnames, scaler):
    for col in colnames:
        # Create x, where x the 'scores' column's values as floats
        x = df[[col]].values.astype(float)
        # Create a minimum and maximum processor object
        # Create an object to transform the data to fit minmax processor
        x_scaled = scaler.fit_transform(x)
        # Run the normalizer on the dataframe
        df[col] = DataFrame(x_scaled)

    print(f'''Normalized Columns: {colnames} using MinMaxScaler.''')

    return df


def main():
    fpath = r'C:\Users\akshitagarwal\Desktop\Keras\datasets\heart failure\predicting-heart-failures-with-clinical-features-using-machine-learning-algorithms\data.csv'
    df = read_csv(fpath)
    ser = df['age']
    distplot(df.serum_sodium, bins=20)
    print((df['serum_sodium'].min()))
    df = clean_data(df)
    # now will split data into train, validation and test sets respectively
    train_df, validate_df, test_df = train_validate_test_split(df, seed=42)
    cols_to_be_normalized = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine',
                             'serum_sodium', 'time']
    mmscaler = MinMaxScaler()
    train_df = normalize_columns(train_df, cols_to_be_normalized, scaler=mmscaler)
    print('sosdcadc')


if __name__ == '__main__':
    main()
