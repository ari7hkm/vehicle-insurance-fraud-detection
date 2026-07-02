import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_loader import load_data



def preprocess(raw_data):
    binary_columns = [col for col in raw_data.columns if raw_data[col].nunique() == 2 and col != "FraudFound_P"]

    print(f"The number of binary columns: {len(binary_columns)}")
    print(binary_columns)

    print(raw_data)

    le = LabelEncoder()

    for col in binary_columns:
        raw_data[col] = le.fit_transform(raw_data[col])
        print(f"label mapping for ({col}): {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Encoding Categorical variables
        for col in ['VehiclePrice', 'DriverRating', 'AgeOfVehicle', 'BasePolicy']:
            print(f"The unique values in {col}: {raw_data[col].unique()}")

        vehicleprice_label = {'more than 69000': 1, '20000 to 29000': 0,  '30000 to 39000': 0, 'less than 20000': 1, '40000 to 59000': 1, '60000 to 69000': 0}
        ageofvehicle_label = {'new': 2, '2 years': 0, '3 years': 2, '4 years': 2, '5 years': 1, '6 years': 1, '7 years': 0, 'more than 7': 0}
        basepolicy_label = {'Liability': 0, 'Collision': 1, 'All Perils': 2}

        raw_data['VehiclePrice'] = raw_data['VehiclePrice'].map(vehicleprice_label)
        raw_data['AgeOfVehicle'] = raw_data['AgeOfVehicle'].map(ageofvehicle_label)
        raw_data['BasePolicy'] = raw_data['BasePolicy'].map(basepolicy_label)


        # Data Reduction
        print(raw_data)
        print(raw_data.index)
        useless_columns = ['Month', 'WeekOfMonth', 'DayOfWeek', 'DayOfWeekClaimed', 'WeekOfMonthClaimed', 'PolicyNumber']
        df = raw_data.drop(columns=['Month', 'WeekOfMonth', 'DayOfWeek', 'DayOfWeekClaimed', 'WeekOfMonthClaimed', 'PolicyNumber'], axis=1)

        print(df)
        # Data Transformation -- One-Hot Encoding
        dtype_change_string = ['RepNumber', 'Deductible', 'Year']
        for col in dtype_change_string:
            df[col] = df[col].astype(str)

        onehot_encoding_columns = ['Make', 'MonthClaimed', 'MaritalStatus', 'PolicyType', 'VehicleCategory', 'RepNumber', 'Deductible', 'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfPolicyHolder', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 'Year']
        print("The number of one-hot encoding target features: ", len(onehot_encoding_columns))    

        df = pd.get_dummies(df, columns=onehot_encoding_columns)

        # Drop Constant features
        onehot_encoded_columns = [col for col in df.columns if '_' in col]
        onehot_encoded_columns.remove("FraudFound_P")
        print("The Number of One-hot Encoded Columns: ", len(onehot_encoded_columns))

        constant_features = []
        for col in onehot_encoded_columns:
            if df[col].sum() <= 5:
                constant_features.append(col)
        print("The Number of Constant Features: ", len(constant_features))
        print(constant_features)

        df.drop(columns=constant_features, axis=1, inplace=True)

        # Data cleaning
        Q1 = df["Age"].quantile(0.25)
        Q3 = df["Age"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df["Age"] >= lower_bound) & (df["Age"] <= upper_bound)]

        def categorize_age(age):
            if age <= 20:
                return 0
            elif age <= 40:
                return 1
            elif age <= 65:
                return 2
            else:
                return 3

        df['Age'] = df['Age'].apply(categorize_age)

        return df


preprocess(load_data("data/fraud_oracle.csv"))