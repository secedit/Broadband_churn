
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """

    # Defining the map function
    #def binary_map(feature):
    #    return feature.map({'Yes': 1, 'No': 0})

    # Encode binary categorical features
    #binary_list = ['SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']
    #df[binary_list] = df[binary_list].apply(binary_map)

    # Drop values based on operational options
    if (option == "Online"):
        columns = ['image', 'newacct_no', 'line_stat', 'bill_cycl', 'serv_type', 'serv_code',
                   'tenure', 'effc_strt_date', 'effc_end_date', 'contract_month',
                   'ce_expiry', 'secured_revenue', 'bandwidth',
                   'term_reas_code', 'term_reas_desc', 'complaint_cnt',
                   'with_phone_service', 'churn', 'current_mth_churn']
        # Encoding the other categorical categoric features with more than two categories
        #df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    elif (option == "Batch"):

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        df2 = sc.fit_transform(df)


    else:
        print("Incorrect operational options")



    # feature scaling
    #sc = MinMaxScaler()
    #df['tenure'] = sc.fit_transform(df[['tenure']])
    #df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    #df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    return df2