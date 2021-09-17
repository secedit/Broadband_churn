
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(r"C:\Users\Tos i5 5\Desktop\Churn_Broadband_Customer.csv")


# The Unnamed:19 column has been permanently removed from the dataset.
df.drop("Unnamed: 19",axis = 1, inplace = True)


# Let's convert the variable type of the churn, current_mth_churn, line_stat and with_phone_service columns to integer.
df.churn.replace({'N':0,'Y':1}, inplace = True)
df.current_mth_churn.replace({'N':0,'Y':1}, inplace = True)
df.with_phone_service.replace({'N':0,'Y':1}, inplace = True)
df.with_phone_service = df.with_phone_service.astype(int)


# complaint_cnt column contains word ' customer/ user pass away'. we replace this word by 0 complaint.
# complaint_cnt has mixed dtype (integer and string)
# first we change integer in string, than replace words we want and than again change dtype to integer
df['complaint_cnt'] = df.complaint_cnt.astype('str')
df['complaint_cnt'] = df.complaint_cnt.str.replace('customer/ user pass away','0')
df['complaint_cnt'] = df.complaint_cnt.astype('int')

# Droping of no more usefule columns
# Every customer is billed monthly so bill_cycl column is not useful
# serv_type for each customer is BBS only one type of service so it is also not usefull
# service_code column is also not useful for us
# Both term_reas_desc column and term_reas_code have same meaning

df.drop(columns=['bill_cycl','serv_type','serv_code','term_reas_desc'],inplace=True)


# There are 510125 rows. But there are only 27605 unique id. Some unique id appear in the data more than once.
# Let's get one of each unique ID with the process here.
yedek = df.copy()
df = df.drop(['tenure', 'bandwidth', 'image', 'secured_revenue', 'complaint_cnt', 'current_mth_churn', 'line_stat', 'term_reas_code'], axis=1)
df = df.drop_duplicates(keep = 'last')
df = yedek.loc[df.index, :]


# Let's drop the duplicate id.
df = df.drop(index=297183)
df = df.drop(index=227698)
df = df.drop(index=65061)

# drop null values of eff_strt_date column
df.dropna(subset=['effc_strt_date'],inplace=True)


# Let's drop the columns that we think will not affect Churn.
df.drop(columns = ['effc_strt_date','effc_end_date', 'term_reas_code', 'line_stat', 'newacct_no','image'], inplace = True)


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
df.drop(df[to_drop], axis=1,inplace=True)


# one-hot encoding for variables with more than 2 categories
df2 = df.copy()
df2 = pd.get_dummies(df2, drop_first=True, columns = ['bandwidth'], prefix = ['bandwidth'])


# Dependent and independent variables were determined.
X = df2.drop('churn', axis=1)
y = df2['churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Defining the modelling function
def modeling(alg, alg_name, params={}):
    model = alg(**params)  # Instantiating the algorithm class and unpacking parameters if any
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance evaluation
    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ", acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ", pre_score)
        rec_score = recall_score(y_true, y_pred)
        print("recall: ", rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ", f_score)

    print_scores(alg, y_test, y_pred)

    return model


# Running RandomForestClassifier model
RF_model = modeling(RandomForestClassifier, 'Random Forest')

#Saving best model
import joblib
#Sava the model to disk
filename = 'model.sav'
joblib.dump(RF_model, filename)
