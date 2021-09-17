
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st


import joblib
#Load the model to disk
model = joblib.load(r"broadband_model.sav")
from PIL import Image



from preprocessing import preprocess
def main():
    # Setting Application title
    st.title('Broadband Customer Churn Prediction App')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional Broadband internet provider company use case. 
     The application is functional for both online prediction and batch data prediction.. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('Pycoders.jpg')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        # Based on our optimal features selection
        tenure = st.number_input('Tenure', value=0)
        contract_month = st.number_input('Contract_month', value=0)
        ce_expiry = st.number_input('Contract expiry date', value=0)
        secured_revenue = st.number_input('Monthly revenue', value=0)
        complaint_cnt = st.number_input('complaint_cnt', value=0)
        with_phone_service = st.selectbox('with_phone_service',('0', '1'))
        bandwidth_100M = st.selectbox('bandwidth_100M',('0', '1') )
        bandwidth_100MFTTO = st.selectbox('bandwidth_100M (FTTO)', ('0', '1'))
        bandwidth_10M = st.selectbox("bandwidth_10M", ('0', '1'))
        bandwidth_300MFTTO = st.selectbox("bandwidth_300M (FTTO)", ('0', '1'))
        bandwidth_30M = st.selectbox("bandwidth_30M", ('0', '1'))
        bandwidth_500MFTTO = st.selectbox("bandwidth_500M (FTTO)", ('0', '1'))
        bandwidth_50M = st.selectbox("bandwidth_50M", ('0', '1'))
        bandwidth_BELOW10M = st.selectbox("bandwidth_BELOW 10M", ('0', '1'))

        data = {
            'tenure': tenure,
            'contract_month': contract_month,
            'ce_expiry': ce_expiry,
            'secured_revenue': secured_revenue,
            'complaint_cnt': complaint_cnt,
            'with_phone_service': with_phone_service,
            'bandwidth_100M': bandwidth_100M,
            'bandwidth_100MFTTO': bandwidth_100MFTTO,
            'bandwidth_10M': bandwidth_10M,
            'bandwidth_300MFTTO': bandwidth_300MFTTO,
            'bandwidth_30M': bandwidth_30M,
            'bandwidth_500MFTTO': bandwidth_500MFTTO,
            'bandwidth_50M': bandwidth_50M,
            'bandwidth_BELOW10M': bandwidth_BELOW10M,

        }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # Preprocess inputs
        #preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(features_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the Company. It is predicted correctly with 98% that the customer will churn.')
            else:
                st.success('No, the customer is happy with Broadband internet provider Company. It is predicted correctly with 98% that the customer will not churn.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                # Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the Company.',
                                                       0: 'No, the customer is happy with Broadband internet provider Company.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)


if __name__ == '__main__':
    main()
