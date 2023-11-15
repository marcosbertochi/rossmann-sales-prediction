import streamlit as st
import pandas as pd
from rossmann.Rossmann import Rossmann
import pickle


#loading model
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))


def load_data(store_id):
	df = pd.read_csv('datasets/test.csv', low_memory=False) #adjusted data path from root
	df_store_raw = pd.read_csv('datasets/store.csv', low_memory=False)

	#merge test dataset + stores info
	df_test= pd.merge(df, df_store_raw,how='left',on='Store')

	#choose store to prediction
	df_test = df_test.loc[df_test['Store'] == store_id]

	#check if found store
	if not df_test.empty:
		#remove closed days
		df_test = df_test[df_test['Open']!=0]
		df_test = df_test[~df_test['Open'].isnull()]
		df_test = df_test.drop('Id',axis=1)

    #not found
	else:
		df_test = None

	return df_test

def main():

    # rossmann img
    image_url = "https://www.ring-center.de/fileadmin/user_upload/GLOBAL/brand_stores/logos/rossmann.jpg"
    st.image(image_url, use_column_width=True)
    st.title("Rossman Sales Prediction App")

    # user select store id
    store_number = st.slider("Select Store Number", min_value=1, max_value=1115, value=1, step=1)

    data = load_data(store_number)

    # trigger prediction
    if st.button("Predict"):
        st.text("Predicting...")

        #if found store id
        if data is not None:
            
            #instantiate class    
            pipeline = Rossmann()
            
            #data cleaning
            df1 = pipeline.data_cleaning(data)

            #feature engineering
            df2 = pipeline.feature_engineering(df1)

            #data preparation
            df3 = pipeline.data_preparation(df2)

            #predict
            df_response = pipeline.get_prediction(model, data, df3)

            #sumarizing result
            df_response['store'] = df_response['store'].astype(int)
            df_response = df_response[['store','prediction']].groupby('store').sum().reset_index()

            st.success(f"In the next 6 weeks for the store {df_response.iloc[:,0].values} the expected revenue is $ {df_response.iloc[:,1].values}")

        #store id not found
        else:
            st.error(f"Store {store_number} doesn't exists")

if __name__ == "__main__":
    main()
