from flask import Flask, request,Response
from rossmann.Rossmann import Rossmann
import pickle
import pandas as pd
import os
import json


#loading model
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))


def load_data(store_id):
	df = pd.read_csv('datasets/test.csv', low_memory=False) #adjusted data path from root
	df_store_raw = pd.read_csv('datasets/store.csv', low_memory=False)

	#merge test dataset + stores info
	df_test= pd.merge(df, df_store_raw,how='left',on='Store')

	#choose store to prediction
	df_test = df_test[df_test['Store'] == store_id]

	#check if found store
	if not df_test.empty:
		#remove closed days
		df_test = df_test[df_test['Open']!=0]
		df_test = df_test[~df_test['Open'].isnull()]
		df_test = df_test.drop('Id',axis=1)

	else:
		data = 'Store Number doesnt exists'

	return data


#initialize API
app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def rossmann_predict():

    #getting input
    test_json = request.json
    print(test_json)

    #testing data input
    if test_json: #data ok
        
        if isinstance(test_json,dict): #if equals dict, means only 1 line 
            
            store_id = test_json.get('store_number')

            print(store_id)

            store_data = load_data(store_id)

            print(store_data)

    else: #empty
        return Response({}, status=200, mimetype='application/json') 
    
    if store_data != 'Store Number doesnt exists':
        #instantiate class    
        pipeline = Rossmann()
        
        #data cleaning
        df1 = pipeline.data_cleaning(store_data)
        print(df1)

        #feature engineering
        df2 = pipeline.feature_engineering(df1)
        print(df2)

        #data preparation
        df3 = pipeline.data_preparation(df2)
        print(df3)

        #predict
        df_response = pipeline.get_prediction(model, store_data, df3)
        
        print(df_response)

        df_response = df_response[['store','prediction']].groupby('store').sum().reset_index()

        return df_response

    else:
        return store_data

if __name__ == '__main__':
	port = os.environ.get('PORT', 5000)
	app.run(host='0.0.0.0', port=port, debug=True)