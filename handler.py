from flask import Flask, request,Response
from rossmann.Rossmann import Rossmann
import pickle
import pandas as pd
import os


#loading model
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))

#initialize API
app = Flask(__name__)


#define endpoint
@app.route('/predict',methods=['POST'])
def rossmann_predict():

    #getting input
    test_json = request.get_json()
    
    #testing data input
    if test_json: #data ok
        
        if isinstance(test_json,dict): #only 1 line
            test_raw = pd.DataFrame(test_json,index=[0])
        
        else: #many 
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
    else: #empty
        return Response({}, status=200, mimetype='application/json') 
    
    
    #instantiate class    
    pipeline = Rossmann()
    
    #data cleaning
    df1 = pipeline.data_cleaning(test_raw)

    #feature engineering
    df2 = pipeline.feature_engineering(df1)
    
    #data preparation
    df3 = pipeline.data_preparation(df2)
    
    #predict
    df_response = pipeline.get_prediction(model, test_raw, df3)
    
    return df_response


if __name__ == '__main__':
	port = os.environ.get('PORT', 5000)
	app.run(host='0.0.0.0', port=port, debug=True)