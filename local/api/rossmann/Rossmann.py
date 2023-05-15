import pickle
import pandas as pd
import inflection
import numpy as np
import datetime

#must encapsulate all transformation made along code (cleaning, transformation, encodes)
class Rossmann(object):
    
    #class initialize brings all encoder in pickle files
    def __init__(self):
        self.home_path = 'C:\\Users\\marco\\OneDrive\\Documents\\repos\\ds_em_producao\\rossmann-sales-prediction\\'
        self.competition_distance_scaler =   pickle.load(open(self.home_path+'local\\parameters\\competition_distance_scaler.pkl','rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path+'local\\parameters\\competition_time_month_scaler.pkl','rb'))
        self.promo_time_week_scaler =        pickle.load(open(self.home_path+'local\\parameters\\promo_time_week_scaler.pkl','rb'))
        self.year_scaler =                   pickle.load(open(self.home_path+'local\\parameters\\year_scaler.pkl','rb'))
        self.store_type_encoder =            pickle.load(open(self.home_path+'local\\parameters\\store_type_encoder.pkl','rb'))


    def data_cleaning(self, df):
        
        #transform to snake_case
        cols_old = df.columns
        snake_case = lambda x: inflection.underscore(x)
        cols_new = list(map(snake_case,cols_old))
        df.columns = cols_new

        df['date'] = pd.to_datetime(df['date'],format="%Y-%m-%d")
        df['open'] = df['open'].astype('int64')
        
        #Fill out NAs
        #CompetitionDistance - distance in meters to the nearest competitor store
        df['competition_distance'] = df['competition_distance'].apply(lambda x: 150000 if pd.isna(x) else x)

        #CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
        df['competition_open_since_month'] = df.apply(lambda x: x['date'].month if pd.isna(x['competition_open_since_month']) else x['competition_open_since_month'],axis=1)                                                                                          
        df['competition_open_since_year'] = df.apply(lambda x: x['date'].year if pd.isna(x['competition_open_since_year']) else x['competition_open_since_year'],axis=1)
                                                                                           

        #Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
        df['promo2_since_week'] = df.apply(lambda x: x['date'].week if pd.isna(x['promo2_since_week']) else x['promo2_since_week'],axis=1)
        df['promo2_since_year'] = df.apply(lambda x: x['date'].year if pd.isna(x['promo2_since_year']) else x['promo2_since_year'],axis=1)

        #PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
        df['promo_interval'].fillna(0, inplace=True)

        #translates month (in numbers) to month (in string) and check in a new attribute if sale was in active promo
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        df['month_map'] = df['date'].dt.month.map(month_map)
        df['is_promo'] = df[['promo_interval','month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        df['competition_open_since_month'] = df['competition_open_since_month'].astype('int64')
        df['competition_open_since_year'] = df['competition_open_since_year'].astype('int64')

        df['promo2_since_week'] = df['promo2_since_week'].astype('int64')
        df['promo2_since_year'] = df['promo2_since_year'].astype('int64')
        
        
        return df
    
        
    def feature_engineering(self, df2):
        
        #creating date features
        #year
        df2['year'] = df2['date'].dt.year

        #day
        df2['day'] = df2['date'].dt.day

        #month
        df2['month'] = df2['date'].dt.month

        #week_of_year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week

        #week-year
        df2['year_week'] = df2['date'].dt.strftime("%Y-%W")

        #competition_since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year= x['competition_open_since_year'], month= x['competition_open_since_month'], day=1), axis=1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype('int64')

        #promo_since
        df2['promo_since'] = df2['promo2_since_year'].astype(str)+'-'+df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x+'-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype('int64')

        #assortment => assortment level: a = basic, b = extra, c = extended
        df2['assortment'] = df2['assortment'].map({'a':'basic', 'b':'extra','c':'extended'})

        #state_holday => a = public holiday, b = Easter holiday, c = Christmas, 0 = None
        df2['state_holiday'] = df2['state_holiday'].map({'a':'public', 'b':'easter','c':'christmas','0':'regular'})
        
        #open == 0 means store closed and we cant predict any sales
        df2 = df2[(df2['open'] != 0)].reset_index(drop=True)

        ### 3.2 Attributes filtering
        cols_drop = ['open','promo_interval','month_map']
        df2 = df2.drop(cols_drop,axis=1)

        return df2
        
        
    def data_preparation(self, df5):
        
        ### 5.2 Rescaling
        #competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)

        #competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)

        #promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']].values)

        #year
        df5['year'] = self.year_scaler.transform(df5[['year']].values)

        ### 5.3 Transformation

        #### 5.3.1 Encoding

        #state_holiday => its a 'status' condition, fits with One Hot Encoding
        df5 = pd.get_dummies(df5,prefix=['state_holiday'],columns=['state_holiday'])

        #store_type => its a type, fits with Label Enconding
        df5['store_type'] = self.store_type_encoder.transform(df5['store_type'])

        #assortment => there is a order dependency, we can label as ordinal, fits with Ordinal Enconding
        label_enconding_map = {'basic':1, 'extra':2, 'extended':3}
        df5['assortment'] = df5['assortment'].map(label_enconding_map)

        #### 5.3.3 Nature Transformation

        #temporal variables have cyclical behavior thus must show it to model => use trigonometry to derive new attributes

        #day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi / 7)))

        #month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2 * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2 * np.pi / 12)))

        #day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2 * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2 * np.pi / 30)))

        #week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52)))
       
    
        cols_selected = ['store', 'promo', 'store_type','assortment', 'competition_distance','competition_open_since_month',
                         'competition_open_since_year', 'promo2','promo2_since_week','promo2_since_year','competition_time_month',
                         'promo_time_week','day_of_week_sin','day_of_week_cos','month_sin','month_cos', 'day_sin', 'day_cos',
                         'week_of_year_sin','week_of_year_cos']
        
        return df5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        
        #prediction
        pred = model.predict(test_data)
        
        #join prediction into original data
        original_data['prediction'] = np.expm1(pred)
        
        #returning to API in JSON format
        return original_data.to_json(orient='records',date_format='iso')