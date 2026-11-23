import pandas as pd 
import datetime as dt
import numpy as np 
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from feature_engine.creation import CyclicalFeatures
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


base_dir=Path(__file__).resolve().parent.parent
data_dir=base_dir/'data'
# laoding datasets
train_df=pd.read_csv(data_dir/'train.csv',index_col='id')
test_df=pd.read_csv(data_dir/'test.csv',index_col='id')
oil_df=pd.read_csv(data_dir/'oil.csv')
hol_event_df=pd.read_csv(data_dir/'holidays_events.csv')
stores_df=pd.read_csv(data_dir/'stores.csv')
sub=pd.read_csv(data_dir/'sample_submission.csv')


# renaming some features
stores_df=stores_df.rename(columns={'type':'store_type'})
hol_event_df=hol_event_df.rename(columns={'type':'day_type'})


#preprocessing oil dataset
oil_df['was_oilprice_missing']=oil_df['dcoilwtico'].isna()
oil_df['dcoilwtico']=oil_df['dcoilwtico'].ffill().bfill()
oil_df['date'] = pd.to_datetime(oil_df['date'])
oil_df['was_oildate_missing']=False
oil_df = (oil_df.set_index('date').asfreq('D').reset_index())
oil_df['was_oildate_missing']=oil_df['was_oildate_missing'].fillna(True)
oil_df['date']=oil_df['date'].astype(str)
oil_df['dcoilwtico']=oil_df['dcoilwtico'].ffill().bfill()
oil_df['was_oilprice_missing']=oil_df['was_oilprice_missing'].fillna(True)



#preprocessing holiday_event dataset
hol_event_df.loc[hol_event_df['transferred'],'day_type'] = 'HolidayTransferred'
hol_event_df.loc[hol_event_df['day_type']=='Transfer','day_type']='Holiday'
hol_event_df=hol_event_df.drop(columns=['transferred'])



# merging datasets
main_oil_df=pd.merge(test_df,oil_df,on='date',how='left')
main_oil_stores_df=pd.merge(main_oil_df,stores_df,on='store_nbr',how='left')

store_locs = main_oil_stores_df[['store_nbr','city','state']].drop_duplicates()
nat = hol_event_df[hol_event_df['locale'] == 'National'].merge(store_locs, how='cross')
reg = hol_event_df[hol_event_df['locale'] == 'Regional'].merge(store_locs, left_on='locale_name', right_on='state')
loc = hol_event_df[hol_event_df['locale'] == 'Local'].merge(store_locs, left_on='locale_name', right_on='city')


store_holidays = (pd.concat([nat, reg, loc], ignore_index=True).drop_duplicates(subset=['date', 'store_nbr'])
                  [['date', 'store_nbr', 'day_type', 'locale', 'locale_name', 'description']])
full_df = main_oil_stores_df.merge(store_holidays, on=['date', 'store_nbr'], how='left')

#creating lag and rolling features
combined = pd.concat([train_df[['store_nbr', 'family', 'date', 'sales']],full_df[['store_nbr', 'family', 'date']].assign(sales=np.nan)])
g = combined.groupby(['store_nbr', 'family'])['sales']
combined['16day_lag'] = g.shift(16)
combined['28day_lag'] = g.shift(28)
combined['16day_roll']=g.transform(lambda s:s.shift(16).rolling(16,min_periods=12).mean())

full_df[['16day_lag','28day_lag','16day_roll']]=combined.iloc[len(train_df):][['16day_lag', '28day_lag', '16day_roll']].to_numpy()


#filling missing values in final training dataset
full_df['day_type']=full_df['day_type'].fillna('no_day_type')
full_df['locale']=full_df['locale'].fillna('no_locale')
full_df['locale_name']=full_df['locale_name'].fillna('no_locale_name')
full_df['description']=full_df['description'].fillna('no_description')


#parsing dates
full_df['date']=pd.to_datetime(full_df['date'])
full_df['year']=full_df['date'].dt.year
full_df['monthofyear']=full_df['date'].dt.month
full_df['dayofmonth']=full_df['date'].dt.day
full_df['dayofweek']=full_df['date'].dt.dayofweek
full_df['is_weekend']=full_df['dayofweek'].isin([5,6])
month_to_season={12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
full_df['season']=full_df['monthofyear'].map(month_to_season)


#changing categorical data from integer to object type
full_df['store_nbr']=full_df['store_nbr'].astype('object')
full_df['cluster']=full_df['cluster'].astype('object')


#creating and dropping features
x_test=full_df.copy()
del full_df
x_test['earthquake_aftermath']=False
x_test.drop(columns=['date','description'],inplace=True)


#encoder fuction 
normal_time_series_features=['monthofyear','dayofweek','season']
cat_features=['store_nbr','family','city','state','cluster','store_type','day_type','locale','locale_name']

class mycfe(BaseEstimator,TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x=X.copy()
        month = x['monthofyear'].to_numpy()
        day = x['dayofmonth'].to_numpy()
        year=x['year'].to_numpy()

        days_in_month = np.select( 
            [
                np.isin(month, [1, 3, 5, 7, 8, 10, 12]),
                np.isin(month, [4, 6, 9, 11]),
                (month ==2 ) & (((year % 4 ==0) & (year % 100!= 0)) | (year % 400==0)),
                (month == 2)
            ],
            [31, 30, 29,28],
            default=np.nan
        )

        angle = 2 * np.pi * (day - 1) / days_in_month
        x['day_sin']=np.sin(angle)
        x['day_cos']=np.cos(angle)
        x.drop(columns=['monthofyear'],inplace=True)
        return x
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['dayofmonth','year','day_sin','day_cos'], dtype=object)

preprocessor_encoder=ColumnTransformer(transformers=[
    ('onehot_encoder',OneHotEncoder(sparse_output=True,handle_unknown='ignore'),cat_features),
    ('cyclic_encoder1',CyclicalFeatures(max_values={'dayofweek':7,'monthofyear':12,'season':4}),normal_time_series_features),
    ('cyclic_encoder2',mycfe(),['dayofmonth','monthofyear','year']),            
],remainder='passthrough')

#loading encoder and model
model_dir = Path(__file__).resolve().parent
preprocessor_encoder=joblib.load(model_dir/'preprocessor_encoder.joblib')
model=joblib.load(model_dir/'final_model.joblib')


#encoding and prediction
x_test_processed = preprocessor_encoder.transform(x_test)
y_predict = np.expm1(model.predict(x_test_processed))
y_predict = np.clip(y_predict, 0, None)

#saving to sample_submission.csv
submission = pd.DataFrame({'id': test_df.index, 'sales': y_predict})
assert len(submission) == len(sub)
submission.to_csv(data_dir / 'submission.csv', index=False)