## 📊 Dataset

The dataset used for this project is publicly hosted on Kaggle. Because of file size limits, the raw data files are not included in this repository. 

* **Source:** [Kaggle - Store Sales Forecasting Challenge](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

## Objective
Explore Ecuador's Stores' Sales Times Series dataset and build machine learning model to predict sales.

## Assumptions Made
* Decimal point shift for 2016-10-7's outlier

## Note 
* This project is for educational purposes. Model results are intermediate and should not be treated as production-ready predictions.
* CV's with optuna has optimistic score cause parameters were chosen for best score .
* Training and Testing file consists of explantion and understanding.
* All required features should be available at prediction time .

## How to Run
  1. Clone this repository. 
  2. Install required dependency. 
  3. Download the dataset from kaggle's link provided above. 
  4. Place the downloaded datasets to data/ as 
    '''
    data/
    |--holidays_events.csv
    |--oil.csv
    |--stores.csv
    |--sample_submission.csv
    |--test.csv
    |--train.csv
    '''
  3. Run pip install -r requirement.txt on your environment.
    * (Optional) You can tune and retrain preexisting model by modifying training_and_testing/Sales_Prediction.ipynb ,then final_model/   final_model_training.ipynb then finally 16days_prediction.py
    * (Optional) Yu can replace test.csv and supplementary datasets with your own dataset, then remove lag and rolling features for final_model_training 
  5. Run each cells of final_model/final_model_training.iynb to get you preprocessor_encoder and model as 
    '''
    final_model/
    |--final_model.joblib
    |--preprocessor_encoder.joblib
  5. Run 16_days_prediction.py to get predictions per id's in data/submission.csvY