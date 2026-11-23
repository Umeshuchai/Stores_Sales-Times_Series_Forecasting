## Objective
Explore Ecuador's Stores' Sales Times Series dataset and build machine learning model to predict sales.

## 📊 Dataset
The dataset used for this project is publicly hosted on Kaggle. Because of file size limits, the raw data files are not included in this repository. 
  * **Source:** [Kaggle - Store Sales Forecasting Challenge](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

## Assumptions Made
* Decimal point shift for 2016-10-7's outlier

## Note 
* This project is for educational purposes. Model results are intermediate and should not be treated as production-ready predictions.
* CV's with optuna has optimistic score cause parameters were chosen for best score .
* Training and Testing file consists of explantion and understanding.
* All required features should be available at prediction time .

## How to Run

1. Clone this repository.
2. Install the required dependencies in your Python environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using the Kaggle link provided above.
4. Place the downloaded CSV files inside the `data/` directory with the following structure:

   ```text
   data/
   ├── holidays_events.csv
   ├── oil.csv
   ├── stores.csv
   ├── sample_submission.csv
   ├── test.csv
   └── train.csv
   ```

5. Run all cells in `final_model/final_model_training.ipynb` to generate the trained model and preprocessing pipeline:

   ```text
   final_model/
   ├── final_model.joblib
   └── preprocessor_encoder.joblib
   ```

6. Run the prediction script to generate predictions for each ID and save the resulting submission file to `data/submission.csv`:

   ```bash
   python 16_days_prediction.py
   ```

### Optional: Model Retraining and Custom Datasets

- You can experiment with hyperparameter tuning and retrain the model by modifying `training_and_testing/Sales_Prediction.ipynb`, followed by `final_model/final_model_training.ipynb`. Make sure to update `16_days_prediction.py` if your model or preprocessing pipeline changes.
- You can also replace `test.csv` and the supplementary datasets with your own data. However, if you use a different dataset, make sure to remove or appropriately modify the lag and rolling features in `final_model/final_model_training.ipynb`, as these features depend on the original time-series data structure.