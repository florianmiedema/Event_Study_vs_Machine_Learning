#################################### LSTM #####################################
# Importing packages
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline  
import datetime
from sklearn.model_selection import TimeSeriesSplit

# Importing the data
data = pd.read_excel('cleaned_data.xlsx')

# Setting timer 
start = datetime.datetime.now()

# Creating dataframe for saving RMSE scores 
rmse_df = pd.DataFrame(columns=['company', 'rmse_1', 'rmse_3', 'rmse_10'])

# Looping through companies
for i in range(2, len(data.columns)):

  # Initiating best rmse variable for each event window size
  best_rmse_1 = 99999
  best_rmse_3 = 99999
  best_rmse_10 = 99999

  # Taking one stock and the date column
  df = data.iloc[:, i:i+1]

  # Resetting the index
  df2 = df.reset_index()[df.columns[0]]

  # Scaling the data
  scaler = MinMaxScaler(feature_range=(0, 1))
  df_scaled = scaler.fit_transform(np.array(df2).reshape(-1, 1))

  # Initiating the time series split
  tscv = TimeSeriesSplit(gap=5, max_train_size=175, n_splits=3, test_size=10)

  for train_index, test_index in tscv.split(df_scaled):

    # Setting the epochs for tuning
    epoch_tuning = [10, 50, 100]

    for j in epoch_tuning:

      # Setting the learning rates for tuning
      lr_tuning = [0.01, 0.001, 0.0001]

      for k in lr_tuning:

        # Splitting the dataset
        X_train, X_test = df_scaled[train_index], df_scaled[test_index]
        y_train, y_test = df_scaled[train_index], df_scaled[test_index]

        # Imputing missing values
        X_train[np.isnan(X_train)] = 0
        y_train[np.isnan(y_train)] = 0
        X_test[np.isnan(X_test)] = 0
        y_test[np.isnan(y_test)] = 0

        # Reshaping the input 
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Initializing the LSTM model 
        model=Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(1, 1))) 
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = Adam(learning_rate=k)
        model.compile(optimizer=opt, loss='mean_squared_error')

        # Fitting the LSTM model to the training data 
        model.fit(X_train, y_train, validation_split=0.2, 
                  epochs=j, verbose=0) 

        # Predicting test set 
        y_pred = model.predict(X_test)

        # Transforming back to the original unscaled values
        y_pred = scaler.inverse_transform(y_pred)

        # Computing the RMSE performance of the model with event window size 1
        rmse_1 = math.sqrt(mean_squared_error(y_test[:1], y_pred[:1]))

        # Saving the lowest rmse of the model with event window size 1
        if rmse_1 <= best_rmse_1:
          best_rmse_1 = rmse_1
        else:
          continue

        # Computing the RMSE performance of the model with event window size 3
        rmse_3 = math.sqrt(mean_squared_error(y_test[:3], y_pred[:3]))

        # Saving the lowest rmse of the model with event window size 3
        if rmse_3 <= best_rmse_3:
          best_rmse_3 = rmse_3
        else:
          continue

        # Computing the RMSE performance of the model with event window size 10
        rmse_10 = math.sqrt(mean_squared_error(y_test[:10], y_pred[:10]))

        # Saving the lowest rmse of the model with event window size 10
        if rmse_10 <= best_rmse_10:
          best_rmse_10 = rmse_10
        else:
          continue

  # Inserting company name and corresponding RMSEs in a dataframe 
  rmse_df = pd.concat([rmse_df, 
                      pd.DataFrame([[df.columns[0], best_rmse_1, best_rmse_3,
                                     best_rmse_10]], 
                                    columns=['company', 'rmse_1', 'rmse_3',
                                             'rmse_10'])], 
                      ignore_index=True)

end = datetime.datetime.now()
runtime_lstm = end - start

print('Runtime LSTM: {}'.format(runtime_lstm))

print(rmse_df)

#################################### GRU ######################################

# Setting timer 
start = datetime.datetime.now()

# Creating dataframe for saving RMSE scores 
rmse_df = pd.DataFrame(columns=['company', 'rmse_1', 'rmse_3', 'rmse_10'])

# Looping through companies
for i in range(2, len(data.columns)):

  # Initiating best rmse variable for each event window size
  best_rmse_1 = 99999
  best_rmse_3 = 99999
  best_rmse_10 = 99999

  # Taking one stock and the date column
  df = data.iloc[:, i:i+1]

  # Resetting the index
  df2 = df.reset_index()[df.columns[0]]

  # Scaling the data
  scaler = MinMaxScaler(feature_range=(0, 1))
  df_scaled = scaler.fit_transform(np.array(df2).reshape(-1, 1))

  # Initiating the time series split
  tscv = TimeSeriesSplit(gap=5, max_train_size=175, n_splits=3, test_size=10)

  for train_index, test_index in tscv.split(df_scaled):

    # Setting the epochs for tuning
    epoch_tuning = [10, 50, 100]

    for j in epoch_tuning:

      # Setting the learning rates for tuning
      lr_tuning = [0.01, 0.001, 0.0001]

      for k in lr_tuning:

        # Splitting the dataset
        X_train, X_test = df_scaled[train_index], df_scaled[test_index]
        y_train, y_test = df_scaled[train_index], df_scaled[test_index]

        # Imputing missing values
        X_train[np.isnan(X_train)] = 0
        y_train[np.isnan(y_train)] = 0
        X_test[np.isnan(X_test)] = 0
        y_test[np.isnan(y_test)] = 0

        # Reshaping the input 
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Initializing the GRU model 
        model=Sequential()
        model.add(GRU(50, return_sequences=True, input_shape=(1, 1))) 
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = Adam(learning_rate=k)
        model.compile(optimizer=opt, loss='mean_squared_error')

        # Fitting the GRU model to the training data 
        model.fit(X_train, y_train, validation_split=0.2, 
                  epochs=j, verbose=0) 

        # Predicting test set 
        y_pred = model.predict(X_test)

        # Transforming back to the original unscaled values
        y_pred = scaler.inverse_transform(y_pred)

        # Computing the RMSE performance of the model with event window size 1
        rmse_1 = math.sqrt(mean_squared_error(y_test[:1], y_pred[:1]))

        # Saving the lowest rmse of the model with event window size 1
        if rmse_1 <= best_rmse_1:
          best_rmse_1 = rmse_1
        else:
          continue

        # Computing the RMSE performance of the model with event window size 3
        rmse_3 = math.sqrt(mean_squared_error(y_test[:3], y_pred[:3]))

        # Saving the lowest rmse of the model with event window size 3
        if rmse_3 <= best_rmse_3:
          best_rmse_3 = rmse_3
        else:
          continue

        # Computing the RMSE performance of the model with event window size 10
        rmse_10 = math.sqrt(mean_squared_error(y_test[:10], y_pred[:10]))

        # Saving the lowest rmse of the model with event window size 10
        if rmse_10 <= best_rmse_10:
          best_rmse_10 = rmse_10
        else:
          continue

  # Inserting company name and corresponding RMSEs in a dataframe 
  rmse_df = pd.concat([rmse_df, 
                      pd.DataFrame([[df.columns[0], best_rmse_1, best_rmse_3,
                                     best_rmse_10]], 
                                    columns=['company', 'rmse_1', 'rmse_3',
                                             'rmse_10'])], 
                      ignore_index=True)

end = datetime.datetime.now()
runtime_gru = end - start

print('Runtime GRU: {}'.format(runtime_gru))

print(rmse_df)

#################################### SRN ######################################

# Setting timer 
start = datetime.datetime.now()

# Creating dataframe for saving RMSE scores 
rmse_df = pd.DataFrame(columns=['company', 'rmse_1', 'rmse_3', 'rmse_10'])

# Looping through companies
for i in range(2, len(data.columns)):

  # Initiating best rmse variable for each event window size
  best_rmse_1 = 99999
  best_rmse_3 = 99999
  best_rmse_10 = 99999

  # Taking one stock and the date column
  df = data.iloc[:, i:i+1]

  # Resetting the index
  df2 = df.reset_index()[df.columns[0]]

  # Scaling the data
  scaler = MinMaxScaler(feature_range=(0, 1))
  df_scaled = scaler.fit_transform(np.array(df2).reshape(-1, 1))

  # Initiating the time series split
  tscv = TimeSeriesSplit(gap=5, max_train_size=175, n_splits=3, test_size=10)

  for train_index, test_index in tscv.split(df_scaled):

    # Setting the epochs for tuning
    epoch_tuning = [10, 50, 100]

    for j in epoch_tuning:

      # Setting the learning rates for tuning
      lr_tuning = [0.01, 0.001, 0.0001]

      for k in lr_tuning:

        # Splitting the dataset
        X_train, X_test = df_scaled[train_index], df_scaled[test_index]
        y_train, y_test = df_scaled[train_index], df_scaled[test_index]

        # Imputing missing values
        X_train[np.isnan(X_train)] = 0
        y_train[np.isnan(y_train)] = 0
        X_test[np.isnan(X_test)] = 0
        y_test[np.isnan(y_test)] = 0

        # Reshaping the input 
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Initializing the SRN model 
        model=Sequential()
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(1, 1))) 
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        opt = Adam(learning_rate=k)
        model.compile(optimizer=opt, loss='mean_squared_error')

        # Fitting the SRN model to the training data 
        model.fit(X_train, y_train, validation_split=0.2, 
                  epochs=j, verbose=0) 

        # Predicting test set 
        y_pred = model.predict(X_test)

        # Transforming back to the original unscaled values
        y_pred = scaler.inverse_transform(y_pred)

        # Computing the RMSE performance of the model with event window size 1
        rmse_1 = math.sqrt(mean_squared_error(y_test[:1], y_pred[:1]))

        # Saving the lowest rmse of the model with event window size 1
        if rmse_1 <= best_rmse_1:
          best_rmse_1 = rmse_1
        else:
          continue

        # Computing the RMSE performance of the model with event window size 3
        rmse_3 = math.sqrt(mean_squared_error(y_test[:3], y_pred[:3]))

        # Saving the lowest rmse of the model with event window size 3
        if rmse_3 <= best_rmse_3:
          best_rmse_3 = rmse_3
        else:
          continue

        # Computing the RMSE performance of the model with event window size 10
        rmse_10 = math.sqrt(mean_squared_error(y_test[:10], y_pred[:10]))

        # Saving the lowest rmse of the model with event window size 10
        if rmse_10 <= best_rmse_10:
          best_rmse_10 = rmse_10
        else:
          continue

  # Inserting company name and corresponding RMSEs in a dataframe 
  rmse_df = pd.concat([rmse_df, 
                      pd.DataFrame([[df.columns[0], best_rmse_1, best_rmse_3,
                                     best_rmse_10]], 
                                    columns=['company', 'rmse_1', 'rmse_3',
                                             'rmse_10'])], 
                      ignore_index=True)

end = datetime.datetime.now()
runtime_srn = end - start

print('Runtime SRN: {}'.format(runtime_srn))

print(rmse_df)