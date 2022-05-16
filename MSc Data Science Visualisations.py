###################### Script for Results Visualisation #######################
# Importing packages 
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt

# Setting working directory
os.chdir('C:/Users/flori/OneDrive/Documenten/\
Master Data Science & Society/Thesis/Data')

##################### EVENT STUDY #####################
# Importing stata data
stata_1 = pd.read_stata('event_study_1.dta')
stata_3 = pd.read_stata('event_study_3.dta')
stata_10 = pd.read_stata('event_study_10.dta')

# Dropping estimation window
stata_1 = stata_1[stata_1['tau'] >= 0]
stata_3 = stata_3[stata_3['tau'] >= 0]
stata_10 = stata_10[stata_10['tau'] >= 0]

# Dropping missing values
stata_1 = stata_1.dropna()
stata_3 = stata_3.dropna()
stata_10 = stata_10.dropna()

# Creating errors (true return minus predicted return) for the event studies
stata_1['error'] = stata_1['return'] - stata_1['NR']
stata_3['error'] = stata_3['return'] - stata_3['NR']
stata_10['error'] = stata_10['return'] - stata_10['NR']

# Computing RMSE
print('Event study')
print('Event study event window 1:\t\t{}'.format(
                                np.sqrt(mean_squared_error(stata_1['return'], 
                                                           stata_1['NR']))))
print('Event study event window 3:\t\t{}'.format(
                                np.sqrt(mean_squared_error(stata_3['return'], 
                                                           stata_3['NR']))))
print('Event study event window 10:\t{}'.format(
                                np.sqrt(mean_squared_error(stata_10['return'], 
                                                           stata_10['NR']))))
print()

##################### LSTM #####################
# Importing data
lstm = pd.read_excel('lstm_rmse.xlsx')

# Computing RMSE
print('LSTM')
print('LSTM event window 1:\t\t\t{}'.format(np.mean(lstm['rmse_1'])))
print('LSTM event window 3:\t\t\t{}'.format(np.mean(lstm['rmse_3'])))
print('LSTM event window 10:\t\t\t{}'.format(np.mean(lstm['rmse_10'])))
print()

##################### SRN #####################
# Importing data
srn = pd.read_excel('srn_rmse.xlsx')

# Computing RMSE
print('SRN')
print('SRN event window 1:\t\t\t\t{}'.format(np.mean(srn['rmse_1'])))
print('SRN event window 3:\t\t\t\t{}'.format(np.mean(srn['rmse_3'])))
print('SRN event window 10:\t\t\t{}'.format(np.mean(srn['rmse_10'])))
print()

##################### GRU #####################
# Importing data
gru = pd.read_excel('gru_rmse.xlsx')

# Computing RMSE
print('GRU')
print('GRU event window 1:\t\t\t\t{}'.format(np.mean(gru['rmse_1'])))
print('GRU event window 3:\t\t\t\t{}'.format(np.mean(gru['rmse_3'])))
print('GRU event window 10:\t\t\t{}'.format(np.mean(gru['rmse_10'])))
print()

##################### DESCRIPTIVE STATISTICS #####################
# Importing data
df = pd.read_excel('cleaned_data.xlsx')

# Creating dataframe
returns = pd.DataFrame(df.iloc[:,2])

# Renaming column name
returns.columns = ['return']

# Concatting all stock returns
for i in df.columns[3:]:
    returns = pd.concat([returns, df[i]])

# Copying returns from first stock
returns = pd.concat([returns, returns.iloc[:2565, 0]])

# Dropping empty column
returns = returns.iloc[2565:, 1]

# Generating descriptive statistics
print(returns.describe())

# Creating distribution
#returns.hist(bins=10000)

# Creating boxplot
#plt.boxplot(returns)
#plt.show()

#################### EVENT STUDY RMSE VS LSTM RSME ###################
es_rmse = [3.5568, 2.8882, 2.2805]
lstm_rmse = [0.2661, 0.5783, 0.7035]
event_window = [1, 3, 10]

X = ['1', '3', '10']
X_axis = np.arange(len(X))

plot = pd.DataFrame([[3.5568, 0.2661, 1], [2.8882, 0.5783, 3], 
                     [2.2805, 0.7035, 10]],
                    columns=['event_study', 'lstm', 'size'])

plot.plot(kind='bar')

plt.bar(X_axis - 0.15, plot['event_study'], width=0.3, label='Event study')
plt.plot(X_axis - 0.15, plot['event_study'], '-o', color='black')
plt.bar(X_axis + 0.15, plot['lstm'], width=0.3, label='LSTM')
plt.plot(X_axis + 0.15, plot['lstm'], '-o', color='black')

plt.xticks(X_axis, X)
plt.xlabel('Event window size')
plt.ylabel('RMSE')
plt.title('RMSE of Event Study and LSTM for Different Event Window Sizes')
plt.legend()
plt.show()

######################## RMSE ALL MODELS ######################## 
X = ['1', '3', '10']
X_axis = np.arange(len(X))

rmses = pd.DataFrame([[3.5568, 0.2661, 0.9096, 0.3837, 1], 
                      [2.8882, 0.5783, 1.4366, 0.8725, 3], 
                      [2.2805, 0.7035, 1.6992, 1.0263, 10]],
                    columns=['event_study', 'lstm', 'srn', 'gru', 'size'])

width = 0.2
far = 0.3
close = 0.1
 
plt.bar(X_axis - far, rmses['event_study'], width=width, label='Event study')
plt.bar(X_axis - close, rmses['lstm'], width=width, label='LSTM')
plt.bar(X_axis + close, rmses['srn'], width=width, label='SRN')
plt.bar(X_axis + far, rmses['gru'], width=width, label='GRU')

plt.xticks(X_axis, X)
plt.xlabel('Event window size')
plt.ylabel('RMSE')
plt.title('RMSEs for Different Event Window Sizes')
plt.legend()
plt.show()

############################ RMSE vs RUNTIME ############################ 
# Runtime
# Event study:		00:01:27.75 --> 0.01
# LSTM:			    30:59:37.47	--> 30.98		
# SRN:              13:20:40.17 --> 13.33
# GRU:	       		24:15:47.46 --> 24.26

x_es = 0.01
y_es = 3.5567

x_lstm = 30.98
y_lstm = 0.2661

x_srn = 13.33
y_srn = 0.4096

x_gru = 24.26
y_gru = 0.3837

plt.scatter(x_es, y_es, c='blue',
            linewidth=2,
            marker='s',
            edgecolor='blue',
            s=100,
            label='Event Study')

plt.scatter(x_lstm, y_lstm, c='orange',
            linewidths=2,
            marker='^',
            edgecolor='orange',
            s=150,
            label='LSTM')

plt.scatter(x_srn, y_srn, c='green',
            linewidth=2,
            marker='*',
            edgecolor='green',
            s=200,
            label='SRN')

plt.scatter(x_gru, y_gru, c='red',
            linewidth=2,
            marker='o',
            edgecolor='red',
            s=100,
            label='GRU')

plt.xlabel('Total runtime (Hours)')
plt.ylabel('RMSE')
plt.title('Predictive Performance vs Total Runtime')
plt.legend()
plt.show()

########################### RMSE DISTRIBUTION ########################### 
stata_1['error'].hist(bins=10)
stata_3['error'].hist(bins=10)
stata_10['error'].hist(bins=10)

plt.boxplot(stata_1['error'])
plt.show()

plt.boxplot(stata_3['error'])
plt.show()

plt.boxplot(stata_10['error'])
plt.show()





