
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import layers
from keras.layers import InputLayer
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# load admissions data
df = pd.read_csv('admissions_data.csv')
print(df.head())
print(df.describe())

## Data preprocessing
# split into features and labels
features = df.iloc[:,0:-1]
labels = df.iloc[:,-1]
#print(features)
#print(labels)
# split into training and testing set
features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size=0.2, random_state=2)
#normalize the numeric columns using ColumnTransformer
numerical_features = features.select_dtypes(include=['float64','int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", Normalizer(), numerical_columns)], remainder='passthrough')
features_train_normalized = ct.fit_transform(features_train)
features_test_normalized = ct.fit_transform(features_test)

def build_model(features):
    # create an instance of sequential model
    my_model = Sequential()
    # create input layer, shape is number of features in dataset
    input = InputLayer(input_shape = (features.shape[1],))
    # add input layer to my model
    my_model.add(input)
    # add hidden layer with relu activation function, 64 hidden units
    # WHY 64
    my_model.add(Dense(64, activation = 'relu'))
    # add output layer - single output regression, 1 neuron
    my_model.add(Dense(1))
    # print model summary
    print(my_model.summary())
    # Create an instance of the Adam optimizer with the learning rate equal to 0.01.
    opt = Adam(learning_rate = 0.01)
    # Compile: loss use the Mean Squared Error (mse);  metrics use the Mean Absolute Error (mae)
    my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
    return my_model
## apply the model
model = build_model(features_train_normalized)

## Fit and Evaluate the model
print("Fitting the model...")
history = model.fit(features_train_normalized, labels_train.to_numpy(), epochs=20, batch_size = 1, verbose = 1, validation_split=0.2)

print("Evaluating the model...")
val_mse, val_mae = model.evaluate(features_test_normalized, labels_test.to_numpy(), verbose = 0)
print("Final loss is:" + str(val_mse) + ", final metric is:" + str(val_mae))


# evauate r-squared score
y_pred = model.predict(features_test_normalized)

print(r2_score(labels_test,y_pred))

# plot MAE and val_MAE over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()
print('done')