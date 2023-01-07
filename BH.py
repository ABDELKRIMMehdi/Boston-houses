import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,sys
from IPython.display import display

fit_verbosity = 1
"""
#----------------------------------------------------------------------------------- dataset a partir de keras :
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(test_split=0.2, seed=113)

"""
#----------------------------------------------------------------------------------- dataset a partir d'un CSV :

#from google.colab import files
#uploaded = files.upload ()
data = pd.read_csv(r'%PATH TO%BostonHousing.csv', header=0)
display(data.head(5).style.format("{0:.2f}").set_caption("Quelques lignes du dataset :"))
print('Données manquantes : ',data.isna().sum().sum(), '  la taille est : ', data.shape)

#---- Melanger les données du dataset 
#---- diviser => x,y (medv est le prix)
#
data = data.sample(frac=1., axis=0)
data_train = data.sample(frac=0.7, axis=0)
data_test  = data.drop(data_train.index)

# ---- diviser => x,y (medv est le prix)
#
x_train = data_train.drop('medv',  axis=1)
y_train = data_train['medv']
x_test  = data_test.drop('medv',   axis=1)
y_test  = data_test['medv']

print('La taille originale des données était : ',data.shape)
print('x_train : ',x_train.shape, 'y_train : ',y_train.shape)
print('x_test  : ',x_test.shape,  'y_test  : ',y_test.shape)

display(x_train.describe().style.format("{0:.2f}").set_caption("Avant la normalisation :"))

mean = x_train.mean()
std  = x_train.std()
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std

display(x_train.describe().style.format("{0:.2f}").set_caption("Apres la normalisation :"))
display(x_train.head(5).style.format("{0:.2f}").set_caption("Quelques lignes du dataset :"))

x_train, y_train = np.array(x_train), np.array(y_train)
x_test,  y_test  = np.array(x_test),  np.array(y_test)

def get_model_v1(shape):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    model.add(keras.layers.Dense(32, activation='relu', name='Dense_n1'))
    model.add(keras.layers.Dense(64, activation='relu', name='Dense_n2'))
    model.add(keras.layers.Dense(32, activation='relu', name='Dense_n3'))
    model.add(keras.layers.Dense(1, name='OutputLayer'))
    
    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] )
    return model

model=get_model_v1( (13,) )
model.summary()

history = model.fit(x_train,
                    y_train,
                    epochs          = 60,
                    batch_size      = 10,
                    verbose         = fit_verbosity,
                    validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('x_test / loss      : {:5.4f}'.format(score[0]))
print('x_test / mae       : {:5.4f}'.format(score[1]))
print('x_test / mse       : {:5.4f}'.format(score[2]))

df=pd.DataFrame(data=history.history)
display(df)

print("min( val_mae ) : {:.4f}".format( min(history.history["val_mae"]) ) )

loss_train = history.history['mse']
loss_val = history.history['val_mse']
epochs = range(0,60)
plt.plot(epochs, loss_train, 'g', label='mse')
plt.plot(epochs, loss_val, 'b', label='val_mse')
plt.title('MSE')
plt.xlabel('Epochs')
plt.ylabel('Valeurs')
plt.legend()
plt.show()

mae_train = history.history['mae']
mae_val = history.history['val_mae']
epochs = range(0,60)
plt.plot(epochs, mae_train, 'r', label='mae')
plt.plot(epochs, mae_val, 'b', label='val_mae')
plt.title('MAE')
plt.xlabel('Epochs')
plt.ylabel('Valeurs')
plt.legend()
plt.show()

my_data = [ 1.26425925, -0.48522739,  1.0436489 , -0.23112788,  1.37120745,
       -2.14308942,  1.13489104, -1.06802005,  1.71189006,  1.57042287,
        0.77859951,  0.14769795,  2.7585581 ]
real_price = 10.4

my_data=np.array(my_data).reshape(1,13)

predictions = model.predict( my_data )
print("Prediction : {:.2f} K$".format(predictions[0][0]))
print("Reality    : {:.2f} K$".format(real_price))
