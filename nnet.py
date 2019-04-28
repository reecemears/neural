import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import random
import pickle
from plotnine import *
from generatedata import generate, in_out_split
import itertools

random.seed(331)

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

num_neurons = [2, 4, 8]
num_vectors = [16, 32, 64]
epochs = 1000
granularity = 0.01
grid = expand_grid({'x': np.arange(0, 1, granularity), 'y': np.arange(0, 1, granularity)})
grid = pd.DataFrame(grid, columns=['x', 'y'])

facet_history = pd.DataFrame(columns=['num_neurons', 'num_vectors', 'epoch', 'loss', 'val_loss'])
facet_heat = pd.DataFrame(columns=['num_neurons', 'num_vectors', 'x', 'y', 'out'])

validators = generate(64)

for neurons in num_neurons:
    for vectors in num_vectors:
        # Generate some training data
        train = generate(vectors)
        train_in, train_out = in_out_split(train)
        # Build and train a model
        model = Sequential([
            Dense(neurons, input_dim=2, activation=tf.keras.activations.sigmoid),
            Dense(1, activation=tf.keras.activations.sigmoid)
            ])
        sgd = SGD(lr=0.1)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        nnet = model.fit(train_in, train_out,
                        validation_data=in_out_split(validators),
                        epochs=epochs,
                        batch_size=1,
                        verbose=0)
        # Create MSE against epoch data
        history = pd.DataFrame({'num_neurons': [neurons]*epochs,
                                'num_vectors': [vectors]*epochs,
                                'epoch': range(1, epochs+1),
                                'loss': nnet.history['loss'],
                                'val_loss': nnet.history['val_loss']})
        facet_history = facet_history.append(history, ignore_index=True)
        # Create heat map
        grid['Prediction'] = model.predict(grid[['x','y']])
        grid['num_neurons'], grid['num_vectors'] = [neurons] * grid.shape[0], [vectors] * grid.shape[0]
        grid = grid[['num_neurons', 'num_vectors', 'x', 'y', 'Prediction']]
        facet_heat = facet_heat.append(grid, ignore_index=True)

# Plot heat maps
p = (ggplot(facet_heat, aes('factor(x)','factor(y)',fill='Prediction')) + geom_tile()
             + scale_fill_gradient(name='', low="#ffffff", high="#000000", limits=[0,1])
             + scale_x_discrete(name='x', breaks=["0", "1"])
             + scale_y_discrete(name='y', breaks=["0", "1"])
             + facet_grid('num_neurons ~ num_vectors'))
p.save(filename='heatmaps.png', height=10, width=10, units = 'in', res=1000)

print(facet_history)
# Plot MSE against epoch
p = (ggplot(facet_history, aes(x='epoch'))
             + geom_line(aes(y='loss', group=1), colour='#000000')
             + geom_line(aes(y='val_loss', group=2), colour='#ff0000')
             + ylab('Mean squared error') + xlab('Epochs')
             + facet_grid('num_neurons ~ num_vectors'))
p.save(filename='mse.png', height=10, width=10, units = 'in', res=1000)

       
             
