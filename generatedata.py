import pandas as pd
import numpy as np

def generate(nrow):
    possibilities = {'x': [1, 1, 0, 0], 'y': [1, 0, 1, 0], 'out': [0, 1, 1, 0]}
    possibilities = pd.DataFrame(possibilities, columns=['x', 'y', 'out'])
    data = pd.concat([possibilities]*(nrow//4), ignore_index=True)

    for col in range(2):
        data.iloc[:,col] += np.random.normal(0, 0.5, nrow)

    return(data)

def in_out_split(df):
    return (df[['x','y']], df['out'])


print(generate(32))
