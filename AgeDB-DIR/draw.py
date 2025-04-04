import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#x = np.arange(0., np.e, 0.01)
#y1 = np.exp(-x)
#y2 = np.log(x)

f = r'C:\Users\rpu2\Desktop\revised NLL MSE MAE + LDS.csv'
df = pd.read_csv(f)


for name in ['NLL', 'NLL_LDS', 'MSE', 'MSE_LDS', 'MAE', 'MAE_LDS']: 
    '''
    a1, a2, a3, a4 = f'{name} Train Frob', f'{name} Train MAE', f'{name} Test Frob', f'{name} Test MAE' 
    print(f'{name} Train Frob', f'{name} Train MAE', f'{name} Test Frob', f'{name} Test MAE' )
    if a1 not in df.keys():
        print('---------',a1)
    if a2 not in df.keys():
        print('---------',a2)
    if a3 not in df.keys():
        print('---------',a3)
    if a4 not in df.keys():
        print('---------',a4)
    print(df.keys())
    '''
    x = df['Train label'].values
    y1 = df[f'{name} Train Frob'].values
    y2 = df[f'{name} Train MAE'].values
    y3 = df[f'{name} Test Frob'].values
    y4 = df[f'{name} Test MAE'].values



    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1,'r',label="Train Frob")
    ax1.plot(x, y3,'g', label="Train MAE")
    ax1.legend(loc=1)
    ax1.set_ylabel('Frob')#
    ax2 = ax1.twinx() # this is the important function
    ax2.plot(x, y2, 'y',label = "Test Frob")
    ax2.plot(x, y4, 'pink',label = "Test MAE")
    ax2.legend(loc=2)
    ax2.set_xlim([0, 100])
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Train Label')
    plt.savefig(f'./{name}.jpg')
    