import numpy as np
import pandas as pd

count=0
df=pd.read_csv('F:/Files/DataMining/midterm/companylist.csv',
                         usecols=[0],skiprows=[0], header=None)
list=df.values
for i in range(len(list)):
    filename = str(list[i])[2:-2] + '.csv'
    try:
        df = pd.read_csv('F:/Files/DataMining/midterm/data/' + filename,
                 usecols=[2], skiprows=[0, 1, 2, 3, 4, 5, 6], header=None)
    except FileNotFoundError:
        pass
    except pd.io.common.EmptyDataError:
        pass
    if len(df.values)>=5010:
        col=df.values[0:5010]
        col=np.nan_to_num(col)
        m=np.median(col)
        sigma = np.sqrt(np.var(col))
        col=(col-m)/sigma
        for j in range(5000):
            col[j+10]=col[j+10]>=np.average(col[j:j+10])
        col=col[10:5010]
        if count==0:
            data=col
            count=1
        else:
            data=np.append(data,col,axis=1)