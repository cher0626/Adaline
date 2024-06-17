import pandas as pd
import numpy as np

class load_iris:
    def define_xy():
        df = pd.read_csv("iris.csv")
        x = df.iloc[0:100, [0,2]].values
        
        #feature scaling - standardization (x' = (x-mean(x))/x_std)
        x_std = np.copy(x)
        x_std[:,0] = (x[:,0] - x[:,0].mean())/x[:,0].std()
        x_std[:,1] = (x[:,1] - x[:,1].mean())/x[:,1].std()

        y = df.iloc[0:100, 4].values
        y = np.where(y=="Setosa", -1, 1)
        return x, y, x_std
