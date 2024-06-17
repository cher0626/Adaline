from load_iris import load_iris
from adalineGD import AdalineGD
import matplotlib.pyplot as plt

def costPlot(classifier):
        plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('cost')

x, y, x_std = load_iris.define_xy()
iris_train = AdalineGD(eta=0.001, n_iter=500)
iris_train_beforeStandardization = iris_train.fit(x,y)
iris_train_afterStandardization = iris_train.fit(x_std,y)

#cost plot before standardization
plt.subplot(2, 1, 1)
costPlot(iris_train_beforeStandardization)

#cost plot after standardization
plt.subplot(2, 1, 2)
costPlot(iris_train_afterStandardization)

plt.show()