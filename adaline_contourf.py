from adalineGD import AdalineGD
from load_iris import load_iris
import matplotlib.pyplot as plt
import numpy as np

#create a graph with decision region
from matplotlib.colors import ListedColormap
def plot_decision_regions(x, y, classifier, resolution=0.02):
    colors = ['red', 'blue', 'lightgreen']
    markers = ['o', 'x']
    labelList = ["Setosa", "Versicolor"]
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #finding the best graph size and plate
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    
    #np.arange(a, b, c) creates a array with min=a, max=b, interval=c
    #np.meshgrid(x,y) creates a matrix
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    #finding the predicted area which is 1 or -1
    z = classifier.prediction(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    #create the plot
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], c=colors[idx], marker=markers[idx], label=labelList[idx])
    
x, y, x_std = load_iris.define_xy()
iris_train = AdalineGD(eta=0.001, n_iter=500)
iris_train_beforeStandardization = iris_train.fit(x,y)
iris_train_afterStandardization = iris_train.fit(x_std,y)



#plot decision regions before standardization
plt.subplot(2,1,1)
plot_decision_regions(x, y, classifier = iris_train)
plt.title('Before Standardization')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

#plot decision regions after standardization
plt.subplot(2,1,2)
plot_decision_regions(x_std, y, classifier = iris_train)
plt.title('Before Standardization')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.subplots_adjust(hspace=0.5)
plt.show()