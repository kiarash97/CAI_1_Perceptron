import numpy as np
import matplotlib.pyplot as plt

class Artifical_Neuron:

    def __init__(self, numoffeatures):
        self.NumberOfFeatures = numoffeatures + 1 # 1 for bias
        self.random_weight_initializing()

    def random_weight_initializing(self):
        self.Weights = np.random.random((self.NumberOfFeatures))
        self.Weights[-1] = 1

    def sum(self, instance):
        return np.dot(self.Weights,instance)

    def change_weight(self,NewWeight):
        self.Weights = NewWeight

def add_bias_to_instances(instances):
    for i in range(len(instances)):
        instances[i] = np.append(instances[i] , [1])

def perceptron_gradien_desent(instances , desired , iternumber):
    LearningRate = 0.2
    Neuron1 = Artifical_Neuron(2)
    counter = 0
    while True:
        counter+=1
        for i in range(4):
            OldWeight = Neuron1.Weights
            Sum = Neuron1.sum(instances[i])
            Error = desired[i] - Sum
            NewWeight = Neuron1.Weights + (LearningRate * Error * instances[i])
            i+=1
            Neuron1.change_weight(NewWeight)
            if (abs(OldWeight[0] - NewWeight[0])<0.00001 and abs(OldWeight[1] - NewWeight[1])<0.00001 and abs(OldWeight[2] - NewWeight[2])<0.00001) or counter > iternumber  :
                return Neuron1


def plot_instances_classification(neuron, instances, desired):
    plt.figure(1)
    counter = 0
    # plot the data
    for i in instances:
        xdata = i[0]
        ydata = i[1]
        if desired[counter] == 1:
            plt.plot(xdata, ydata, 'bo')
        else:
            plt.plot(xdata, ydata, 'ro')
        counter += 1

    # plot the classification line
    xconstant = neuron.Weights[0]
    yconstant = neuron.Weights[1]
    bias = neuron.Weights[2]
    print("Weights: ", xconstant, ",", yconstant)
    print("bias: ", bias)
    x = np.linspace(-1, 2, 100)
    y = ((-x * xconstant) - bias) / yconstant
    plt.plot(x,y)
    plt.show()


i1 = np.array([1,1])
i2 = np.array([0,1])
i3 = np.array([1,0])
i4 = np.array([0,0])
desired = [1,-1,-1,-1]
instances = []
instances.append(i1)
instances.append(i2)
instances.append(i3)
instances.append(i4)
add_bias_to_instances(instances)

Neuron1 = perceptron_gradien_desent(instances , desired , 1000)
plot_instances_classification(Neuron1 , instances , desired)