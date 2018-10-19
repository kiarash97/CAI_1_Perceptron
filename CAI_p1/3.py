import numpy as np
import matplotlib.pyplot as plt

class Artifical_Neuron:

    def __init__(self, numoffeatures):
        self.NumberOfFeatures = numoffeatures + 1 # 1 for bias
        self.random_weight_initializing()

    def random_weight_initializing(self):
        self.Weights = np.random.random((self.NumberOfFeatures))
        # self.Weights[-1] = 1

    def sum(self, instance):
        return np.dot(self.Weights,instance)

    def change_weight(self,NewWeight):
        self.Weights = NewWeight


def read_file(address, instances, desired):
    f = open(address,'r')
    for line in f:
        x = line.rstrip().split(',')

        for i in range(len(x)):
            x[i] = float(x[i])
        instances.append(np.array([x[0],x[1]]))
        if x[2] == 1 :
            desired.append(1.0)
        else :
            desired.append(-1.0)

def add_bias_to_instances(instances):
    for i in range(len(instances)):
        instances[i] = np.append(instances[i] , [1])

def perceptron_gradien_desent(instances , desired, iternumber , TotalError):
    LearningRate = 0.3
    Neuron1 = Artifical_Neuron(2)
    counter = 0
    while True:
        counter +=1
        for i in range(100):
            OldWeight = Neuron1.Weights
            Sum = Neuron1.sum(instances[i])
            Error = desired[i] - Sum # use for new weights
            TotalError.append(1/2 * (Error**2)) #this will need for plot
            NewWeight = Neuron1.Weights + (LearningRate * Error * instances[i])
            Neuron1.change_weight(NewWeight) #change the weights
            LearningRate = LearningRate / (1+ float(counter)/len(instances))  #change learning rate , faster learning

            if (abs(OldWeight[0] - NewWeight[0])<0.00001 and abs(OldWeight[1] - NewWeight[1])<0.00001 and abs(OldWeight[2] - NewWeight[2])<0.00001) or counter > iternumber  :
                return Neuron1

def normalize_instances(instances):
    for i in range(len(instances)):
        instances[i][0] = (instances[i][0] - 30.05882244669796) / 69.76903535022332 # x-min / max - min
        instances[i][1] = (instances[i][1] - 30.60326323428011) / 68.266172507926 # y-min / max - min

def activation_function(instance, neuron):
    if neuron.sum(instance) > 0 :
        return 1
    else :
        return -1

def plot_instances_classification(neuron, instances , desired):
    # plt.figure(1)
    plt.subplot(212)
    counter = 0
    # plot the data
    for i in instances:
        xdata = i[0]
        ydata = i[1]
        if desired[counter] == 1:
            plt.plot(xdata , ydata , 'bo')
        else:
            plt.plot(xdata , ydata , 'ro')
        counter+=1

    # plot the classification line
    xconstant = neuron.Weights[0]
    yconstant = neuron.Weights[1]
    bias = neuron.Weights[2]
    print ("Weights: ",xconstant ,",", yconstant)
    print ("bias: ",bias)
    x = np.linspace(0, 1, 100)
    y = ((-x * xconstant) - bias)  / yconstant
    plt.plot(x,y)

def plot_error(error):
    plt.subplot(211)
    plt.plot(error)



instances = []
desired = []
TotalError = []
read_file("P1-dataset.txt", instances , desired)
normalize_instances(instances)
add_bias_to_instances(instances)
Learned_Neuron = perceptron_gradien_desent(instances , desired, 10000 , TotalError)
plot_instances_classification(Learned_Neuron , instances , desired)
plot_error(TotalError)
plt.show()