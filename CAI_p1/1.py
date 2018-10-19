import numpy as np

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

Neuron1 = Artifical_Neuron(2)
NewWeight = np.array([1,  1 ,  1.5])  # 1.5 is for bias
Neuron1.change_weight(NewWeight)