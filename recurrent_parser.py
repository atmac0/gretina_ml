import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans

import csv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from math import *
import random

from operator import itemgetter

import itertools

class DataWrangler:

    def __init__(self, filenames, max_clusters_per_file):
        self.dataset = []
        self.labels  = []

        self.energies = [1173, 1332] # possible energies present, usd for generating the labels
        self.deviance = 3 # distance in keV from a measured value that is considered the same energy
        
        self.max_interactions = 20
        self.dimensionality_of_interaction = 5 # number of dimensions on each interaction. 5 is energy, x, y, z, distance_from_origin

        self.train_porportion = 0.8
        self.max_clusters_per_file = max_clusters_per_file
    
        self.create_dataset(filenames)

        data_len = len(self.dataset)
        
        self.dataset = np.array( self.dataset )
        self.labels  = np.array( self.labels )


    def get_training_dataset():
        num_train = int(len(self.labels) * self.train_porportion)

        train_data   = self.dataset[:num_train]
        train_labels = self.labels[:num_train]

        test_data   = self.dataset[num_train:]
        test_labels = self.labels[num_train:]

        return train_data, train_labels, test_data, test_labels
        
    def create_dataset(self, filenames):
        for filename in filenames:
            cluster_counter = 0
            with open(filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                gamma_count = int(next(csvreader, None)[0])

                cluster = []
        
                for row in csvreader:
                    row = row[0].split(',')
                    row = [float(i) for i in row]
                    if(row[0] == -1):
                        # if end of clu
                        if(len(cluster) <= self.max_interactions):
                            cluster = self.sort_cluster(cluster)
                            label = self.create_label(cluster)
                        
                            cluster.extend([[0] * self.dimensionality_of_interaction] * (self.max_interactions - len(cluster))) # pad the list with empty interactions so it can be initialized to a rectangular numpy array

                            if(label != None):
                                self.dataset.append(cluster)
                                self.labels.append(label)
                        else:
                            print("There were too many interactions in that cluster. Consider raising the maximum number of interactions allowed in an input.")
                        cluster = []

                        # go to next file if max clusters read from current file
                        cluster_counter = cluster_counter + 1
                        if(cluster_counter >= self.max_clusters_per_file):
                            break
                    else:
                        row.append(self.distance_from_origin(row[1], row[2], row[3]))
                        cluster.append(row)

    # sort a cluster by distance from the origin (i.e. interactions closer to the origin appear first in the list)
    def sort_cluster(self, cluster):
        return sorted(cluster, key=itemgetter(4)) # itemgetter of 4 selects the distance from the origin as the value to be sorted

    # labels are going to be a list, in order, of the indices in the cluster that make up the gamma ray of the first interaction point
    def create_label(self, cluster):
        
        range_list = list(range(0, len(cluster)))

        for i in range(1, len(cluster)):
            possible_labels = list(itertools.combinations(range_list, i))
            for possible_label in possible_labels:
                if(possible_label[0] != 0):
                    break

                energy_sum = 0
                for index in possible_label:
                    energy_sum = energy_sum + cluster[index][0]

                if(self.energy_in_list(energy_sum, self.energies)):
                    return possible_label
        return None
    
    def distance_from_origin(self, x, y, z):
        return sqrt( pow(x, 2) + pow(y, 2) + pow(z, 2) )

    def get_dataset(self):
        return self.dataset

    def energy_in_list(self, energy, lst):
        for i in lst:
            if(self.near_value(i, energy)):
                return True
        return False

    def near_value(self, val, number):
        if( (val - self.deviance) < number < (val + self.deviance) ):
            return True
        return False
                   
def train_recurrent(data):

    train_data, train_labels, test_data, test_labels = data.get_training_dataset()
    print("Successfully obtained training input and labels")    

    n_hidden_layer = 32 # size of hidden layer
    n_output_layer = 2
    
    model = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(n_output_layer, return_sequences=True), input_shape=(max_interactions, dimensionality_of_interaction)),
        keras.layers.Bidirectional( keras.layers.LSTM(n_hidden_layer) ),
        keras.layers.Dense(2)
    ])    

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    

    print("Model go beep boop")
    model.fit(train_inputs, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    
    

def main():

    files = ['out_2502.csv', 'out_1332.csv', 'out_1173.csv']
    
    data_all = DataWrangler(files)
    
    data2 = DataWrangler(['out_2505.csv'])

    train_recurrent(data)

    
if(__name__ == "__main__"):
    main()
