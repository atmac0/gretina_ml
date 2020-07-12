import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import csv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from math import *
import random

from operator import itemgetter

import itertools

import neural_net_gamma_count

class DataWrangler:

    def __init__(self, filenames, max_clusters_per_file=0, for_training=True):
        self.dataset = []
        self.labels  = []

        self.energies = [1173, 1332] # possible energies present, used for generating the labels
        self.deviance = 3 # distance in keV from a measured value that is considered the same energy
        
        self.max_interactions = 20
        self.dimensionality_of_interaction = 5 # number of dimensions on each interaction. 5 is energy, x, y, z, distance_from_origin

        self.train_porportion = 0.8
        self.max_clusters_per_file = max_clusters_per_file


        if(for_training):
            self.create_dataset( filenames, for_training )

            self.dataset = np.asarray( self.dataset )
            self.labels  = np.asarray( self.labels )
            
            # shuffle the train and test data the same way by resetting numpy's RNG state for each shuffle
            rng_state = np.random.get_state()
            np.random.shuffle(self.dataset)

            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)
            
        else:
            self.create_dataset( filenames, for_training )
            self.dataset = np.asarray( self.dataset )
            self.labels  = np.asarray( self.labels )


        #self.normalize_dataset()

        data_len = len(self.dataset)

    def normalize_dataset(self):
        pass
        
    def get_dimensionality_of_interaction(self):
        return self.dimensionality_of_interaction

    def get_max_interactions(self):
        return self.max_interactions
    
    def get_training_dataset(self):
        num_train = int(len(self.labels) * self.train_porportion)

        train_input   = self.dataset[:num_train]
        train_labels = self.labels[:num_train]

        test_input   = self.dataset[num_train:]
        test_labels = self.labels[num_train:]

        return train_input, train_labels, test_input, test_labels

    def get_dataset(self):
        return self.dataset, self.labels

    # create a 3d numpy array with the dimension cluster, interaction, dimension of interaction
    # a cluster is a set of related interaction points
    # an interaction is a measured event in the detector
    # the dimension of interaction is at a minimum energy, x, y, and z coordinates. Other dimensions may be calculated and added on (like total distance from origin). 
    def create_dataset(self, filenames, for_training):
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
                        if(len(cluster) <= self.max_interactions):
                            cluster = self.sort_cluster(cluster)
                            
                            if(for_training == False):
                                label = self.create_label(cluster)
                            else:
                                if(gamma_count == 1):
                                    label = gamma_count
                                elif(self.create_label(cluster)):
                                    label = gamma_count
                                else:
                                    label = None
                                

                            cluster.extend([[float(0)] * self.dimensionality_of_interaction] * (self.max_interactions - len(cluster))) # pad the list with empty interactions so it can be initialized to a rectangular numpy array

                            if(label != None):
                                self.dataset.append(cluster)
                                self.labels.append(label)
                                cluster_counter = cluster_counter + 1
                        else:
                            print("There were too many interactions in that cluster. Consider raising the maximum number of interactions allowed in an input.")
                        cluster = []

                        # go to next file if max clusters read from current file
                        if((self.max_clusters_per_file != 0) and (cluster_counter >= self.max_clusters_per_file)):
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

    def energy_in_list(self, energy, lst):
        for i in lst:
            if(self.near_value(i, energy)):
                return True
        return False

    def near_value(self, val, number):
        if( (val - self.deviance) < number < (val + self.deviance) ):
            return True
        return False
                   
def get_recurrent_model(data):

    train_inputs, train_labels, test_inputs, test_labels = data.get_training_dataset()
    print("Successfully obtained training input and labels")    

    n_hidden_layer = 32 # size of hidden layer
    n_output_layer = 2

    max_interactions = data.get_max_interactions()
    dimensionality_of_interaction = data.get_dimensionality_of_interaction()
    
    model = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(n_output_layer, return_sequences=True), input_shape=(max_interactions, dimensionality_of_interaction)),
        keras.layers.Bidirectional( keras.layers.LSTM(n_hidden_layer) ),
        keras.layers.Dense(2)
    ])    

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    

    print("Model go beep boop")
    model.fit(train_inputs, train_labels, epochs=2)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)


    return model
        

# use interactions of cluster as 1d array. Appends 0's until the array reaches cluster size. Put into a simple neural network and train.
def get_classification_model(data):

    max_interactions              = data.get_max_interactions()
    dimensionality_of_interaction = data.get_dimensionality_of_interaction()

    train_inputs, train_labels, test_inputs, test_labels = data.get_training_dataset()





    assert not np.any(np.isnan(train_inputs))
    exit(0)
    

    train_shape=train_inputs.shape
    test_shape=test_inputs.shape
    temp_train_inputs = np.empty([train_shape[0], train_shape[1], train_shape[2]-1])
    temp_test_inputs  = np.empty([test_shape[0], test_shape[1], test_shape[2]-1])


    for i in range(0, train_shape[0]):
        for j in range(0, 20):
            for k in range(0, 4):
                temp_train_inputs[i][j][k] = train_inputs[i][j][k]

    for i in range(0, test_shape[0]):
        for j in range(0, 20):
            for k in range(0, 4):
                temp_test_inputs[i][j][k] = test_inputs[i][j][k]


    #train_inputs = temp_train_inputs
    #test_inputs = temp_test_inputs
    


    
    print("Successfully obtained training input and labels")    
    
    n_hidden_layer = 128 # size of hidden layer
    n_output_layer = 2

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(max_interactions, dimensionality_of_interaction-1)),
        keras.layers.Dense(n_hidden_layer, activation='relu'),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    print("Model go beep boop")
    model.fit(train_inputs, train_labels, epochs=20)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    exit(0)

        
# get num of interactions, barred the padding
def num_interactions(cluster):
    num_interactions = 0

    for interaction in cluster:
        if(interaction[0] == 0):
            return num_interactions
        num_interactions = num_interactions + 1
        
def parse_gammas(data, model):

    dataset, labels = data.get_dataset()
    
    max_interactions              = data.get_max_interactions()
    dimensionality_of_interaction = data.get_dimensionality_of_interaction()

    num_predictions = 0
    num_correct = 0
    
    # refactor this to just pipe in all possible combos at once into the model
    for index in range(0, len(dataset)):
        print(str(index) + '/' + str(len(dataset)))
        cluster = dataset[index]
        label = labels[index]
        
        range_list = list(range(0, num_interactions(cluster)))

        single_gamma_probability = []

        all_possible_labels = []
        
        # generate all possible labels, then find the label that is most probable to be created by 1 gamma
        for i in range(1, num_interactions(cluster)):
            possible_labels = list(itertools.combinations(range_list, i))

            # iterate through all possible labels
            for possible_label in possible_labels:

                # move on to next set of possible labels if the first element of the current label is not zero
                # since we are looking for the interactions, combined with the 0th interaction that make 1 gamma
                if(possible_label[0] != 0):
                    break

                test_cluster = []
                for j in possible_label:
                    test_cluster.append(cluster[j].tolist())
                    
                test_cluster.extend([[0] * dimensionality_of_interaction] * (max_interactions - len(test_cluster))) # pad the list with empty interactions so it can be initialized to a rectangular numpy array

                test_cluster = [test_cluster] # make it 3D
                test_cluster = np.array(test_cluster)
                
                x = model.predict(test_cluster)
                single_gamma_probability.append(x[0][0])
                all_possible_labels.append(possible_label)

        if(len(single_gamma_probability) >= 2):
            max_prob_index = single_gamma_probability.index(max(single_gamma_probability[1:]))
        elif(len(single_gamma_probability) >= 1):
            max_prob_index = single_gamma_probability.index(max(single_gamma_probability))
        else:
            continue

        #print("MAX PROB INDEX: ", max_prob_index)
        #print("LEN POSSIBLE LABELS: ", len(possible_labels))
        #print("LEN probs: ", len(single_gamma_probability))
        #print("PROBS: ", single_gamma_probability)
        #exit(0)
        #print(all_possible_labels)
        #print(len(all_possible_labels))
        #print(max_prob_index)
        final_label = all_possible_labels[max_prob_index]

        #print("LABEL:       ", label)
        #print("FINAL LABEL: ", final_label)
        #print("max prob:    ", single_gamma_probability[max_prob_index])
        #exit(0)
        if(final_label == label):
            num_correct = num_correct + 1
        num_predictions = num_predictions + 1
        print("accuracy: ", num_correct/num_predictions)

    print("TOTAL NUM PREDICTIONS: ", num_predictions)
    print("TOTAL CORRECT:         ", num_correct)
    print("ACCURACY:              ", num_correct/num_predictions)
    
def main():

    files = ['out_1173.csv', 'out_1332.csv', 'out_2505.csv']
    max_clusters_per_file_ = 4000
    
    #data_all = DataWrangler(files, max_clusters_per_file=max_clusters_per_file_)

    data2 = DataWrangler(['out_2505.csv'], for_training=False)
    
    #model = get_recurrent_model(data_all)
    #model = get_classification_model(data_all)


    #model = neural_net_gamma_count.get_model()
    #model.save('recurrent_model_100_epochs')
    model = keras.models.load_model('recurrent_model_100_epochs/')

    parse_gammas(data2, model)

    
if(__name__ == "__main__"):
    main()
