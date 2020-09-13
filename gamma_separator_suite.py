import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import time

import csv
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from math import *
import random

from operator import itemgetter

import itertools

class DataWrangler:

    def __init__(self, filenames, max_clusters_per_file=0, is_training=True, expected_energies=None, randomize=True, normalize=True):

        if(is_training == False and expected_energies == None):
            print("Invalid training configuration!")
            exit(0)    
        
        self.dataset = []
        self.labels  = []

        self.energies = expected_energies # possible energies present, used for generating the labels
        self.deviance = 3 # distance in keV from a measured value that is considered the same energy
        
        self.max_interactions = 20
        self.dimensionality_of_interaction = 5 # number of dimensions on each interaction. 5 is energy, x, y, z, distance_from_origin

        self.train_porportion = 0.8
        self.max_clusters_per_file = max_clusters_per_file

        if(is_training):
            self.create_dataset( filenames, is_training )

            self.dataset = np.asarray( self.dataset )
            self.labels  = np.asarray( self.labels )

            self.labels = self.labels - 1 # shift over gamma count to start at 0 so keras can understand (output layer indexing starts at 0).
            
            if(normalize):
                self.normalize_dataset()
            
            # shuffle the train and test data the same way by resetting numpy's RNG state for each shuffle
            if(randomize):
                rng_state = np.random.get_state()
                np.random.shuffle(self.dataset)
                np.random.set_state(rng_state)
                np.random.shuffle(self.labels)
            
        else:
            self.create_dataset( filenames, is_training )
            self.dataset = np.asarray( self.dataset )
            self.labels  = np.asarray( self.labels )

        data_len = len(self.dataset)
        
        print("Finished initializing dataset.")

    def normalize_dataset(self):
        self.dataset = keras.utils.normalize(self.dataset)

    # a fix sized 2d array is used to store all interactions in a cluster. This returns the width of that array
    def get_dimensionality_of_interaction(self):
        return self.dimensionality_of_interaction

    # a fix sized 2d array is used to store all interactions in a cluster. This returns the length of that array
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
    def create_dataset(self, filenames, is_training):

        if(is_training):
            print("Creating dataset and labels to be used for training the model...")
        else:
            print("Creating dataset and labels to be used for testing the model...")

        if(self.max_clusters_per_file == 0):
            print("Reading all clusters from the file(s)...")
        else:
            print("Reading " + str(self.max_clusters_per_file) + " clusters per file...")
            
            
        for filename in filenames:
            
            print("Sorting through file " + str(filename) + "...")
            
            cluster_counter = 0
            with open(filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

                # the first line of the file is assumed to be a single digit which contains the number of gammas used to produce the interactions seen in the file
                gamma_count = int(next(csvreader, None)[0])

                # the cluster starts as an empty list, with each interaction point appended
                cluster = []
        
                for row in csvreader:
                    row = row[0].split(',')
                    row = [float(i) for i in row]

                    # if a single digit of -1 is reached, the end of the cluster has been reached
                    if(row[0] == -1):

                        # check if there were more interactions than allowed per clusted. This is needed because tensorflow requires arrays of a fixed size
                        # as an input, so a hard maximum is needed for that array size.
                        if(len(cluster) <= self.max_interactions):
                            cluster = self.sort_cluster(cluster)

                            # if the dataset is for training, assign the number of gammas as the label
                            if(is_training):
                                label = gamma_count
                            # if the dataset is for testing, create a testing label
                            else:
                                label = self.create_testing_label(cluster)

                            # pad the list with empty interactions so it can be initialized to a rectangular numpy array
                            cluster.extend([[float(0)] * self.dimensionality_of_interaction] * (self.max_interactions - len(cluster))) 
                            # none in the case that no testing label could be found
                            if(label != None):
                                self.dataset.append(cluster)
                                self.labels.append(label)
                                cluster_counter = cluster_counter + 1
                                print('Sorted {0} clusters'.format(cluster_counter))
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
        sorted_cluster = sorted(cluster, key=itemgetter(4), reverse=True) # itemgetter of 4 selects the distance from the origin as the value to be sorted
        return sorted_cluster 

    # labels are going to be a list, in order, of the indices in the cluster that make up the gamma ray of the first interaction point
    def create_testing_label(self, cluster):

        # create a range from 0 to the number of interactions in the cluster.
        range_list = list(range(0, len(cluster)))

        # generate all possible combinations of interactions that have the interaction nearest to the origin contained within it (the interaction at the 0th
        # index of the cluster). The combination containing only the interaction of the interaction nearest to the origin will not be included, because the
        # prediction of the number of gammas a single interaction was produced by is always strongly predicted as 1.
        for i in range(1, len(cluster) + 1):
            possible_labels = list(itertools.combinations(range_list, i))
            
            for possible_label in possible_labels:
                # the first element of the label must be the first interaction in the cluster, so if we have reached the combinations that do not contain the 0th
                # index as the first element we can skip the rest
                if(possible_label[0] != 0):
                    break

                # sum up the energy by indexing the cluster with the indices from the label.
                energy_sum = 0
                for index in possible_label:
                    energy_sum = energy_sum + cluster[index][0]

                # if the energy is found in the expected energy list (within some boundary), return the label as the correct label
                if(self.energy_in_list(energy_sum, self.energies)):
                    return possible_label

        # if no combination of interactions produces an expected energy, return None
        return None

    # find the distance of the interaction from the origin
    def distance_from_origin(self, x, y, z):
        return sqrt( pow(x, 2) + pow(y, 2) + pow(z, 2) )

    # find if an energy is in a list, within some deviance
    def energy_in_list(self, energy, lst):
        for i in lst:
            if(self.near_value(i, energy)):
                return True
        return False

    # find if a number is within some deviance of another number
    def near_value(self, val, number):
        if( (val - self.deviance) < number < (val + self.deviance) ):
            return True
        return False


def train_model(data):

    train_inputs, train_labels, test_inputs, test_labels = data.get_training_dataset()
    print("Successfully obtained training input and labels")    

    n_hidden_layer = 16 # size of hidden layer
    n_output_layer = 2

    max_interactions = data.get_max_interactions()
    dimensionality_of_interaction = data.get_dimensionality_of_interaction()
    
    model = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(n_output_layer, return_sequences=True), input_shape=(max_interactions, dimensionality_of_interaction)),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Bidirectional( keras.layers.LSTM(n_hidden_layer, activation='tanh') ),
        keras.layers.Dense(n_output_layer, activation='softmax')
    ])    

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    

    print("Model go beep boop")
    model.fit(train_inputs, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    return model
        
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

    # these are for gathering calibration statistics
    csv_header = ["max probability", "max_prob_index", "num_interactions",  "predicted_label", "correct_label"]
    correct_data = [csv_header]
    incorrect_data = [csv_header]
    recovered_gammas = []
    
    # refactor this to just pipe in all possible combos at once into the model
    for index in range(0, len(dataset)):
        print(str(index) + '/' + str(len(dataset)))
        cluster = dataset[index]
        label = labels[index]
        
        range_list = list(range(0, num_interactions(cluster)))
        all_possible_labels = []

        all_clusters = []
        # generate all possible labels, then find the label that is most probable to be created by 1 gamma
        for i in range(1, num_interactions(cluster) + 1):
             
            possible_labels = list(itertools.combinations(range_list, i))
            # iterate through all possible labels
            for possible_label in possible_labels:

                # move on to next set of possible labels if the first element of the current label is not zero
                # since we are looking for the interactions, combined with the 0th interaction that make 1 gamma
                if(possible_label[0] != 0):
                    break

                # if there is more than one interaction point, skip the label (0, )
                if((num_interactions(cluster) > 1) and (len(possible_label) == 1)):
                    continue
                
                test_cluster = []
                for j in possible_label:
                    test_cluster.append(cluster[j].tolist())
                    
                test_cluster.extend([[0] * dimensionality_of_interaction] * (max_interactions - len(test_cluster))) # pad the list with empty interactions so it can be initialized to a rectangular numpy array

                all_clusters.append(test_cluster)

                all_possible_labels.append(possible_label)

        all_clusters = np.asarray(all_clusters)

        # each prediction is indexable by the certainty of the number of gammas needed to produce that configuration of interaction points.
        predictions = model.predict(all_clusters)
        
        single_gamma_probabilities = predictions[:,0].tolist()

        # get the index of the prediction most likely to be from a single gamma
        max_prob_index = single_gamma_probabilities.index(max(single_gamma_probabilities))

        # get the label of the combination of interactions most likely to be one gamma ray
        final_label = all_possible_labels[max_prob_index]

        # recover the interactions of the predicted gamma, and store it
        retrieved_interactions = [cluster[i] for i in final_label]
        recovered_gammas.append(retrieved_interactions)

        csv_data = [str(single_gamma_probabilities[max_prob_index]), str(max_prob_index), str(num_interactions(cluster)), str(final_label), str(label)]

        if(final_label == label):
            num_correct = num_correct + 1
            correct_data.append(csv_data)
        else:
            incorrect_data.append(csv_data)
            
        num_predictions = num_predictions + 1
        print('accuracy: {0}\r'.format(num_correct/num_predictions)),


    accuracy = num_correct/num_predictions
    print()
    print("TOTAL NUM PREDICTIONS: ", num_predictions)
    print("TOTAL CORRECT:         ", num_correct)
    print("ACCURACY:              ", accuracy)

    return recovered_gammas, np.asarray(correct_data), np.asarray(incorrect_data), accuracy


def get_cluster_energy(cluster):
    cluster_energy = 0
    for interaction in cluster:
        cluster_energy = cluster_energy + interaction[0]

    return cluster_energy

def create_histogram(lst, label):

    max_allowed = 2800
    lst = [x for x in lst if x < max_allowed]
    
    file_name = label + '.png'

    n, bins, patches = plt.hist(lst, bins=max_allowed, facecolor='blue', alpha=0.5)
    plt.savefig(file_name, dpi=300)
    plt.clf()

def analyze_cluster_lists(label, cluster_lists):
    cluster_energies = []
    
    for lst in cluster_lists:
        for cluster in lst:
            cluster_energies.append(get_cluster_energy(cluster))

    with open(label + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for energy in cluster_energies:
                csvwriter.writerow([energy])
                
    create_histogram(cluster_energies, label)
            
def get_model(save=False):

    files = ['out_1173.csv', 'out_1332.csv', 'out_2505.csv']
    max_clusters_per_file = 6000

    data = DataWrangler(files, max_clusters_per_file=max_clusters_per_file)

    model = train_model(data)

    if(save):
        model.save('model')
    
    return model

def load_model(model_name):
    model = keras.models.load_model(model_name)
    return model

def main():

    #model = get_model(save=True)
    model  = load_model('model/')
    
    max_clusters_per_file = 6000
    
    data2505 = DataWrangler(['out_2505.csv'], expected_energies = [1173, 1332], is_training=False, max_clusters_per_file=max_clusters_per_file)
    data1173 = DataWrangler(['out_1173.csv'], expected_energies = [1173], is_training=False, max_clusters_per_file=max_clusters_per_file)
    data1332 = DataWrangler(['out_1332.csv'], expected_energies = [1332], is_training=False, max_clusters_per_file=max_clusters_per_file)
    
    recovered_2505, certainty_correct_2505, certainty_incorrect_2505, accuracy_2505 = parse_gammas(data2505, model)
    recovered_1173, certainty_correct_1173, certainty_incorrect_1173, accuracy_1173 = parse_gammas(data1173, model)
    recovered_1332, certainty_correct_1332, certainty_incorrect_1332, accuracy_1332 = parse_gammas(data1332, model)

    print("FINAL RESULTS:")
    print("2505 accuracy: ", accuracy_2505)
    print("1173 accuracy: ", accuracy_1173)
    print("1332 accuracy: ", accuracy_1332)
    
if(__name__ == "__main__"):
    main()
