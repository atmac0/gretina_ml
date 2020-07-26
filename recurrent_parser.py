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

    def __init__(self, filenames, max_clusters_per_file=0, for_training=True, expected_energies=None):

        if(for_training == False and expected_energies == None):
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

        print("Finished initializing dataset.")

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

        if(for_training):
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
                            if(for_training):
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
    def create_testing_label(self, cluster):

        # create a range from 0 to the number of interactions in the cluster.
        range_list = list(range(0, len(cluster)))

        # generate all possible combinations of interactions that have the interaction nearest to the origin contained within it (the interaction at the 0th
        # index of the cluster). The combination containing only the interaction of the interaction nearest to the origin will not be included, because the
        # prediction of the number of gammas a single interaction was produced by is always strongly predicted as 1.
        for i in range(1, len(cluster)):
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

    # these are for gathering calibration statistics
    correct_data = [["max probability", "final_label", "max_prob_index"]]
    incorrect_data = [["max probability", "final_label", "max_prob_index"]]
    
    
    # refactor this to just pipe in all possible combos at once into the model
    for index in range(0, len(dataset)):
        print(str(index) + '/' + str(len(dataset)))
        cluster = dataset[index]
        label = labels[index]
        
        range_list = list(range(0, num_interactions(cluster)))

        all_possible_labels = []

        all_clusters = []
        
        # generate all possible labels, then find the label that is most probable to be created by 1 gamma
        for i in range(2, num_interactions(cluster)):
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

                all_clusters.append(test_cluster)

                all_possible_labels.append(possible_label)

        all_clusters = np.asarray(all_clusters)
        try:
            predictions = model.predict(all_clusters)
        except:
            print("EXCEPTION HIT")
            print(all_clusters)
            continue

        single_gamma_probabilities = predictions[:,0].tolist()

        
        if(len(single_gamma_probabilities) == 0):
            print(single_gamma_probabilites)
            print(predicitons)
            print(all_clusters)
            exit(0)
            continue
        

        max_prob_index = single_gamma_probabilities.index(max(single_gamma_probabilities[1:]))


        final_label = all_possible_labels[max_prob_index]
        
        if(final_label == label):
            num_correct = num_correct + 1
            #correct_data.append([str(single_gamma_probabilities[max_prob_index]), str(final_label), str(max_prob_index)])
            correct_data.append(list([single_gamma_probabilities[max_prob_index], max_prob_index].append(x) for x in final_label))
        else:
            #incorrect_data.append([str(single_gamma_probabilities[max_prob_index]), str(final_label), str(max_prob_index)])
            incorrect_data.append(list([single_gamma_probabilities[max_prob_index], max_prob_index].append(x) for x in final_label))
        num_predictions = num_predictions + 1
        print("accuracy: ", num_correct/num_predictions)

    print("TOTAL NUM PREDICTIONS: ", num_predictions)
    print("TOTAL CORRECT:         ", num_correct)
    print("ACCURACY:              ", num_correct/num_predictions)

    return np.asarray(correct_data), np.asarray(incorrect_data)
    
def main():

    files = ['out_1173.csv', 'out_1332.csv', 'out_2505.csv']
    max_clusters_per_file_ = 4000
    
    #data_all = DataWrangler(files, max_clusters_per_file=max_clusters_per_file_)
    #data2505 = DataWrangler(['out_2505.csv'], expected_energies = [1173, 1332], for_training=False)
    data1173 = DataWrangler(['out_1173.csv'], expected_energies = [1173], for_training=False)
    #data1332 = DataWrangler(['out_1332.csv'], expected_energies = [1332], for_training=False)
    
    #model = get_recurrent_model(data_all)
    #model = get_classification_model(data_all)


    #model = neural_net_gamma_count.get_model()
    #model.save('recurrent_model_100_epochs')
    model = keras.models.load_model('recurrent_model_100_epochs/')

    #certainty_correct_2505, certainty_incorrect_2505 = parse_gammas(data2505, model)
    certainty_correct_1173, certainty_incorrect_1173 = parse_gammas(data1173, model)
    #certainty_correct_1332, certainty_incorrect_1332 = parse_gammas(data1332, model)
    print(certainty_correct_1173)
    #print(certainty_correct_2505)
    #print(certainty_correct_1173)
    #print(certainty_incorrect_1173)
    np.savetxt("certainty_correct_2505.csv", certainty_correct_2505, delimiter=",")
    #np.savetxt("certainty_incorrect_2505.csv", certainty_incorrect_2505, delimiter=",")

    
if(__name__ == "__main__"):
    main()
