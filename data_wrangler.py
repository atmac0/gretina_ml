import numpy as np
import random
from math import *
from operator import itemgetter
import itertools
import csv

class DataWrangler:

    def __init__(self, filenames, max_clusters_per_file=0, is_training=True, expected_energies=None, randomize=True, normalize=False):

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
