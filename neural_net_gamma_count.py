import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans

import csv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from math import *
import random

import recurrent_parser

# randomly select if something will be for testing the model (the negative being for training)
def is_test(probability):
    return random.random() > probability

# class to handle and organize data, from the extraction from a csv to getting it into a form ready for ml training.
# data is structured in clusters, which contain a list of all interactions, and the number of gammas needed to produce those interactions.
class Data:

    def __init__(self, file_names, file_read_limit, test_porportion):        

        self.file_read_limit = file_read_limit
        
        self.train_clusters = []
        self.test_clusters = []

        self.make_train_and_test_clusters_from_csv(file_names, test_porportion)
        
        self.cluster_list = self.make_clusters_from_csv(file_names)


    def make_train_and_test_clusters_from_csv(self, file_names, test_porportion):

        cluster_list = []

        for file_name in file_names:
            with open(file_name, newline='') as csvfile:
                
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                gamma_count = int(next(csvreader, None)[0])
                self.train_clusters.append(Cluster(gamma_count))

                active_on_train_list = True # tells if current cluster being populated is on the train list (the negative being for the test list)
                cluster_counter = 1
        
                for row in csvreader:
                    row = row[0].split(',')
                    if(row[0] == '-1'):
                        if(cluster_counter < self.file_read_limit):

                            if(is_test(test_porportion)):
                                self.test_clusters.append(Cluster(gamma_count))
                                active_on_train_list = False
                            else:
                                self.train_clusters.append(Cluster(gamma_count))
                                active_on_train_list = True
                            
                            cluster_counter = cluster_counter + 1
                        else:
                            break
                    else:
                        interaction = Interaction(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                        if(active_on_train_list):
                            self.train_clusters[-1].add_interaction(interaction)
                        else:
                            self.test_clusters[-1].add_interaction(interaction)

        
    def make_clusters_from_csv(self, file_names):

        cluster_list = []

        for file_name in file_names:
            with open(file_name, newline='') as csvfile:
                
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                gamma_count = int(next(csvreader, None)[0])
                cluster_list.append(Cluster(gamma_count))

                cluster_counter = 1
        
                for row in csvreader:
                    row = row[0].split(',')
                    if(row[0] == '-1'):
                        if( (self.file_read_limit == 0) or (cluster_counter < self.file_read_limit) ):

                            cluster_list.append(Cluster(gamma_count))
                            cluster_counter = cluster_counter + 1
                        else:
                            break
                    else:
                        interaction = Interaction(float(row[0]), float(row[1]), float(row[2]), float(row[2]))
                        cluster_list[-1].add_interaction(interaction)

        return cluster_list

    def make_dataframe(self):
        pass

    def get_cluster_list(self):
        return self.cluster_list

    def get_train_clusters(self):
        return self.train_clusters

    def get_test_clusters(self):
        return self.test_clusters

    def get_cluster_count(self):
        return len(self.cluster_list)

    def get_train_count(self):
        return len(self.train_clusters)

    def get_test_count(self):
        return len(self.test_clusters)
    
    def get_dataframe(self):
        return self.dataframe


    # normalize all data points to be between 0 and 1
    def normalize_train_test(self):

        #assign initial values to values of first cluster
        max_energy  = self.train_clusters[0].get_interactions()[0].get_energy()
        min_spatial = self.train_clusters[0].get_interactions()[0].get_x()
        max_spatial = self.train_clusters[0].get_interactions()[0].get_x()

        # iterate through all interactions in each cluster to find the maximum energy, and the min and max spatial coordinate
        for cluster in self.train_clusters:
            for interaction in cluster.get_interactions():
                if(interaction.get_energy() > max_energy):
                    max_energy = interaction.get_energy()

                for coord in interaction.get_spatial():
                    if(coord < min_spatial):
                        min_spatial = coord
                    if(coord > max_spatial):
                        max_spatial = coord

        # iterate through all interactions in each cluster to find the maximum energy, and the min and max spatial coordinate
        for cluster in self.test_clusters:
            for interaction in cluster.get_interactions():
                if(interaction.get_energy() > max_energy):
                    max_energy = interaction.get_energy()

                for coord in interaction.get_spatial():
                    if(coord < min_spatial):
                        min_spatial = coord
                    if(coord > max_spatial):
                        max_spatial = coord

        spatial_offset        = abs(min_spatial)
        spatial_normalization = max_spatial + spatial_offset # value to turn any offset spatial coord to between 0 and 1
        
        for cluster in self.train_clusters:
            cluster.normalize(max_energy, spatial_normalization, spatial_offset)

        for cluster in self.test_clusters:
            cluster.normalize(max_energy, spatial_normalization, spatial_offset)
            
        print("Successfully normalized the train and test data")
    
    # normalize all data points to be between 0 and 1
    def normalize(self):

        #assign initial values to values of first cluster
        max_energy  = self.cluster_list[0].get_interactions()[0].get_energy()
        min_spatial = self.cluster_list[0].get_interactions()[0].get_x()
        max_spatial = self.cluster_list[0].get_interactions()[0].get_x()

        # iterate through all interactions in each cluster to find the maximum energy, and the min and max spatial coordinate
        for cluster in self.cluster_list:
            for interaction in cluster.get_interactions():
                if(interaction.get_energy() > max_energy):
                    max_energy = interaction.get_energy()

                for coord in interaction.get_spatial():
                    if(coord < min_spatial):
                        min_spatial = coord
                    if(coord > max_spatial):
                        max_spatial = coord

        spatial_offset        = abs(min_spatial)
        spatial_normalization = max_spatial + spatial_offset # value to turn any offset spatial coord to between 0 and 1
        
        for cluster in self.cluster_list:
            cluster.normalize(max_energy, spatial_normalization, spatial_offset)

        print("Successfully normalized the data")
        
# a cluster is an object that contains related interaction points. All the points in a cluster were near in time in the detector.
# a known number of gammas needed to produce the total energy deposited in the cluster is stored in the cluster object. This is known
# from the dataset, where the energies observed were from a known source. Since the decay scheme of the source is known, certain enrgies
# are only achievable by the combination of multiple gammas.
class Cluster:

    def __init__(self, gamma_count):
        self.interactions = []
        self.gamma_count = gamma_count
        
    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def get_interactions(self):
        return self.interactions

    def get_gamma_count(self):
        return self.gamma_count

    def normalize(self, energy_normalization, spatial_normalization, spatial_offset):
        for interaction in self.interactions:
            interaction.normalize(energy_normalization, spatial_normalization, spatial_offset)

    
class Interaction:
    def __init__(self, energy, x, y, z):

        self.energy = energy

        self.x = x
        self.y = y
        self.z = z

        self.r = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))

    # find the distnce between interaction points
    def get_distance_from_other(self, interaction):

        x_diff = interaction.x - self.x
        y_diff = interaction.y - self.y
        z_diff = interaction.z - self.z

        distance = sqrt( pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2) )
        return distance

    def get_distance_from_origin(self):
        
        distance = sqrt( pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2) )
        return distance        

    def get_energy(self):
        return self.energy

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
    
    def get_z(self):
        return self.z

    # returns a list of spatial coordinates, in order [x, y, z]
    def get_spatial(self):
        return [self.x, self.y, self.z]

    # returns all data in a list, in order [energy, x, y, z]    
    def get_all(self):
        return [self.energy, self.x, self.y, self.z, self.r]    
    # find the angle between two interaction points. The angle is relative to a line drawn from the origin
    # to the interaction point containing this function.
    def find_theta(self, interaction):
        #     a: distance from origin to self
        #     b: distance from origin to other interaction
        #     c: distance between self and other interaction
        #   phi: angle opposite of b, or the angle between line a and c
        # theta: supplementary angle to phi
        a = self.get_distance_from_origin()
        b = interaction.get_distance_from_origin()
        c = self.get_distance_from_other(interaction)

        # calculated using the angle version of law of cosines
        phi = acos( (pow(a, 2) + pow(c, 2) - pow(b, 2)) / (2 * a * c))
        theta = pi - phi

        return theta

    def normalize(self, energy_normalization, spatial_normalization, spatial_offset):

        self.energy = self.energy/energy_normalization
        self.x      = (self.x + spatial_offset)/spatial_normalization
        self.y      = (self.y + spatial_offset)/spatial_normalization
        self.z      = (self.z + spatial_offset)/spatial_normalization
        
        if(self.energy < 0 or self.energy > 1):
            print("Energy normalization constant too small!")
            exit(1)
        if(self.x < 0 or self.y < 0 or self.z < 0):
            print("Spatial offset too small! Spatial location below zero!")
            exit(2)
        if(self.x > 1 or self.y > 1 or self.z > 1):
            print("Spatial constant too small! Spatial location above one!")
            exit(3)

# plot in a polar representation, all points of all clusters, where the distance (r) is the enregy of the point, and the angle is the relative angle to every other point.
# for example, a cluster has 5 interaction points. The energy and angle of the last 4 relative to the first are all plotted. Iterate through the other points, plotting the same (e.g. 2nd point
# as reference, plot all other points in cluster).
def make_polar(cluster_list):

    theta_list  = []
    energy_list = []
    
    
    for cluster in cluster_list:

        interactions = cluster.get_interactions()
        
        for i in interactions:
            for j in interactions:
                if(i is j):
                    pass
                else:
                    theta_list.append(i.find_theta(j))
                    energy_list.append(i.get_energy())

    print("Number of clusters: ", len(cluster_list))

    theta_list = np.asarray(theta_list)
    energy_list = np.asarray(energy_list)

    n_energy_bins = 6000
    n_theta_bins  = 360
    
    energy_bins = np.linspace(0, 3000, n_energy_bins + 1)
    theta_bins  = np.linspace(0, 2*np.pi, n_theta_bins + 1)
    
    H, _, _ = np.histogram2d(energy_list, theta_list, [energy_bins, theta_bins])
                    
    ax = plt.subplot(111, polar=True)
    Theta, R = np.meshgrid(theta_bins, energy_bins)
    ax.pcolormesh(Theta, R, H)
    ax.set_title("1173 relative interaction angles")
                    
    plt.savefig('1173_polar.png')
    

# flatten each cluster into 
def train_recursive(cluster_list):
    pass


def transform_cluster_list_into_3d(cluster_list, dimensionality_of_interaction, max_interactions):

    inputs = np.zeros([len(cluster_list), max_interactions, dimensionality_of_interaction])
    labels = np.zeros(len(cluster_list), dtype='i4')
    
    for i in range(0, len(cluster_list)):
        cluster = cluster_list[i]
        
        if(len(cluster.get_interactions() ) <= max_interactions):
            
            input_data = np.zeros([max_interactions, dimensionality_of_interaction])

            interactions = cluster.get_interactions()
            
            for j in range(0, len(interactions)):
                interaction = np.asarray(interactions[j].get_all())
                input_data = np.insert(input_data, j, interaction, axis=0)

            input_data.resize([max_interactions, dimensionality_of_interaction]) # resize cause I'm stupid and can't figure out how to assign a 1d inside a 2d array instead of inserting more rows
            inputs[i] = input_data
            labels[i] = cluster.get_gamma_count()
        else:
            print("Input size not large enough to contain all interaction points. Increase total allowed interactions per input set")
            exit(1)

    labels = labels - 1 # shift over gamma count to start at 0 so keras can understand (output layer indexing starts at 0).

    return inputs, labels

# create a 3d array from the cluster list, but each input layer is sorted by distance from the origin
def transform_cluter_list_into_sorted_3d(cluster_list, dimensionality_of_interaction, max_interactions):

    inputs = np.zeros([len(cluster_list), max_interactions, dimensionality_of_interaction])
    labels = np.zeros(len(cluster_list), dtype='i4')
    
    for i in range(0, len(cluster_list)):
        cluster = cluster_list[i]
        
        if(len(cluster.get_interactions() ) <= max_interactions):
            
            input_data = np.zeros([max_interactions, dimensionality_of_interaction])

            interactions = cluster.get_interactions()
            
            for j in range(0, len(interactions)):
                interaction = np.asarray(interactions[j].get_all())
                input_data = np.insert(input_data, j, interaction, axis=0)

            input_data.resize([max_interactions, dimensionality_of_interaction]) # resize cause I'm stupid and can't figure out how to assign a 1d inside a 2d array instead of inserting more rows
            inputs[i] = input_data
            labels[i] = cluster.get_gamma_count()
        else:
            print("Input size not large enough to contain all interaction points. Increase total allowed interactions per input set")
            exit(1)

    labels = labels - 1 # shift over gamma count to start at 0 so keras can understand (output layer indexing starts at 0).
            
    return inputs, labels

# use interactions of cluster as 1d array. Appends 0's until the array reaches cluster size. Put into a simple neural network and train.
def train_fixed_input(data):

    max_interactions = 20 # fixed size of input neurons, representing a cluster. It is expected that the total number of interactions in a cluster is less than this number
    dimensionality_of_interaction = 5 # number of dimensions in an interaction. 4 would represent x, y, z, energy
    
    # pull raw data out from clusters and interactions into a 2d array. Each row of the input array will be each a 1D list of every interaction, cycling in order of energy, x, y, z.

    train_list = data.get_train_clusters()
    test_list  = data.get_test_clusters()

    train_inputs, train_labels = transform_cluster_list_into_3d(train_list, dimensionality_of_interaction, max_interactions)
    test_inputs, test_labels = transform_cluster_list_into_3d(test_list, dimensionality_of_interaction, max_interactions)
    
            
    print("Successfully created training input and labels")    
    
    n_hidden_layer = 128 # size of hidden layer
    n_output_layer = 2

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(max_interactions, dimensionality_of_interaction)),
        keras.layers.Dense(n_hidden_layer, activation='relu'),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    print("Model go beep boop")
    model.fit(train_inputs, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    
    return model
    

def train_full_array_input(cluster_list):
    pass


def near_value(val, deviance, number):
    if( (val - deviance) < number < (val + deviance) ):
        return True
    return False

def k_means_clustering(data):


    cluster_counter = 0
    counter_1332 = 0
    counter_1173 = 0
    counter_both = 0
    
    cluster_list = data.get_cluster_list()
    
    for cluster in cluster_list:

        interactions = cluster.get_interactions()
        
        interaction_coords   = np.array([interaction.get_spatial() for interaction in interactions])
        interaction_energies = np.array([interaction.get_energy() for interaction in interactions])

        if(len(interaction_coords) <= 1):
            continue
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(interaction_coords)
        
        sum_zero_energy = 0
        sum_one_energy  = 0
        
        for i in range(0, len(interaction_coords)):
            if(kmeans.labels_[i] == 0):
                sum_zero_energy = sum_zero_energy + interaction_energies[i]
            elif(kmeans.labels_[i] == 1):
                sum_one_energy = sum_one_energy + interaction_energies[i]
            else:
                print("something went wrong...?")
            
        cluster_counter = cluster_counter + 1
        
        if(near_value(1332, 5, sum_zero_energy) or near_value(1332, 5, sum_one_energy)):
            counter_1332 = counter_1332 + 1
            
        if(near_value(1173, 5, sum_zero_energy) or near_value(1173, 5, sum_one_energy)):
            counter_1173 = counter_1173 + 1

        if(near_value(1173, 5, sum_zero_energy) and near_value(1332, 5, sum_one_energy)):
            counter_both = counter_both + 1

        if(near_value(1332, 5, sum_zero_energy) and near_value(1173, 5, sum_one_energy)):
            counter_both = counter_both + 1
            

    print("TOTAL CLUSTERS     : ", cluster_counter)
    print("TOTAL 1173 CLUSTERS: ", counter_1173)
    print("TOTAL 1332 CLUSTERS: ", counter_1332)
    print("TOTAL BOTH CLUSTERS: ", counter_both)


def train_recurrent(data):

    max_interactions = 20 # fixed size of input neurons, representing a cluster. It is expected that the total number of interactions in a cluster is less than this number
    dimensionality_of_interaction = 5 # number of dimensions in an interaction. 4 would represent x, y, z, energy
    
    # pull raw data out from clusters and interactions into a 2d array. Each row of the input array will be each a 1D list of every interaction, cycling in order of energy, x, y, z.

    train_list = data.get_train_clusters()
    test_list  = data.get_test_clusters()

    train_inputs, train_labels = transform_cluster_list_into_3d(train_list, dimensionality_of_interaction, max_interactions)
    test_inputs, test_labels = transform_cluster_list_into_3d(test_list, dimensionality_of_interaction, max_interactions)
            
    print("Successfully created training input and labels")
    

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


    return model


def get_model():
    file_names = ["out_1173.csv", "out_1332.csv", "out_2505.csv"]
    file_read_limit = 10000 # max number of clusters to read from a file. Zero is no limit
    test_porportion = 0.2 # porportion of the data used for testing
    
    data = Data(file_names, file_read_limit, test_porportion)

    print("Total clusters read from files: ", data.get_cluster_count())
    print("Train clusters read from files: ", data.get_train_count())
    print("Test clusters read from files : ", data.get_test_count())

    
    
    #data.normalize_train_test()
    
    # clusters are used as the input to the neural net. A cluster contains a collection of interaction points, along with the total number of gammas needed
    # to produce the total energy seen from the cluster. 


    
    #k_means_clustering(data)
    
    return train_fixed_input(data)    
    #return train_recurrent(data)    

def main():
    
    file_names = ["out_1173.csv", "out_1332.csv", "out_2505.csv"]
    file_read_limit = 10000 # max number of clusters to read from a file. Zero is no limit
    test_porportion = 0.2 # porportion of the data used for testing
    
    data = Data(file_names, file_read_limit, test_porportion)

    print("Total clusters read from files: ", data.get_cluster_count())
    print("Train clusters read from files: ", data.get_train_count())
    print("Test clusters read from files : ", data.get_test_count())

    
    
    #data.normalize_train_test()
    
    # clusters are used as the input to the neural net. A cluster contains a collection of interaction points, along with the total number of gammas needed
    # to produce the total energy seen from the cluster. 

    #train_fixed_input(data)    
    
    #k_means_clustering(data)

    #model = train_recurrent(data)
    
    

    

if(__name__ == "__main__"):
    main()
