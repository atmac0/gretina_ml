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

class DataWrangler:

    def __init__(self, filename):
        self.dataset = []
        self.labels  = []

        self.energies = [1173, 1332] # possible energies present, usd for generating the labels
        
        self.max_interactions = 20
        self.dimensionality_of_interaction = 5 # number of dimensions on each interaction. 5 is energy, x, y, z, distance_from_origin

        self.train_porportion = 0.8
        
        self.create_dataset(filename)

        data_len = len(self.dataset)
        
        self.dataset = np.array( self.dataset )        
        
    def create_dataset(self, filename):
    
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
                        cluster.extend([[0] * self.dimensionality_of_interaction] * (self.max_interactions - len(cluster)))
                        label = create_label(cluster)
                        if(label != None):
                            self.dataset.append(cluster)
                            self.labels.append(label)
                    else:
                        print("There were too many interactions in that cluster. Consider raising the maximum number of interactions allowed in an input.")
                    cluster = []
                else:
                    row.append(self.distance_from_origin(row[1], row[2], row[3]))
                    cluster.append(row)

    # sort a cluster by distance from the origin (i.e. interactions closer to the origin appear first in the list)
    def sort_cluster(self, cluster):
        return sorted(cluster, key=itemgetter(4)) # itemgetter of 4 selects the distance from the origin as the value to be sorted

    def create_label(cluster):

        
        
        return None
    
    def distance_from_origin(self, x, y, z):
        return sqrt( pow(x, 2) + pow(y, 2) + pow(z, 2) )

    def get_dataset(self):
        return self.dataset

    def

def train_recurrent(data):

def main():

    data = DataWrangler('out_2505.csv')

if(__name__ == "__main__"):
    main()
