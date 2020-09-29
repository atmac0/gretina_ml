import tensorflow as tf
from tensorflow import keras

import time

import csv
import numpy as np

from math import *

import itertools

from data_wrangler import DataWrangler
from model_trainer import get_model
from gamma_parser  import *
from cluster_tools import *

output_dir = 'output/'

def cluster_list_to_histo(label, cluster_list):
    cluster_energies = np.asarray([get_cluster_energy(cluster) for cluster in cluster_list])
    np.savetxt(output_dir + label + '.csv', cluster_energies, delimiter=',')
                
# remove all interactions with multiple gammas. Accuracy should match the accuracy of the model
def clean_multiple_gammas(data, model):
    dataset, labels = data.get_dataset()

    gamma_counts = predict_gamma_counts(model, dataset)

    # generate a mask to select only the clusters generated from 1 gamma
    mask = []
    for gamma_count in gamma_counts:
        if(gamma_count == 1):
            mask.append(True)
        else:
            mask.append(False)
    
    cleaned_dataset = dataset[mask]
    
    return cleaned_dataset
    

def load_model(model_name):
    model = keras.models.load_model(model_name)
    return model

                                                
def output_raw_spectrum():

    model  = load_model('model/')    
    max_clusters_per_file = 0
    data = DataWrangler(['out_1173.csv', 'out_1332.csv', 'out_2505.csv'], max_clusters_per_file=max_clusters_per_file)

    dataset, _ = data.get_dataset()
    
    cluster_list_to_histo('raw_histo', dataset)

    
def output_recovered_spectrum():
    model  = load_model('model/')    
    max_clusters_per_file = 0
    data = DataWrangler(['out_2505.csv', 'out_1173.csv', 'out_1332.csv'], expected_energies = [1173, 1332], is_training=False, max_clusters_per_file=max_clusters_per_file)
    
    recovered_gammas, accuracy = parse_gammas(data, model)
    
    cluster_list_to_histo('recovered_all_histo', recovered_gammas)
    
    print("RECOVERED GAMMAS FINAL RESULTS:")
    print("Accuracy: ", accuracy)


def output_cleaned_spectrum():
    
    model  = load_model('model/')    
    max_clusters_per_file = 0
    data = DataWrangler(['out_1173.csv', 'out_1332.csv', 'out_2505.csv'], max_clusters_per_file=max_clusters_per_file)
 
    cleaned_dataset = clean_multiple_gammas(data, model)
    cluster_list_to_histo('cleaned_all_histos', cleaned_dataset)    
    
def main():

    #model = get_model(save=True)
    output_cleaned_spectrum()
    output_raw_spectrum()
    output_recovered_spectrum()


    
if(__name__ == "__main__"):
    main()
