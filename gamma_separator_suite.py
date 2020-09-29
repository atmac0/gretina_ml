import tensorflow as tf
from tensorflow import keras

import time

import csv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from math import *

import itertools

from data_wrangler import DataWrangler
from model_trainer import get_model
from gamma_parser  import parse_gammas
from cluster_tools import *

def create_histogram_from_csv(file_name, output_name):
    lst = np.loadtxt(file_name + '.csv', delimiter='\n', dtype=np.float32)
    print(lst)
    create_histogram(lst, output_name)

def create_histogram(lst, label):

    max_plot_energy = 2800
    lst = [x for x in lst if x < max_plot_energy]
    
    file_name = label + '.png'

    n, bins, patches = plt.hist(lst, bins=max_plot_energy, facecolor='blue', alpha=0.5)

    axes = plt.gca()
    axes.set_xlim([0,max_plot_energy])
    axes.set_ylim([0,22000])
    
    plt.savefig(file_name, dpi=300)
    plt.clf()

def analyze_cluster_lists(label, cluster_lists):
    cluster_energies = []

    for cluster_list in cluster_lists:
        for cluster in cluster_list:
            cluster_energies.append(get_cluster_energy(cluster))

    with open(label + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for energy in cluster_energies:
                csvwriter.writerow([energy])
                
    create_histogram(cluster_energies, label)

# remove all interactions with multiple gammas
def clean_multiple_gammas(data, model):
    dataset, labels = data.get_dataset()
    bad_indices     = []

    total_predictions = 0
    total_correct     = 0

    print("Predicting gammas....")
    # predictions will be a 2D array, with each index giving a list of the probability of each # of gammas. the 0th index is 1 gamma, incrementing the gamma with the index. The maximum of the gamma prediction is the # of predicted gammas.
    predictions = model.predict(dataset)
    predictions = np.argmax(predictions, axis=1)

    print("Cleaning dataset....")
    for i in range(0, len(dataset)):        
        if(predictions[i] > 0):
            bad_indices.append(i)
        
        if(predictions[i] == labels[i]):
            total_correct += 1
        total_predictions += 1

    clean_dataset = np.delete(dataset, bad_indices, 0)
    accuracy = total_correct/total_predictions

    print("Uncleaned shape: ", dataset.shape)
    print("Cleaned shape:   ", clean_dataset.shape)
    print("Total removed:   ", len(bad_indices))
    
    return clean_dataset, accuracy
    

def load_model(model_name):
    model = keras.models.load_model(model_name)
    return model

                                                
def output_raw_spectrum():

    model  = load_model('model/')    
    max_clusters_per_file = 100000
    data = DataWrangler(['out_1173.csv', 'out_1332.csv', 'out_2505.csv'], max_clusters_per_file=max_clusters_per_file)

    dataset, _ = data.get_dataset()
    
    analyze_cluster_lists('raw_histo', [dataset])

def output_recovered_spectrum():
    model  = load_model('model/')    
    max_clusters_per_file = 10000
    data2505 = DataWrangler(['out_2505.csv'], expected_energies = [1173, 1332], is_training=False, max_clusters_per_file=max_clusters_per_file)
    data1173 = DataWrangler(['out_1173.csv'], expected_energies = [1173], is_training=False, max_clusters_per_file=max_clusters_per_file)
    data1332 = DataWrangler(['out_1332.csv'], expected_energies = [1332], is_training=False, max_clusters_per_file=max_clusters_per_file)
    
    recovered_2505, correct_2505, incorrect_2505, accuracy_2505 = parse_gammas(data2505, model)
    recovered_1173, correct_1173, incorrect_1173, accuracy_1173 = parse_gammas(data1173, model)
    recovered_1332, correct_1332, incorrect_1332, accuracy_1332 = parse_gammas(data1332, model)
    
    recovered_all = [recovered_2505, recovered_1173, recovered_1332]
    
    analyze_cluster_lists('recovered_all_histo', recovered_all)
    
    print("FINAL RESULTS:")
    print("2505 accuracy: ", accuracy_2505)
    print("1173 accuracy: ", accuracy_1173)
    print("1332 accuracy: ", accuracy_1332)

def output_cleaned_spectrum():
    
    model  = load_model('model/')    
    max_clusters_per_file = 100000
    data = DataWrangler(['out_1173.csv', 'out_1332.csv', 'out_2505.csv'], max_clusters_per_file=max_clusters_per_file)
 
    clean_dataset, accuracy = clean_multiple_gammas(data, model)

    analyze_cluster_lists('cleaned_all_histos', [clean_dataset])
    print("Accuracy of cleaned dataset: ", accuracy)
    
def main():

    #model = get_model(save=True)
    #output_cleaned_spectrum()
    #output_raw_spectrum()
    output_recovered_spectrum()


    
if(__name__ == "__main__"):
    main()
