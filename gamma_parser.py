from math import *
import numpy as np

import itertools

from cluster_tools import *

def predict_gamma_counts(model, dataset):
    predicted_gamma_counts = model.predict(dataset)
    
    max_indices = np.argmax(predicted_gamma_counts, axis=1)
    max_indices += 1 # tensorflow starts labels at 0, so 1 gamma == label 0, so offset by 1 to get true gamma count
    return max_indices

def parse_gammas(data, model):

    dataset, labels = data.get_dataset()

    max_interactions              = data.get_max_interactions()
    dimensionality_of_interaction = data.get_dimensionality_of_interaction()

    num_predictions = 0
    num_correct = 0

    # these are for gathering calibration statistics
    csv_header = ["max probability", "max_prob_index", "num_interactions",  "predicted_label", "correct_label"]

    recovered_gammas = []

    predicted_gamma_counts = predict_gamma_counts(model, dataset)
    
    # refactor this to just pipe in all possible combos at once into the model
    for index in range(0, len(dataset)):
        print(str(index) + '/' + str(len(dataset)))
        cluster = dataset[index]
        label = labels[index]

        predicted_gamma_count = predicted_gamma_counts[index]

        if(predicted_gamma_count == 1):
            recovered_gammas.append(cluster)
            predicted_label = tuple([j for j in range(0, get_cluster_length(cluster))])
            if(predicted_label == label):
                num_correct = num_correct + 1
                
            num_predictions = num_predictions + 1
            print('accuracy: {0}\r'.format(num_correct/num_predictions)),
        else:
        
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
            predicted_label = all_possible_labels[max_prob_index]

            # recover the interactions of the predicted gamma, and store it
            retrieved_interactions = np.asarray([cluster[i] for i in predicted_label])
            leftover_interactions  = np.delete(cluster, predicted_label, 0) # the leftover are all the interactions not in the recovered

            # create padding of zeros for retrieved and leftover
            retrieved_padding = np.zeros((max_interactions - len(retrieved_interactions), dimensionality_of_interaction))
            leftover_padding  = np.zeros((max_interactions - len(leftover_interactions), dimensionality_of_interaction))

            retrieved_interactions = np.concatenate((retrieved_interactions, retrieved_padding))
            leftover_interactions = np.concatenate((leftover_interactions, leftover_padding))

            recovered_gammas.append(retrieved_interactions)
            recovered_gammas.append(leftover_interactions)

            csv_data = [str(single_gamma_probabilities[max_prob_index]), str(max_prob_index), str(num_interactions(cluster)), str(predicted_label), str(label)]

            if(predicted_label == label):
                num_correct = num_correct + 1

            num_predictions = num_predictions + 1
            print('accuracy: {0}\r'.format(num_correct/num_predictions)),


    accuracy = num_correct/num_predictions
    print()
    print("TOTAL NUM PREDICTIONS: ", num_predictions)
    print("TOTAL CORRECT:         ", num_correct)
    print("ACCURACY:              ", accuracy)

    return np.asarray(recovered_gammas), accuracy
