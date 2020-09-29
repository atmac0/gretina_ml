def get_cluster_length(cluster):
    length = 0
    for interaction in cluster:
        if(interaction[0] == 0):
            break
        length += 1

    return length

# get num of interactions, barred the padding
def num_interactions(cluster):
    num_interactions = 0

    for interaction in cluster:
        if(interaction[0] == 0):
            return num_interactions
        num_interactions = num_interactions + 1

def get_cluster_energy(cluster):
    cluster_energy = cluster.sum(axis=0)[0]
    return cluster_energy
