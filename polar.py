import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt

class Cluster:
    
    def __init__(self):
        self.interactions = []

    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def get_interactions(self):
        return self.interactions
        
class Interaction:
    def __init__(self, energy, x, y, z):

        self.energy = energy

        self.x = x
        self.y = y
        self.z = z

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
        
def make_clusters_from_csv(file_name):

    cluster_list = []
    
    with open(file_name, newline='') as csvfile:

        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        cluster_list.append(Cluster())
        
        for row in csvreader:
            row = row[0].split(',')
            if(row[0] == '-1'):
                cluster_list.append(Cluster())
            else:
                interaction = Interaction(float(row[0]), float(row[1]), float(row[2]), float(row[2]))
                cluster_list[-1].add_interaction(interaction)

    return cluster_list


def main():
    
    cluster_list = make_clusters_from_csv('out_1173.csv')

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

    n_energy_bins = 1000
    n_theta_bins  = 200
    
    energy_bins = np.linspace(0, np.max(energy_list), n_energy_bins + 1)
    theta_bins  = np.linspace(0, 2*np.pi, n_theta_bins + 1)
    
    H, _, _ = np.histogram2d(energy_list, theta_list, [energy_bins, theta_bins])
                    
    ax = plt.subplot(111, polar=True)
    Theta, R = np.meshgrid(theta_bins, energy_bins)
    ax.pcolormesh(Theta, R, H)
    ax.set_title("1173 relative interaction angles")
                    
    plt.savefig('1173_polar.png')


if(__name__ == "__main__"):
    main()
