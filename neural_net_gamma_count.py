import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt

# class to handle and organize data, from the extraction from a csv to getting it into a form ready for ml training.
# data is structured in clusters, which contain a list of all interactions, and the number of gammas needed to produce those interactions.
class Data:

    def __init__(self, file_names):        

        self.cluster_list = self.make_clusters_from_csv(file_names)
        self.dataframe    = self.make_dataframe()
    
    def make_clusters_from_csv(self, file_names):

        cluster_list = []

        for file_name in file_names:
            with open(file_name, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                gamma_count = int(next(csvreader, None)[0])
                
                cluster_list.append(Cluster(gamma_count))
        
                for row in csvreader:
                    row = row[0].split(',')
                    if(row[0] == '-1'):
                        cluster_list.append(Cluster(gamma_count))
                    else:
                        interaction = Interaction(float(row[0]), float(row[1]), float(row[2]), float(row[2]))
                        cluster_list[-1].add_interaction(interaction)

        return cluster_list

    def make_dataframe(self):
        pass

    def get_cluster_list(self):
        return self.cluster_list

    def get_dataframe(self):
        return self.dataframe

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

    def get_gamma_count():
        return gamma_count

    
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
    

def main():
    file_names = ["out_1173.csv", "out_1332.csv", "out_2505.csv"]
    data = Data(file_names)
    cluster_list = data.get_cluster_list()
    
    

    

if(__name__ == "__main__"):
    main()
