import matplotlib.pyplot as plt
import numpy as np

from .sim import euclidean_dist


class DBC(object):

    def __init__(self, dataset, min_pts, epsilon):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon

    def snapshot(self,P_index, assignment):
        fig, ax = plt.subplots()
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], color=colors[assignment].tolist(), s=10, alpha=0.8)
        cir = plt.Circle((self.dataset[P_index][1], self.dataset[P_index][1]), self.epsilon, color="r", fill=False)
        ax.add_patch(cir)
        fig.savefig('temp.png')
        plt.close()

    def epsilon_neighborhood(self, P_index):
        neighborhood = []
        for PN in range(len(self.dataset)):
            if P_index != PN and euclidean_dist(self.dataset[PN], self.dataset[P_index]) <= self.epsilon:
                neighborhood.append(PN)

        return neighborhood

    def explore_and_assign_eps_neighborhood(self, P_index, cluster, assignments):
        neighborhood = self.epsilon_neighborhood(P_index)

        while neighborhood:
            neighbor_of_P = neighborhood.pop()
            if assignments[neighbor_of_P] != 0:
                continue
            assignments[neighbor_of_P] = cluster
            self.snapshot(P_index, assignments)

            next_neighborhood = self.epsilon_neighborhood(neighbor_of_P)
            if len(next_neighborhood) >= self.min_pts:
                # this is a core point
                # its neighbors should be explored / assigned also
                neighborhood.extend(next_neighborhood)

        return assignments

    def dbscan(self):
        """
            returns a list of assignments. The index of the
            assignment should match the index of the data point
            in the dataset.
        """

        assignments = [0 for _ in range(len(self.dataset))]
        cluster = 1

        for P in range(len(self.dataset)):
            
            if assignments[P] != 0:
                # already part of a cluster
                continue

            if len(self.epsilon_neighborhood(P)) >= self.min_pts:
                # core point
                assignments = self.explore_and_assign_eps_neighborhood(
                    P, cluster, assignments)

            cluster += 1

        return assignments
