
import numpy as np
class Kmeans:
    def __init__(self,data,num_clustres):
        self.data = data
        self.num_clusters = num_clusters

    def train(self,max_interations):
        #随机选择中心点
        centroids = Kmeans.centroids_init(self.data,self.num_clustres) 
        #训练
        num_examples = self.data.shape[0]
        closest_centroids_i = np.empty((num_examples,1))
        for i in range(max_intrations):
            #得到样本到中心点距离
            closest_centroids_i = Kmeans_f_closest(self.data,centroids)
            #中心更新
            centroids = Kmeans.centroids_comp(self.data,closest_centorids_i,self.num_clustres)
        return centroids,closest_centroids_i
            
    def centroids_init(self,data,num_clustres):
        num_examples = data.shape[0]
        radom_i = np.random.permutation(num_examples)
        centroids = data[random_i[:num_clusters],:]
        return centroids
    def centroids_f_closest(self,data,centroids):
        num_examples = self.data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_i = np.zeros((num_examples,1))
        for example_index in range(num_examples):
            distance = np.zeros(num_centroids,1)
            for example_index in range(num_centroids):
                distance_dif = data[example_index,:] - centroids[centroid_index,:]
                distance[centroid_index] = np.sum(distance_diff**2)
            closest_centroids_i[example_index] = np.armin(distance)
        return closest_centroids_i                   
    def centroids_comp(self,data,closest_centoids_i,num_clustres):
        centroids = np.zeros((num_clustres,num_fea))
        for centroid_i in range(num_clustres):
            closest_i = closest_centroids_i == centroid_i
            centroids[closest_i] = np.mean(data[closest_i.flatten(),:],axis=0)
            return centroids 