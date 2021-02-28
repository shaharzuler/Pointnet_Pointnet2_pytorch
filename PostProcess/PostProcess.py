from sklearn.cluster import KMeans
import numpy as np

class KMeansPostProcessor:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def cluster_mobile_links(self, points, seg,):
        tot_sliders_ind = np.where(seg > 0)
        moving_points = points[tot_sliders_ind[0], :3]
        kmeans = KMeans(n_clusters=self.n_clusters).fit(moving_points)
        slider1_ind = tot_sliders_ind[0][kmeans.labels_ == 0]
        slider2_ind = tot_sliders_ind[0][kmeans.labels_ == 1]
        #
        seg[slider1_ind] = 1
        seg[slider2_ind] = 2

        return seg

#TODO consider o3d.remove_statistical_outlier for postprocess separately for each link
