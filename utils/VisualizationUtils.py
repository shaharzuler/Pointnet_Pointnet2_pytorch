import gc

import numpy as np
from mayavi import mlab


class VisualizationUtils():
    def __init__(self):
        #
        self.colors = [(0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.1, 0.9, 0.1)]  # todo add more colors to support more classes
        # os.environ['ETS_TOOLKIT'] = 'qt4'

    def save_point_cloud_image(self, path, point_cloud, seg=None, target=None):
        pcds = []
        if seg is not None:
            set_ = set(target) if target is not None else set(seg)
            for i in set_:
                pcd = point_cloud[seg == i, :3]
                pcds.append(pcd)
            colors = self.colors[:len(pcds)]
        else:
            pcds.append(point_cloud)
            colors = self.colors[0]

        target_pcds = []
        if target is not None:
            for i in set(target):
                pcd = point_cloud[target == i, :3]
                target_pcds.append(pcd)

        mlab.figure(size=(300, 300))
        for pcd, color in zip(pcds, colors):
            mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], color=color, scale_factor=0.035)
        if target is not None:
            for target_pcd, color in zip(target_pcds, colors):
                mlab.points3d(target_pcd[:, 0] + 3, target_pcd[:, 1], target_pcd[:, 2], color=color, scale_factor=0.035)
        mlab.view(azimuth=90, elevation=165, roll=180)

        # mlab.show()
        if not (path.endswith(".png")):
            path += ".png"
        mlab.savefig(path, size=(300, 300))
        mlab.close()
        gc.collect()


def garbage():
    # point_set.shape==N,3
    model_pCloud0 = o3d.geometry.PointCloud()
    model_pCloud0.points = o3d.utility.Vector3dVector(point_set[seg == 0, :])
    model_pCloud0.paint_uniform_color([0.1, 0.9, 0.1])

    model_pCloud1 = o3d.geometry.PointCloud()
    model_pCloud1.points = o3d.utility.Vector3dVector(point_set[seg == 1, :])
    model_pCloud1.paint_uniform_color([0.9, 0.1, 0.1])

    model_pCloud2 = o3d.geometry.PointCloud()
    model_pCloud2.points = o3d.utility.Vector3dVector(point_set[seg == 2, :])
    model_pCloud2.paint_uniform_color([0.1, 0.1, 0.9])

    o3d.visualization.draw_geometries([model_pCloud0, model_pCloud1, model_pCloud2])

    ####################3
    import open3d as o3d
    points_ = np.transpose(points.cpu().numpy()[0, :3, :])

    seg = target.cpu().numpy()[:2048]
    seg__ = pred_choice[:2048]

    model_pCloud0 = o3d.geometry.PointCloud()
    model_pCloud0.points = o3d.utility.Vector3dVector(points_[seg == 1, :])
    model_pCloud0.paint_uniform_color([0.1, 0.9, 0.1])

    model_pCloud1 = o3d.geometry.PointCloud()
    model_pCloud1.points = o3d.utility.Vector3dVector(points_[seg == 2, :])
    model_pCloud1.paint_uniform_color([0.9, 0.1, 0.1])

    model_pCloud2 = o3d.geometry.PointCloud()
    model_pCloud2.points = o3d.utility.Vector3dVector(points_[seg == 3, :])
    model_pCloud2.paint_uniform_color([0.1, 0.1, 0.9])

    model_pCloud3 = o3d.geometry.PointCloud()
    model_pCloud3.points = o3d.utility.Vector3dVector(points_[seg__ == 1, :] + 1)
    model_pCloud3.paint_uniform_color([0.1, 0.9, 0.1])

    model_pCloud4 = o3d.geometry.PointCloud()
    model_pCloud4.points = o3d.utility.Vector3dVector(points_[seg__ == 2, :] + 1)
    model_pCloud4.paint_uniform_color([0.9, 0.1, 0.1])

    model_pCloud5 = o3d.geometry.PointCloud()
    model_pCloud5.points = o3d.utility.Vector3dVector(points_[seg__ == 3, :] + 1)
    model_pCloud5.paint_uniform_color([0.1, 0.1, 0.9])

    o3d.visualization.draw_geometries([model_pCloud0, model_pCloud1, model_pCloud2, model_pCloud3, model_pCloud4, model_pCloud5])
