import open3d as o3d
import numpy as np

#point_set.shape==N,3

model_pCloud0 = o3d.geometry.PointCloud()
model_pCloud0.points = o3d.utility.Vector3dVector(point_set[seg==0,:])
model_pCloud0.paint_uniform_color([0.1, 0.9, 0.1])

model_pCloud1 = o3d.geometry.PointCloud()
model_pCloud1.points = o3d.utility.Vector3dVector(point_set[seg==1,:])
model_pCloud1.paint_uniform_color([0.9, 0.1, 0.1])

model_pCloud2 = o3d.geometry.PointCloud()
model_pCloud2.points = o3d.utility.Vector3dVector(point_set[seg==2,:])
model_pCloud2.paint_uniform_color([0.1, 0.1, 0.9])

o3d.visualization.draw_geometries([model_pCloud0,model_pCloud1,model_pCloud2])

