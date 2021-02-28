import open3d as o3d
import numpy as np

#point_set.shape==N,3
import open3d as o3d
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


####################3
import open3d as o3d
points_ = np.transpose(points.cpu().numpy()[0,:3,:])

seg = target.cpu().numpy()[:2048]
seg__ = pred_choice[:2048]

model_pCloud0 = o3d.geometry.PointCloud()
model_pCloud0.points = o3d.utility.Vector3dVector(points_[seg==1,:])
model_pCloud0.paint_uniform_color([0.1, 0.9, 0.1])

model_pCloud1 = o3d.geometry.PointCloud()
model_pCloud1.points = o3d.utility.Vector3dVector(points_[seg==2,:])
model_pCloud1.paint_uniform_color([0.9, 0.1, 0.1])

model_pCloud2 = o3d.geometry.PointCloud()
model_pCloud2.points = o3d.utility.Vector3dVector(points_[seg==3,:])
model_pCloud2.paint_uniform_color([0.1, 0.1, 0.9])

model_pCloud3 = o3d.geometry.PointCloud()
model_pCloud3.points = o3d.utility.Vector3dVector(points_[seg__==1,:]+1)
model_pCloud3.paint_uniform_color([0.1, 0.9, 0.1])

model_pCloud4 = o3d.geometry.PointCloud()
model_pCloud4.points = o3d.utility.Vector3dVector(points_[seg__==2,:]+1)
model_pCloud4.paint_uniform_color([0.9, 0.1, 0.1])

model_pCloud5 = o3d.geometry.PointCloud()
model_pCloud5.points = o3d.utility.Vector3dVector(points_[seg__==3,:]+1)
model_pCloud5.paint_uniform_color([0.1, 0.1, 0.9])

o3d.visualization.draw_geometries([model_pCloud0,model_pCloud1,model_pCloud2,model_pCloud3,model_pCloud4,model_pCloud5])