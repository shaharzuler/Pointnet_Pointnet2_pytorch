a = ["1\n"]*700
b = ["2\n"]*700
c = ["3\n"]*700

txt = a+b+c

import random

# random.shuffle(txt)

with open(r"/home/fiman/projects/DMHackathon/Pointnet_Pointnet2_pytorch/data/labels/1.txt", "w") as f:
    f.writelines(txt)


# recoloring for kmeans:
# import numpy as np
# col=0
# ind1 = np.where(point_set[:,col]>0.5 )[0]
# ind2 = np.where(point_set[:,col]<-0.5 )[0]
#
# seg[ind1] = 4
# seg[ind2] = 5
#
# seg[seg<4] = 0
# seg[seg==4] = 1
# seg[seg==5] = 2
# np.savetxt("data/custom_partseg_data/labels/1.txt", (seg+1).astype(np.int),fmt='%i')

