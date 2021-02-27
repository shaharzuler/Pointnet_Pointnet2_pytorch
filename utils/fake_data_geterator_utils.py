a = ["1\n"]*700
b = ["2\n"]*700
c = ["3\n"]*700

txt = a+b+c

import random

# random.shuffle(txt)

with open(r"/home/fiman/projects/DMHackathon/Pointnet_Pointnet2_pytorch/data/labels/1.txt", "w") as f:
    f.writelines(txt)

