a = ["1\n"]*1200
b = ["2\n"]*300
c = ["3\n"]*548

txt = a+b+c

import random

random.shuffle(txt)

with open(r"C:\Users\nm1bvb\DM_hackathon\Pointnet_Pointnet2_pytorch\data\custom_partseg_data\labels\1.txt", "w") as f:
    f.writelines(txt)

