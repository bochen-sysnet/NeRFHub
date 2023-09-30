# This file intends to transform the data generated after NeRF training
# so that it can be adapted by adjusting various knobs
# draco quantization parameter
# pruning ratio of each layer
# bit_width
# 

object_name = "chair"
prune_ratio = 0.5
qp = 11
qt = 10
qn = 8
qg = 8
cl = 7 #0-10

import json
import numpy as np
import subprocess
import glob
import os

# Define the directory path
directory_path = object_name + '_phone/'

# Use glob to search for files with .obj extension
obj_files = glob.glob(directory_path + '*.obj')

# DRACO
# -qp <value>           quantization bits for the position attribute, default=11.
# -qt <value>           quantization bits for the texture coordinate attribute, default=10.
# -qn <value>           quantization bits for the normal vector attribute, default=8.
# -qg <value>           quantization bits for any generic attribute, default=8.
# -cl <value>           compression level [0-10], most=10, least=0, default=7.
for file in obj_files:
    basename = file[:-4]
    result = subprocess.run(["draco_encoder",
                            "-i", basename + '.obj',
                            "-o", basename + '.drc',
                            "-qp", f"{qp}",
                            "-qt", f"{qt}",
                            "-qn", f"{qn}",
                            "-qg", f"{qg}",
                            "-cl", f"{cl}",
                            ], 
                            stdout=subprocess.PIPE, text=True)

    # Print the output
    print(result.stdout)
exit(0)

# prune by ratio, give warning if a whole layer is pruned
# the number of prunable layers is hard-coded to 2 in MobileNeRF
prunable_num = 2
channel_imp = [[] for _ in range(prunable_num)]
with open(object_name + '_phone/mlp.json', 'r') as file:
    data = json.load(file)

# cal importance
for obj in data:
    if isinstance(data[obj],list):
        param = np.array(data[obj])
        if obj == '1_weights':
            channel_imp[0] += np.abs(param).sum(axis=-1).tolist()
        elif obj == '2_weights':
            channel_imp[1] += np.abs(param).sum(axis=-1).tolist()

# prune channels
for i in range(prunable_num):
    ci = channel_imp[i]
    flat_imp = np.array(ci)
    sorted_imp = np.sort(flat_imp)
    thre_index = int(len(sorted_imp) * prune_ratio)
    threshold = sorted_imp[thre_index]
    in_weight_name = f'{i}_weights'
    out_weight_name = f'{i+1}_weights'
    in_bias_name = f'{i}_bias'
    data[in_weight_name] = np.array(data[in_weight_name])[:,ci>=threshold].tolist()
    data[out_weight_name] = np.array(data[out_weight_name])[ci>=threshold].tolist()
    data[in_bias_name] = np.array(data[in_bias_name])[ci>=threshold].tolist()
    # data[in_weight_name] = np.array(data[in_weight_name]).repeat(6,axis=1).tolist()
    # data[out_weight_name] = np.array(data[out_weight_name]).repeat(6,axis=0).tolist()
    # data[in_bias_name] = np.array(data[in_bias_name]).repeat(6).tolist()

# check results
for obj in data:
    if obj != 'obj_num':
        param = np.array(data[obj])
        print(obj, param.shape)

# write to json
with open(object_name + '_phone/mlp_prune.json', 'wb') as f:
  f.write(json.dumps(data).encode('utf-8'))