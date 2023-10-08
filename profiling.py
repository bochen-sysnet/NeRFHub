import requests
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import requests
import socket
import pickle
import random
import math
import json
import subprocess

# run as a thread
# share a pipe of received data
# every mlp has one profile
# every mlp will have a separate problem to make searching fast
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=7, n_obj=2, n_constr=0, 
                         xl=[1,0,0,0,0,0,0], xu=[8,10,9,4,14,14,10], vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # send config to html for test

        # retrieve result and return
        out["F"] = - np.min(x * [3, 1], axis=1)

default_config = [8,10,9,4,0,0,0,96]
url = 'http://130.126.139.208:8000/send_message'
num_samples = 3

def prune(prune_chan, object_name):
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
        flat_imp = np.array(channel_imp[i])
        sorted_imp = np.sort(flat_imp)
        if prune_chan > len(sorted_imp):
            prune_chan = len(sorted_imp)
        threshold = sorted_imp[-prune_chan]
        in_weight_name = f'{i}_weights'
        out_weight_name = f'{i+1}_weights'
        in_bias_name = f'{i}_bias'
        data[in_weight_name] = np.array(data[in_weight_name])[:,channel_imp[i]>=threshold].tolist()
        data[out_weight_name] = np.array(data[out_weight_name])[channel_imp[i]>=threshold].tolist()
        data[in_bias_name] = np.array(data[in_bias_name])[channel_imp[i]>=threshold].tolist()

    # write to json
    with open(object_name + f'_phone/mlp_p.json', 'wb') as f:
        f.write(json.dumps(data).encode('utf-8'))

def draco(qp,qt,cl,object_name):
    # Use glob to search for files with .obj extension
    obj_files = glob.glob(object_name + f'_phone/' + '*.obj')

    # DRACO
    for file in obj_files:
        basename = file[:-4]
        result = subprocess.run(["draco_encoder",
                                "-i", basename + '.obj',
                                "-o", basename + '.drc',
                                "-qp", f"{qp}",
                                "-qt", f"{qt}",
                                "-cl", f"{cl}",
                                ], 
                                stdout=subprocess.PIPE, text=True)

        # # Print the output
        # print(result.stdout)

def png(d,f,l,s,object_name):
    # texture compression
    pattern = object_name + '_phone/shape[0-9].pngfeat[0-9].png'
    png_files = glob.glob(pattern)
    for file in png_files:
        basename = file[:-4]
        # the problem is the 1/17 pad seems to create a lot of white tone
        subprocess.run(["convert",
                        basename + '.png',
                        '-depth', f'{d}',
                        '-define', f'png:compression-filter={f}',
                        '-define', f'png:compression-level={l}',
                        '-define', f'png:compression-strategy={s}',
                        basename + '.x.png',
                        ], 
                        stdout=subprocess.PIPE, text=True)

def eval_one_config(config, object_name, prefix, azimuthal_angle, polar_angle):
    d,f,l,s,qp,qt,cl,mlp_channel = config

    # mlp prune
    prune(mlp_channel,object_name)

    # mesh compression
    draco(qp,qt,cl,object_name)

    # texture compression
    png(d,f,l,s,object_name)

    # calculate file size
    size = os.path.getsize(object_name + '_phone/mlp_p.json')

    for file in glob.glob(object_name + '_phone/shape[0-9].pngfeat[0-9].x.png'):
        size += os.path.getsize (file)

    for file in glob.glob(object_name + f'_phone/' + '*.drc'):
        size += os.path.getsize (file)

    # send to html for snapshots
    data = {'mlp':mlp_channel,
            'scene_option':object_name,
            'azimuthal_angle':azimuthal_angle,
            'polar_angle':polar_angle,
            'num_samples':num_samples,
            'prefix':prefix
            }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Request evaluation.")
    else:
        print(f"Error: {response.status_code}")

    # Set up a socket to receive data from the Flask server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 8001))  # Use the same IP and port as in the Flask app
        s.listen()
        conn, addr = s.accept()

        with conn:
            data = conn.recv(1024)
            received_data = pickle.loads(data)
            print(received_data)

    return size

def profiling():
    # the goal is to find similar quality, real-time and low latency config

    # use xxxxx96 at max channel and lossless compression as gt
    # compare model with less channels or default model to it
    # should handle the case with exact same config as gt
    # two types of objects
    # 'chair': original 
    # 'chair96': adaptable version
    azimuthal_angle = [random.uniform(0, 2*math.pi) for _ in range(num_samples)]
    polar_angle = [random.uniform(-math.pi, math.pi) for _ in range(num_samples)]
    prefix,object_name = 'gt','chair96'
    size = eval_one_config(default_config, object_name, prefix, azimuthal_angle, polar_angle)
    print('Total size:',size)
    prefix,object_name = 're','chair96'
    probe_config = [8,10,9,4,8,11,0,16]
    size = eval_one_config(probe_config, object_name, prefix, azimuthal_angle, polar_angle)
    print('Total size:',size)

    # calculate image quality

    # request one baseline 
    # request download links with a few angles
    # download files
    # compare optimal and customized
    # calculate result

    # Perform operations on the array
    
    # problem = MyProblem()

    # algorithm = NSGA2(pop_size=10)

    # res = minimize(problem,
    #             algorithm,
    #             ('n_gen', 20),
    #             seed=1,
    #             save_history=True)

    # print("Best solution found: %s" % res.X)
    # print("Function value: %s" % res.F)
    # print("Constraint violation: %s" % res.CV)

    # with open('profiling.log','w') as f:
    #     f.write(str(res.X) + '\n')
    #     f.write(str(res.F) + '\n')
    #     f.write(str([a.pop.get("X").tolist() for a in res.history]) + '\n')
    #     f.write(str([a.pop.get("F").tolist() for a in res.history]) + '\n')
    #     f.write(str([a.pop.get("feasible").tolist() for a in res.history]) + '\n')

    # return {"result": "Next configuration"}

if __name__ == '__main__':
    profiling()