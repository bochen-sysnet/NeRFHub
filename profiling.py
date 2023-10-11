import requests
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import lpips
import torch
from torchvision import transforms
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
import cv2
from commons import ssim_fn, AverageMeter

# run as a thread
# share a pipe of received data
# every mlp has one profile
# every mlp will have a separate problem to make searching fast
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=7, n_obj=2, n_constr=0, 
                         xl=[1,0,0,0,0,0,0,4], xu=[8,10,9,4,14,14,10,96], vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        points = []
        for row in range(x.shape[0]):
            psnr,size = eval_config(probe_config=x[row,:])
            points += [[psnr,size]]

        # retrieve result and return
        out["F"] = np.array(points)


default_config = [8,10,9,4,0,0,0,96]
url = 'http://130.126.139.208:8000/send_message'
num_samples = 10
# Define the LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex')  # You can also use 'vgg' or 'squeeze' here

# Define the transformation to ensure images have the right format
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def prune(prune_chan, d, object_name):
    # the number of prunable layers is hard-coded to 2 in MobileNeRF
    prunable_num = 2
    channel_imp = [[] for _ in range(prunable_num)]
    if d == 8:
        with open(object_name + '_phone/mlp.json', 'r') as file:
            data = json.load(file)
    else:
        with open(object_name + f'_phone/mlp.{d}.json', 'r') as file:
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
        if d == 8:
            subprocess.run(["convert",
                            basename + '.png',
                            '-depth', f'{d}',
                            '-define', f'png:compression-filter={f}',
                            '-define', f'png:compression-level={l}',
                            '-define', f'png:compression-strategy={s}',
                            basename + '.x.png',
                            ], 
                            stdout=subprocess.PIPE, text=True)
        else:
            img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
            img[:,:,3] = (img[:,:,3] - 128) * 2
            mult = 2**(8-d)
            img = img//mult*mult
            img[:,:,3] = img[:,:,3] // 2 + 128
            cv2.imwrite('tmp.png', img, [cv2.IMWRITE_PNG_COMPRESSION,9])
            subprocess.run(["convert",
                            'tmp.png',
                            '-depth', f'8',
                            '-define', f'png:compression-filter={f}',
                            '-define', f'png:compression-level={l}',
                            '-define', f'png:compression-strategy={s}',
                            basename + '.x.png',
                            ], 
                            stdout=subprocess.PIPE, text=True)

def eval_one_model(config, object_name, prefix, azimuthal_angle, polar_angle, baseline=False):
    d,f,l,s,qp,qt,cl,mlp_channel = config

    if not baseline:
        texture = 'pngx'
        mesh = 'drc'
        # mlp prune
        prune(mlp_channel,d,object_name)

        # mesh compression
        draco(qp,qt,cl,object_name)

        # texture compression
        png(d,f,l,s,object_name)
    else:
        texture = 'png'
        mesh = 'obj'
        mlp_channel = 96

    # calculate file size
    mlp_size = os.path.getsize(object_name + '_phone/mlp_p.json')

    png_size = 0
    for file in glob.glob(object_name + '_phone/shape[0-9].pngfeat[0-9].x.png'):
        png_size += os.path.getsize (file)

    drc_size = 0
    for file in glob.glob(object_name + f'_phone/' + '*.drc'):
        drc_size += os.path.getsize (file)

    size = (mlp_size, png_size, drc_size)

    # send to html for snapshots
    data = {'mlp':mlp_channel,
            'tex':texture,
            'mesh':mesh,
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
            # print(received_data)

    return size

def eval_config(probe_config = [8,10,9,4,0,0,0,16], baseline_name = 'chair96', target_name = 'chair', base_prefix = 'gt', target_prefix = 're'):
    # train a 96-model as baseline
    
    # two types of objects
    # 'chair': original 
    # 'chair96': adaptable version
    azimuthal_angle = [random.uniform(0, 2*math.pi) for _ in range(num_samples)]
    polar_angle = [random.uniform(-math.pi, math.pi) for _ in range(num_samples)]
    eval_one_model(default_config, baseline_name, base_prefix, azimuthal_angle, polar_angle, baseline=True)
    size = eval_one_model(probe_config, target_name, target_prefix, azimuthal_angle, polar_angle)
    total_size = sum(size) / 2**20

    # calculate image quality
    psnr_module = AverageMeter()
    ssim_module = AverageMeter()
    lpips_module = AverageMeter()
    for i in range(1,num_samples+1):
        gt = cv2.imread(f'profiling/{base_prefix}_{i}.png', cv2.IMREAD_UNCHANGED) 
        gt1 = gt / 255.0
        re = cv2.imread(f'profiling/{target_prefix}_{i}.png', cv2.IMREAD_UNCHANGED) 
        re1 = re / 255.0
        # psnr
        psnr = float(-10 * np.log10(np.mean(np.square(re1 - gt1))))
        psnr_module.update(psnr)
        # ssim
        ssim = ssim_fn(re, gt)
        ssim_module.update(ssim)
        # lpips
        image1 = transform(cv2.cvtColor(gt, cv2.COLOR_RGBA2RGB)).unsqueeze(0)  # Assuming image1 is a tensor in range [0, 1]
        image2 = transform(cv2.cvtColor(re, cv2.COLOR_RGBA2RGB)).unsqueeze(0)  # Assuming image2 is a tensor in range [0, 1]
        # Calculate the LPIPS metric
        lpips_value = loss_fn_alex(image1, image2)
        lpips_module.update(float(lpips_value))

    print(psnr_module.avg, ssim_module.avg, lpips_module.avg, size, total_size)

    return psnr_module.avg, size



def profiling():
    # the goal is to find similar quality, real-time and low latency config

    # use xxxxx96 at max channel and lossless compression as gt
    # compare model with less channels or default model to it
    # should handle the case with exact same config as gt
    
    problem = MyProblem()

    algorithm = NSGA2(pop_size=10)

    res = minimize(problem,
                algorithm,
                ('n_gen', 20),
                seed=1,
                save_history=True)

    with open('profiling.log','w') as f:
        f.write(str(res.X) + '\n')
        f.write(str(res.F) + '\n')
        f.write(str([a.pop.get("X").tolist() for a in res.history]) + '\n')
        f.write(str([a.pop.get("F").tolist() for a in res.history]) + '\n')
        f.write(str([a.pop.get("feasible").tolist() for a in res.history]) + '\n')

if __name__ == '__main__':
    # eval_config(probe_config = [8,10,9,4,0,0,0,96], baseline_name = 'chair96', target_name = 'chair96',)
    # eval_config(probe_config = [8,10,9,4,0,0,0,48], baseline_name = 'chair96', target_name = 'chair96',)
    # eval_config(probe_config = [8,10,9,4,0,0,0,16], baseline_name = 'chair96', target_name = 'chair96',)
    # eval_config(probe_config = [8,10,9,4,0,0,0,8], baseline_name = 'chair96', target_name = 'chair96',)
    # eval_config(probe_config = [8,10,9,4,0,0,0,4], baseline_name = 'chair96', target_name = 'chair96',)
    eval_config(probe_config = [8,10,9,4,0,0,0,16], baseline_name = 'chair96', target_name = 'chair',)