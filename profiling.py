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
from commons import ssim_fn, AverageMeter, data_type
import time
from multiprocessing.pool import ThreadPool
from PIL import Image
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from multiprocessing.connection import Listener



default_config = [8,10,9,4,0,0,0,96]
url = 'http://130.126.139.208:8000/send_message'
# Define the LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex')  # You can also use 'vgg' or 'squeeze' here


def load_dataset(scene_type="synthetic", object_name="drums", root_dir="../dataset/nerf_synthetic/", selected=0):
    #%% --------------------------------------------------------------------------------
    # ## Load the dataset
    #%%
    # """ Load dataset """
    scene_dir = root_dir+object_name

    if scene_type=="synthetic":
        white_bkgd = True
    elif scene_type=="forwardfacing":
        white_bkgd = False
    elif scene_type=="real360":
        white_bkgd = False


    #https://github.com/google-research/google-research/blob/master/snerg/nerf/datasets.py


    if scene_type=="synthetic":
        import jax.numpy as np

        def load_blender(data_dir, split):
            with open(os.path.join(data_dir, "transforms_{}.json".format(split)), "r") as fp:
                meta = json.load(fp)

            cams = []
            paths = []

            if selected > 0:
                selected_frames = meta["frames"][:selected]
            else:
                selected_frames = meta["frames"]
            for i in range(len(selected_frames)):
                frame = meta["frames"][i]
                cams.append(np.array(frame["transform_matrix"], dtype=data_type))

                fname = os.path.join(data_dir, frame["file_path"] + ".png")
                paths.append(fname)

            def image_read_fn(fname):
                with open(fname, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=data_type) / 255.
                return image
            with ThreadPool() as pool:
                images = pool.map(image_read_fn, paths)
                pool.close()
                pool.join()

            images = np.stack(images, axis=0)
            if white_bkgd:
                images = (images[..., :3] * images[..., -1:] + (1. - images[..., -1:]))
            else:
                images = images[..., :3] * images[..., -1:]

            h, w = images.shape[1:3]
            camera_angle_x = float(meta["camera_angle_x"])
            focal = .5 * w / np.tan(.5 * camera_angle_x)

            hwf = np.array([h, w, focal], dtype=data_type)
            poses = np.stack(cams, axis=0)
            return {'images' : images, 'c2w' : poses, 'hwf' : hwf}

        data = {'train' : load_blender(scene_dir, 'train'),
                'test' : load_blender(scene_dir, 'test')}

        splits = ['train', 'test']
        # for s in splits:
        #     print(s)
        #     for k in data[s]:
        #         print(f'  {k}: {data[s][k].shape}')

        images, poses, hwf = data['train']['images'], data['train']['c2w'], data['train']['hwf']

        import numpy as np
    
    elif scene_type=="forwardfacing" or scene_type=="real360":

        import jax.numpy as jnp
        import numpy as np

        def _viewmatrix(z, up, pos):
            """Construct lookat view matrix."""
            vec2 = _normalize(z)
            vec1_avg = up
            vec0 = _normalize(np.cross(vec1_avg, vec2))
            vec1 = _normalize(np.cross(vec2, vec0))
            m = np.stack([vec0, vec1, vec2, pos], 1)
            return m

        def _normalize(x):
            """Normalization helper function."""
            return x / np.linalg.norm(x)

        def _poses_avg(poses):
            """Average poses according to the original NeRF code."""
            hwf = poses[0, :3, -1:]
            center = poses[:, :3, 3].mean(0)
            vec2 = _normalize(poses[:, :3, 2].sum(0))
            up = poses[:, :3, 1].sum(0)
            c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
            return c2w

        def _recenter_poses(poses):
            """Recenter poses according to the original NeRF code."""
            poses_ = poses.copy()
            bottom = np.reshape([0, 0, 0, 1.], [1, 4])
            c2w = _poses_avg(poses)
            c2w = np.concatenate([c2w[:3, :4], bottom], -2)
            bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
            poses = np.concatenate([poses[:, :3, :4], bottom], -2)
            poses = np.linalg.inv(c2w) @ poses
            poses_[:, :3, :4] = poses[:, :3, :4]
            poses = poses_
            return poses

        def _transform_poses_pca(poses):
            """Transforms poses so principal components lie on XYZ axes."""
            poses_ = poses.copy()
            t = poses[:, :3, 3]
            t_mean = t.mean(axis=0)
            t = t - t_mean

            eigval, eigvec = np.linalg.eig(t.T @ t)
            # Sort eigenvectors in order of largest to smallest eigenvalue.
            inds = np.argsort(eigval)[::-1]
            eigvec = eigvec[:, inds]
            rot = eigvec.T
            if np.linalg.det(rot) < 0:
                rot = np.diag(np.array([1, 1, -1])) @ rot

            transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
            bottom = np.broadcast_to([0, 0, 0, 1.], poses[..., :1, :4].shape)
            pad_poses = np.concatenate([poses[..., :3, :4], bottom], axis=-2)
            poses_recentered = transform @ pad_poses
            poses_recentered = poses_recentered[..., :3, :4]
            transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

            # Flip coordinate system if z component of y-axis is negative
            if poses_recentered.mean(axis=0)[2, 1] < 0:
                poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
                transform = np.diag(np.array([1, -1, -1, 1])) @ transform

            # Just make sure it's it in the [-1, 1]^3 cube
            scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
            poses_recentered[:, :3, 3] *= scale_factor
            transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

            poses_[:, :3, :4] = poses_recentered[:, :3, :4]
            poses_recentered = poses_
            return poses_recentered, transform

        def load_LLFF(data_dir, split, factor = 4, llffhold = 8):
            # Load images.
            imgdir_suffix = ""
            if factor > 0:
                imgdir_suffix = "_{}".format(factor)
            imgdir = os.path.join(data_dir, "images" + imgdir_suffix)
            if not os.path.exists(imgdir):
                raise ValueError("Image folder {} doesn't exist.".format(imgdir))
            imgfiles = [
                os.path.join(imgdir, f)
                for f in sorted(os.listdir(imgdir))
                if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
            ]
            if selected > 0 and split == 'train':
                imgfiles = imgfiles[:selected]
            def image_read_fn(fname):
                with open(fname, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=data_type) / 255.
                return image
            with ThreadPool() as pool:
                images = pool.map(image_read_fn, imgfiles)
                pool.close()
                pool.join()
            images = np.stack(images, axis=-1)

            # Load poses and bds.
            with open(os.path.join(data_dir, "poses_bounds.npy"), "rb") as fp:
                poses_arr = np.load(fp)
            poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
            bds = poses_arr[:, -2:].transpose([1, 0])
            if poses.shape[-1] != images.shape[-1]:
                raise RuntimeError("Mismatch between imgs {} and poses {}".format(
                    images.shape[-1], poses.shape[-1]))

            # Update poses according to downsampling.
            poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
            poses[2, 4, :] = poses[2, 4, :] * 1. / factor

            # Correct rotation matrix ordering and move variable dim to axis 0.
            poses = np.concatenate(
                [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
            poses = np.moveaxis(poses, -1, 0).astype(data_type)
            images = np.moveaxis(images, -1, 0)
            bds = np.moveaxis(bds, -1, 0).astype(data_type)


            if scene_type=="real360":
                # Rotate/scale poses to align ground with xy plane and fit to unit cube.
                poses, _ = _transform_poses_pca(poses)
            else:
                # Rescale according to a default bd factor.
                scale = 1. / (bds.min() * .75)
                poses[:, :3, 3] *= scale
                bds *= scale
                # Recenter poses
                poses = _recenter_poses(poses)

            # Select the split.
            i_test = np.arange(images.shape[0])[::llffhold]
            i_train = np.array(
                [i for i in np.arange(int(images.shape[0])) if i not in i_test])
            if split == "train":
                indices = i_train
            else:
                indices = i_test
            images = images[indices]
            poses = poses[indices]

            camtoworlds = poses[:, :3, :4]
            focal = poses[0, -1, -1]
            h, w = images.shape[1:3]

            hwf = np.array([h, w, focal], dtype=data_type)

            return {'images' : jnp.array(images), 'c2w' : jnp.array(camtoworlds), 'hwf' : jnp.array(hwf)}

        data = {'train' : load_LLFF(scene_dir, 'train'),
                'test' : load_LLFF(scene_dir, 'test')}

        # splits = ['train', 'test']
        # for s in splits:
        #     print(s)
        #     for k in data[s]:
        #         print(f'  {k}: {data[s][k].shape}')

        images, poses, hwf = data['train']['images'], data['train']['c2w'], data['train']['hwf']
        
    return data

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                clear_directory(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def probe_different_knobs():
    object_name = 'flower'
    scene_type = scene2type(object_name)
    root_dir = scene2root(object_name)
    dataset = load_dataset(scene_type=scene_type, object_name=object_name, root_dir=root_dir, selected=0)
    for i in range(1,9):
        psnr,_,_,size = eval_config(dataset, probe_config = [i,10,9,4,0,0,0,16], )
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

    for i in range(0,11):
        psnr,_,_,size = eval_config(dataset, probe_config = [8,i,9,4,0,0,0,16],)
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

    for i in range(0,10):
        psnr,_,_,size = eval_config(dataset, probe_config = [8,10,i,4,0,0,0,16],)
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

    for i in range(0,5):
        psnr,_,_,size = eval_config(dataset, probe_config = [8,10,9,i,0,0,0,16],)
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

    for i in range(0,15):
        psnr,_,_,size = eval_config(dataset, probe_config = [8,10,9,4,i,0,0,16],)
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

    for i in range(0,15):
        psnr,_,_,size = eval_config(dataset, probe_config = [8,10,9,4,0,i,0,16],)
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

    for i in range(0,11):
        psnr,_,_,size = eval_config(dataset, probe_config = [8,10,9,4,0,0,i,16],)
        with open('knob.log','a+') as f:
            f.write(f'{i},{psnr},{size}\n')

def prune(prune_chan, d, object_name):
    # the number of prunable layers is hard-coded to 2 in MobileNeRF
    prunable_num = 2
    channel_imp = [[] for _ in range(prunable_num)]
    if d >= 5:
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
            tmp_file = basename + f'.{d}.png'
            if not os.path.exists(tmp_file):
                img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
                img[:,:,3] = (img[:,:,3] - 128) * 2 # 0-255
                
                mult = 2**(8-d) # 1 (d=8), 128 (d=1)
                valid_pixels = (img[..., 2] != 0)
                img[valid_pixels] = np.clip(img[valid_pixels] // mult * mult, 1, 255)

                img[:,:,3] = img[:,:,3] // 2 + 128
                cv2.imwrite(tmp_file, img, [cv2.IMWRITE_PNG_COMPRESSION,9])
            subprocess.run(["convert",
                            tmp_file,
                            '-depth', f'8',
                            '-define', f'png:compression-filter={f}',
                            '-define', f'png:compression-level={l}',
                            '-define', f'png:compression-strategy={s}',
                            basename + '.x.png',
                            ], 
                            stdout=subprocess.PIPE, text=True)

def eval_one_model(config, object_name, transform_matrix, prefix='g', measure_fps=False):
    d,f,l,s,qp,qt,cl,mlp_channel = config

    t_0 = time.perf_counter()
    if prefix != 'g':
        texture = 'pngx'
        mesh = 'drc'
        # mlp prune
        prune(mlp_channel,d,object_name)

        # mesh compression
        draco(qp,qt,cl,object_name)

        # texture compression
        png(d,f,l,s,object_name)

        # calculate file size
        mlp_size = os.path.getsize(object_name + '_phone/mlp_p.json')

        png_size = 0
        for file in glob.glob(object_name + '_phone/shape[0-9].pngfeat[0-9].x.png'):
            png_size += os.path.getsize (file)

        drc_size = 0
        for file in glob.glob(object_name + f'_phone/' + '*.drc'):
            drc_size += os.path.getsize (file)
    else:
        texture = 'png'
        mesh = 'obj'
        mlp_channel = 96

        # calculate file size
        mlp_size = os.path.getsize(object_name + '_phone/mlp.json')

        png_size = 0
        for file in glob.glob(object_name + '_phone/shape[0-9].pngfeat[0-9].png'):
            png_size += os.path.getsize (file)

        drc_size = 0
        for file in glob.glob(object_name + f'_phone/' + '*.obj'):
            drc_size += os.path.getsize (file)

    cmp_time = time.perf_counter() - t_0

    size = (mlp_size, png_size, drc_size)

    # send to html for snapshots
    data = {'mlp':mlp_channel,
            'tex':texture,
            'mesh':mesh,
            'scene_option':object_name,
            'transform_matrix':transform_matrix,
            'prefix':prefix,
            'measure_fps':measure_fps
            }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Request evaluation:",config)
    else:
        print(f"Error: {response.status_code}")

    address = ('localhost', 8015)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()
    while True:
        data = conn.recv()
        received_data = pickle.loads(data)
        print(received_data)
        break
    listener.close()

    # # Set up a socket to receive data from the Flask server
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.bind(('localhost', 8015))  # Use the same IP and port as in the Flask app
    #     s.listen()
    #     conn, addr = s.accept()

    #     with conn:
    #         data = conn.recv(1024)
    #         received_data = pickle.loads(data)
    #         # print(config, prefix, "Message from flask server:",received_data)

    return size, cmp_time

def write_floatpoint_image(name,img):
    img = np.clip(np.array(img)*255,0,255).astype(np.uint8)
    cv2.imwrite(name,img[:,:,::-1])

def eval_config(dataset, probe_config = [8,10,9,4,0,0,0,16], base_name = 'chairD', target_name='chair', 
                scene_type = 'synthetic', target_prefix='t', split = 'train'):
    # two types of objects
    if split == 'train':
        images, poses = dataset['train']['images'], dataset['train']['c2w']
    else:
        images, poses = dataset['test']['images'], dataset['test']['c2w']
    N = images.shape[0]
    transform_matrix = poses.reshape(N,-1).tolist()
    if scene_type != 'synthetic':
        # Create the [0, 0, 0, 1] array
        extra_cols = np.array([0, 0, 0, 1])

        # Stack the original array with [0, 0, 0, 1]
        transform_matrix = np.hstack((transform_matrix, np.tile(extra_cols, (len(transform_matrix), 1)))).tolist()
    
    # if the ground truth has not been downloaded, request eval
    # need to empty the folder every time
    if not os.path.exists(f'profiling_cache/g1.png'):
        eval_one_model(default_config, base_name, transform_matrix, prefix='g')
    size,cmp_time = eval_one_model(probe_config, target_name, transform_matrix, prefix=target_prefix)
    total_size = sum(size) / 2**20

    # check not the same images in comparison
    if probe_config == [8, 10, 9, 4, 0, 0, 0, 96]:
        return 100

    # Define the transformation to ensure images have the right format
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # calculate image quality
    psnr_module = AverageMeter()
    ssim_module = AverageMeter()
    lpips_module = AverageMeter()
    for i in range(1,N+1):
        g = cv2.imread(f'profiling_cache/g{i}.png', cv2.IMREAD_UNCHANGED) 
        g = cv2.cvtColor(g, cv2.COLOR_RGBA2RGB)
        g32 = g / 255.
        t = cv2.imread(f'profiling_cache/{target_prefix}{i}.png', cv2.IMREAD_UNCHANGED) 
        t = cv2.cvtColor(t, cv2.COLOR_RGBA2RGB)
        t32 = t / 255.
        # psnr
        psnr = float(-10 * np.log10(np.mean(np.square(g32 - t32)) + 1e-6))
        psnr_module.update(psnr)
        # ssim
        ssim = float(ssim_fn(g32, t32))
        ssim_module.update(ssim)
        # lpips
        lpips_value = loss_fn_alex(transform(g).unsqueeze(0), transform(t).unsqueeze(0))
        lpips_module.update(float(lpips_value))

    print(psnr_module.avg, ssim_module.avg, lpips_module.avg, total_size, cmp_time)

    return psnr_module.avg, ssim_module.avg, lpips_module.avg, total_size

def scene2type(target_name):
    synthetic_list = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    scene_type = 'forwardfacing'
    for scene in synthetic_list:
        if scene in target_name:
            scene_type = 'synthetic'
            break
    return scene_type

def scene2root(target_name):
    synthetic_list = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    root_dir='../dataset/nerf_llff_data/'
    for scene in synthetic_list:
        if scene in target_name:
            root_dir = "../dataset/nerf_synthetic/"
            break
    return root_dir

def profiling(object_name, channel_num=32, pop_size=50, n_gen=10, split='train', selected=10):
    t = time.time()
    clear_directory('profiling_cache/')
    target_name=object_name+'H'
    base_name=object_name+'H'
    scene_type = scene2type(object_name)
    root_dir = scene2root(object_name)
    dataset = load_dataset(scene_type=scene_type, object_name=object_name, root_dir=root_dir, selected=selected)

    class MyProblem(Problem):
        def __init__(self, mlp_channel, target, base, scene_type, metrics=0, split='train'):
            super().__init__(n_var=7, n_obj=2, n_constr=0, 
                            xl=[1,0,0,0,7,11,0], xu=[8,10,9,4,14,14,6], vtype=int)
            self.mlp_channel = mlp_channel
            self.target = target
            self.base = base
            self.scene_type = scene_type
            self.metrics = 0
            self.split = split
            self.counter = 0

        def _evaluate(self, x, out, *args, **kwargs):
            print(self.counter, end=',')
            self.counter += 1

            points = []
            for row in range(x.shape[0]):
                cfg = *x[row,:],self.mlp_channel
                
                psnr,ssim,lpips,size = eval_config(dataset, probe_config=cfg, target_name=self.target, 
                                        base_name=self.base, scene_type=self.scene_type, split=self.split)
                
                if self.metrics == 0:
                    points += [[-psnr,size]]
                elif self.metrics == 1:
                    points += [[-ssim,size]]
                elif self.metrics == 2:
                    points += [[lpips,size]]
            # retrieve result and return
            out["F"] = np.array(points)
    
    problem = MyProblem(channel_num, target_name, base_name, scene_type, metrics=0, split=split)
    
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    algorithm = NSGA2(pop_size=pop_size,
                      sampling=IntegerRandomSampling(),
                      crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      eliminate_duplicates=True,)

    res = minimize(problem,
                algorithm,
                ('n_gen', n_gen),
                seed=1,
                save_history=True)

    if split == 'train':
        filename = f'profiles/profile_{target_name}.{channel_num}.{split}.log'
    else:
        filename = f'profiles/profile_{target_name}.{channel_num}.optimal.log'
    with open(filename,'w') as f:
        f.write(str(res.X.tolist()) + '\n')
        f.write(str(res.F.tolist()) + '\n')
        f.write(str([a.pop.get("X").tolist() for a in res.history]) + '\n')
        f.write(str([a.pop.get("F").tolist() for a in res.history]) + '\n')
    t_total = time.time() - t
    with open(f"profiles/time.log",'a+') as f:
        f.write(f'{target_name},{channel_num},{t_total}\n')

def data_from_profiling(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i,line in enumerate(lines):
            if i == 0:
                configs = np.array(eval(line))
            elif i == 1:
                metrics = np.array(eval(line)); metrics[...,0] *= -1
            elif i == 2:
                explored_configs = np.array(eval(line))
            elif i == 3:
                explored_metrics = np.array(eval(line)); explored_metrics[...,0] *= -1
            else:
                break
    return configs,metrics,explored_configs,explored_metrics

def evaluate_profile(object_name, channel_num=32, n_gen=10, selected=10):
    target_name=object_name+'H'
    base_name=object_name+'H'
    scene_type = scene2type(object_name)
    root_dir = scene2root(object_name)
    dataset = load_dataset(scene_type=scene_type, object_name=object_name, root_dir=root_dir, selected=selected)
    profile,_,_,_ = data_from_profiling(f'profiles/profile_{target_name}.{channel_num}.train.log')

    # first run the profile on test set
    clear_directory('profiling_cache/')
    scene_type = scene2type(target_name)
    results = []
    for config in profile:
        cfg = *config,channel_num
        psnr,ssim,lpips,size = eval_config(dataset, probe_config=cfg, target_name=target_name, 
                                        base_name=base_name, scene_type=scene_type, split='test')
        results += [[psnr, size, ssim, lpips]]

    with open(f'profiles/profile_{target_name}.{channel_num}.eval.log','a+') as f:
        f.write(str(results) + '\n')
    
    # second run the default configuration on train and test sets
    if not os.path.exists(f'profiles/profile_{target_name}.default.log'):
        for split in ['train','test']:
            clear_directory('profiling_cache/')
            psnr,ssim,lpips,size = eval_config(dataset, target_name=target_name, 
                                            base_name=base_name, scene_type=scene_type, split=split)
            with open(f'profiles/profile_{target_name}.default.log','a+') as f:
                f.write(f'{psnr},{size},{ssim},{lpips}\n')

    # third run the profiling process on the test set for offline optimal
    # skip this step for now
    # if not os.path.exists(f'profiles/profile_{target_name}.{channel_num}.optimal.log'):
    #     profiling(object_name,channel_num=channel_num, n_gen=n_gen, pop_size=pop_size, split='test')

    # finally find the correct configuration to run on each scene, device
    generate_profile_final(object_name, channel)

def find_closest_row(arr, X):
    # Compute absolute differences
    differences = np.abs(arr[:, 0] - X)
    
    # Find index of minimum difference
    closest_row_index = np.argmin(differences)
    
    return closest_row_index

# save a single profile for each scene and channel by finding the cfg with the closest psnr
def generate_profile_final(object_name, channel):
    with open(f'profiles/profile_{object_name}H.{channel}.train.log', 'r') as file:
        lines = file.readlines()
        configs = np.array(eval(lines[0]))
        metrics = np.array(eval(lines[1]))
        metrics[...,0] *= -1

    with open(f'profiles/profile_{object_name}H.{channel}.eval.log', 'r') as file:
        lines = file.readlines()
        eval_metrics = np.array(eval(lines[0]))

    with open(f'profiles/profile_{object_name}H.default.log', 'r') as file:
        lines = file.readlines()
        default_train = lines[0].strip().split(',')

    closest_row = find_closest_row(metrics, float(default_train[0]))
    chosen_cfg = configs[closest_row].tolist()
    chosen_cfg.append(channel)

    # derive the actual size we achieve
    psnr, size, ssim, lpips = eval_metrics[closest_row].tolist()
    our_eval_result = f'{psnr},{size},{ssim},{lpips}'

    # record config for online test
    with open('profiles/profile_final.log', 'a+') as f:
        f.write(f'{object_name}\n{chosen_cfg}\n{our_eval_result}\n')

def device_profiling(N = 5):
    clear_directory('profiling_cache/')
    object_to_profile = ['chair','flower']
    for object_name in object_to_profile:
        ch_list = [16,32,48,64,80,96]
        for channel in ch_list:
            config = [8,10,9,4,0,0,0,channel]

            target_name=object_name+'H'
            transform_matrix = []
            # test for a few rounds
            # metrics will be saved to metrics.log
            for _ in range(N):
                eval_one_model(config, target_name, transform_matrix, prefix='t', measure_fps=True)

def simulation(N = 1):
    # repeat for different network

    # this function tests the latency at all possible channels
    # to find the actual latency of a hardware, 
    # we simply find the channel that is suitable for the hardware.

    object_to_config = {}
    with open('profiles/profile_final.log', 'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if i%3 == 1:
                line = eval(line)
                object_to_config[object_name].append(line)
            else:
                if i%3 == 0:
                    object_name = line.strip()
                    if object_name not in object_to_config:
                        object_to_config[object_name] = []
    
    for object_name in object_to_config:
        configs = object_to_config[object_name]
        configs.append([8,10,9,4,0,0,0,16])
        for config in configs:
            target_name=object_name+'H'
            transform_matrix = []
            # test for a few rounds
            # metrics will be saved to metrics.log
            for _ in range(N):
                eval_one_model(config, target_name, transform_matrix, prefix='t', measure_fps=True)


    # plot [default, 16,32,48,64]

# usage:
# profiling() and evaluate_profile() for all objects and all channels to obtain quality and size metrics
# simulation() to obtain latency and fps metrics on different networks and devices
# device_profiling() to obtain device profiles, after all 
        
if __name__ == '__main__':
    channel_to_profile = [16,32,48,64]
    object_to_profile = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship',
                         'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']

    # object_name = 'flower'
    selected = 10
    pop_size = 50
    n_gen = 20

    # assume the directory name is object_name + 'H'
    # 'lego','hotdog',
    for object_name in ['leaves','trex','chair']:
        for channel in channel_to_profile:
            profiling(object_name,channel_num=channel, n_gen=n_gen, pop_size=pop_size, split='train', selected=selected)
            evaluate_profile(object_name, channel_num=channel, n_gen=n_gen, selected=selected)
    
    
    # simulation()