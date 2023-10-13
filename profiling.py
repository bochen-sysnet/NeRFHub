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


default_config = [8,10,9,4,0,0,0,96]
url = 'http://130.126.139.208:8000/send_message'
# Define the LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex')  # You can also use 'vgg' or 'squeeze' here
selected = 10


def load_dataset(scene_type="synthetic", object_name="drums", root_dir="../dataset/nerf_synthetic/"):
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

            if selected > 0:
                images = images[:selected]
                camtoworlds = camtoworlds[:selected]

            return {'images' : jnp.array(images), 'c2w' : jnp.array(camtoworlds), 'hwf' : jnp.array(hwf)}

        data = {'train' : load_LLFF(scene_dir, 'train'),
                'test' : load_LLFF(scene_dir, 'test')}

        # splits = ['train', 'test']
        # for s in splits:
        #     print(s)
        #     for k in data[s]:
        #         print(f'  {k}: {data[s][k].shape}')

        images, poses, hwf = data['train']['images'], data['train']['c2w'], data['train']['hwf']
        
    # print(hwf)
    # for i in range(len(images)):
    #     image,pose = images[i],poses[i]
    #     print(pose)
    #     write_floatpoint_image(f"profiling/{object_name}_{i}.png",image)
    # for i in range(3):
    #     plt.figure()
    #     plt.scatter(poses[:,i,3], poses[:,(i+1)%3,3])
    #     plt.axis('equal')
    #     plt.savefig("profiling/training_camera"+str(i)+".png")
    # exit(0)
    return data

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

def eval_one_model(config, object_name, transform_matrix, prefix='g'):
    d,f,l,s,qp,qt,cl,mlp_channel = config

    t_0 = time.perf_counter()
    if prefix == 't':
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

    cmp_time = time.perf_counter() - t_0

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
            'transform_matrix':transform_matrix,
            'prefix':prefix
            }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Request evaluation:",config)
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
            # print(config, prefix, "Message from flask server:",received_data)

    return size, cmp_time

def write_floatpoint_image(name,img):
    img = np.clip(np.array(img)*255,0,255).astype(np.uint8)
    cv2.imwrite(name,img[:,:,::-1])

def eval_config(dataset, probe_config = [8,10,9,4,0,0,0,16], base_name = 'chair96', target_name='chair', scene_type = 'synthetic'):
    # train a 96-model as baseline
    # two types of objects
    # 'chair': original 
    # 'chair96': adaptable version
    images, poses = dataset['train']['images'], dataset['train']['c2w']
    N = images.shape[0]
    transform_matrix = poses.reshape(N,-1).tolist()
    if scene_type != 'synthetic':
        # Create the [0, 0, 0, 1] array
        extra_cols = np.array([0, 0, 0, 1])

        # Stack the original array with [0, 0, 0, 1]
        transform_matrix = np.hstack((transform_matrix, np.tile(extra_cols, (transform_matrix.shape[0], 1))))
    
    # if the ground truth has not been downloaded, request eval
    if not os.path.exists(f'profiling/g{selected}.png'):
        eval_one_model(default_config, base_name, transform_matrix, prefix='g')
    size,cmp_time = eval_one_model(probe_config, target_name, transform_matrix, prefix='t')
    total_size = sum(size) / 2**20

    # Define the transformation to ensure images have the right format
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # calculate image quality
    psnr_module = AverageMeter()
    ssim_module = AverageMeter()
    lpips_module = AverageMeter()
    for i in range(1,N+1):
        g = cv2.imread(f'profiling/g{i}.png', cv2.IMREAD_UNCHANGED) 
        g = cv2.cvtColor(g, cv2.COLOR_RGBA2RGB)
        g32 = g / 255.
        t = cv2.imread(f'profiling/t{i}.png', cv2.IMREAD_UNCHANGED) 
        t = cv2.cvtColor(t, cv2.COLOR_RGBA2RGB)
        t32 = t / 255.
        # psnr
        psnr = float(-10 * np.log10(np.mean(np.square(g32 - t32))))
        psnr_module.update(psnr)
        # ssim
        ssim = float(ssim_fn(g32, t32))
        ssim_module.update(ssim)
        # lpips
        lpips_value = loss_fn_alex(transform(g).unsqueeze(0), transform(t).unsqueeze(0))
        lpips_module.update(float(lpips_value))

    print(psnr_module.avg, ssim_module.avg, lpips_module.avg, total_size, cmp_time)

    return psnr_module.avg, ssim_module.avg, lpips_module.avg, total_size



def profiling(dataset, pop_size=10, n_gen=50):
    # the goal is to find similar quality, real-time and low latency config

    # use xxxxx96 at max channel and lossless compression as gt
    # compare model with less channels or default model to it
    # should handle the case with exact same config as gt

    class MyProblem(Problem):
        def __init__(self, mlp_channel, target, base, scene_type, metrics=0):
            super().__init__(n_var=7, n_obj=2, n_constr=0, 
                            xl=[1,0,0,0,7,11,0], xu=[8,10,9,4,14,14,10], vtype=int)
            self.mlp_channel = mlp_channel
            self.target = target
            self.base = base
            self.scene_type = scene_type
            self.metrics = 0

        def _evaluate(self, x, out, *args, **kwargs):
            points = []
            for row in range(x.shape[0]):
                cfg = *x[row,:],self.mlp_channel
                
                psnr,ssim,lpips,size = eval_config(dataset, probe_config=cfg, target_name=self.target, 
                                        base_name=self.base, scene_type=self.scene_type)
                
                if self.metrics == 0:
                    points += [[-psnr,size]]
                elif self.metrics == 1:
                    points += [[-ssim,size]]
                elif self.metrics == 2:
                    points += [[lpips,size]]
            # retrieve result and return
            out["F"] = np.array(points)
    
    problem = MyProblem(32, 'chair96', 'chair96', 'synthetic', metrics=0)
    
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

    with open(f'profiling.{n_gen}.log','w') as f:
        f.write(str(res.X) + '\n')
        f.write(str(res.F) + '\n')
        f.write(str([a.pop.get("X").tolist() for a in res.history]) + '\n')
        f.write(str([a.pop.get("F").tolist() for a in res.history]) + '\n')
        f.write(str([a.pop.get("feasible").tolist() for a in res.history]) + '\n')

if __name__ == '__main__':
    # synthetic
    # nerf_synthetic
    # chair drums ficus hotdog lego materials mic ship

    # forwardfacing
    # nerf_llff_data
    # fern flower fortress horns leaves orchids room trex


    object_name = 'chair'
    dataset = load_dataset(scene_type="synthetic", object_name=object_name, root_dir="../dataset/nerf_synthetic/")
    for n_gen in [5,10,20]:
        profiling(dataset,n_gen=n_gen, pop_size=50)

    # eval_config(dataset, probe_config = [8,10,9,4,0,0,0,16], base_name='chair96', target_name = object_name, scene_type='synthetic')

