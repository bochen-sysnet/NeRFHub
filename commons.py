# shared variables and functions
import jax
import jax.numpy as np
import functools

from jax.experimental.host_callback import call
import copy


# whether train multiple mlp
onn = True

# test at epoch 0
zero_test = False

# starting step
step_init = 1

# model config
num_bottleneck_features = 8
data_type, floatnum = np.float32, 32

# single test during training
pruned_to_test = 0

if not onn:
  pruned_to_eval = [0]
  channel_width = 16
  total_phases = 0
  step_per_phase = 200000
  def phase2pruned_channel(phase):
     return 0
else:
  step_per_phase = 100000
  # channel_width,total_phases,pruned_to_eval = 96,8,[80]
  # def phase2pruned_channel(phase):
  #   return np.array([16 * phase, 80]).min()
  channel_width,total_phases,pruned_to_eval = 64,5,[48]
  def phase2pruned_channel(phase):
    return np.array([16 * phase, 48]).min()

# synthetic
# nerf_synthetic
# chair drums ficus hotdog lego materials mic ship

# forwardfacing
# nerf_llff_data
# fern flower fortress horns leaves orchids room trex

# real360
# bicycle flowerbed gardenvase stump treehill
# fulllivingroom kitchencounter kitchenlego officebonsai
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

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

# do pruning on MLP
def apply_prune(mlp, prune_chan = 0, prunable_num = 2):
  # calculate importance
  channel_imp = []
  for i in range(prunable_num):
    param = mlp['params'][f'Dense_{i+1}']['kernel']
    channel_imp.append(np.abs(param).sum(axis=-1))
  # prune channels
  tmp_mlp = copy.copy(mlp)
  for i in range(prunable_num):
      sorted_imp = np.sort(channel_imp[i])
      threshold = sorted_imp[prune_chan]
      # Broadcast the mask to the shape of param
      broadcasted_imp = np.broadcast_to(channel_imp[i], mlp['params'][f'Dense_{i}']['kernel'].shape)
      tmp_mlp['params'][f'Dense_{i}']['kernel'] *= np.where(broadcasted_imp < threshold, 0.0, 1.0)
      broadcasted_imp = np.broadcast_to(channel_imp[i][:,np.newaxis], mlp['params'][f'Dense_{i+1}']['kernel'].shape)
      tmp_mlp['params'][f'Dense_{i+1}']['kernel'] *= np.where(broadcasted_imp < threshold, 0.0, 1.0)
      tmp_mlp['params'][f'Dense_{i}']['bias'] *= np.where(channel_imp[i] < threshold, 0.0, 1.0)
      # call(lambda x: print(f"------{x}-------"), threshold)
      # call(lambda x: print(f"------{x}-------"), np.where(channel_imp[i] < threshold, 0.0, 1.0))

  return tmp_mlp

def prune_grad(mlp, mlp_grad, prune_chan = 0, prunable_num = 2):
  # calculate importance
  channel_imp = []
  for i in range(prunable_num):
    param = mlp['params'][f'Dense_{i+1}']['kernel']
    channel_imp.append(np.abs(param).sum(axis=-1))
  # prune channels
  for i in range(prunable_num):
      sorted_imp = np.sort(channel_imp[i])
      threshold = sorted_imp[prune_chan]
      # Broadcast the mask to the shape of param
      broadcasted_imp = np.broadcast_to(channel_imp[i], mlp['params'][f'Dense_{i}']['kernel'].shape)
      mlp_grad['params'][f'Dense_{i}']['kernel'] *= np.where(broadcasted_imp < threshold, 0.0, 1.0)
      broadcasted_imp = np.broadcast_to(channel_imp[i][:,np.newaxis], mlp['params'][f'Dense_{i+1}']['kernel'].shape)
      mlp_grad['params'][f'Dense_{i+1}']['kernel'] *= np.where(broadcasted_imp < threshold, 0.0, 1.0)
      mlp_grad['params'][f'Dense_{i}']['bias'] *= np.where(channel_imp[i] < threshold, 0.0, 1.0)
  return mlp_grad

import jax.numpy as jnp
import jax.scipy as jsp

def compute_ssim(img0,
                 img1,
                 max_val,
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 return_map=False):
  """Computes SSIM from two images.
  This function was modeled after tf.image.ssim, and should produce comparable
  output.
  Args:
    img0: array. An image of size [..., width, height, num_channels].
    img1: array. An image of size [..., width, height, num_channels].
    max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
    filter_size: int >= 1. Window size.
    filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
    k1: float > 0. One of the SSIM dampening parameters.
    k2: float > 0. One of the SSIM dampening parameters.
    return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned
  Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
  """
  # Construct a 1D Gaussian blur filter.
  hw = filter_size // 2
  shift = (2 * hw - filter_size + 1) / 2
  f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma)**2
  filt = jnp.exp(-0.5 * f_i)
  filt /= jnp.sum(filt)

  # Blur in x and y (faster than the 2D convolution).
  filt_fn1 = lambda z: jsp.signal.convolve2d(z, filt[:, None], mode="valid")
  filt_fn2 = lambda z: jsp.signal.convolve2d(z, filt[None, :], mode="valid")

  # Vmap the blurs to the tensor size, and then compose them.
  num_dims = len(img0.shape)
  map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
  for d in map_axes:
    filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
    filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
  filt_fn = lambda z: filt_fn1(filt_fn2(z))

  mu0 = filt_fn(img0)
  mu1 = filt_fn(img1)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filt_fn(img0**2) - mu00
  sigma11 = filt_fn(img1**2) - mu11
  sigma01 = filt_fn(img0 * img1) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  sigma00 = jnp.maximum(0., sigma00)
  sigma11 = jnp.maximum(0., sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
  return ssim_map if return_map else ssim

# Compiling to the CPU because it's faster and more accurate.
ssim_fn = jax.jit(
    functools.partial(compute_ssim, max_val=1.), backend="cpu")