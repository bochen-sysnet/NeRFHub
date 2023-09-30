# shared variables and functions

VQ = True
if VQ:
  bit_width = 12
  quant_levels = 2**bit_width
else:
  bit_width = 0

# channel_width = 16
channel_width = 96

num_bottleneck_features = 8

prefix = f'B{bit_width}_C{channel_width}_F{num_bottleneck_features}_'

scene_type = "synthetic"
object_name = "chair"
scene_dir = "../dataset/nerf_synthetic/"+object_name

# synthetic
# chair drums ficus hotdog lego materials mic ship

# forwardfacing
# fern flower fortress horns leaves orchids room trex

# real360
# bicycle flowerbed gardenvase stump treehill
# fulllivingroom kitchencounter kitchenlego officebonsai


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
