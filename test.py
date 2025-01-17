import open3d as o3d
from open3d.t.geometry import PointCloud
from open3d.core import Tensor, Device

print("Is CUDA supported?:", o3d.core.cuda.is_available())
