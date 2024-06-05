import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from utils.math import qvec2rotmat
from gaussian_splatting.utils.graphics_utils import focal2fov
    
class DroidParser:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.color_paths = sorted(glob.glob(f"{self.dataset_path}/color/*.jpg"))
        if len(self.color_paths) == 0:
            self.color_paths = sorted(glob.glob(f"{self.dataset_path}/color/*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.dataset_path}/depth/*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(self.dataset_path)
        self.load_kf_indices(self.dataset_path)
    
    def load_kf_indices(self,dataset_path):
        kf_info = np.load(os.path.join(dataset_path,"trajectory_w2c_kf.npy"))
        self.kf_indices = []
        for extr in kf_info:
            self.kf_indices.append(int(extr[0]))
            
    def load_poses(self, dataset_path):
       
        
        self.R = []
        self.T = []
        self.poses = []
        self.poses_droid = []
        frames = []
        
        
        T_droid = np.load(os.path.join(dataset_path,"trajectory_w2c.npy"))
        try:
            T_gt = np.loadtxt(os.path.join(dataset_path,"gt_w2c.txt"))
            
            for i, (extr_cam, extr_gt) in enumerate(zip(T_droid, T_gt)):
                R_cam = qvec2rotmat(extr_cam[[7, 4, 5, 6]])
                T_cam = np.array(extr_cam[1:4])
                pose_cam = np.eye(4)
                pose_cam[:3, :3] = R_cam
                pose_cam[:3, 3] = T_cam
                self.poses_droid.append(pose_cam)

                R_gt = qvec2rotmat(extr_gt[[6, 3, 4, 5]])
                T_gt = np.array(extr_gt[0:3])
                pose_gt = np.eye(4)
                pose_gt[:3, :3] = R_gt
                pose_gt[:3, 3] = T_gt
                self.poses.append(pose_gt)

                frame = {
                    "file_path": self.color_paths[i],
                    "depth_path": self.depth_paths[i],
                    "transform_matrix": pose_gt.tolist(), 
                }
                frames.append(frame)
            
        except:
            T_gt = None
            for i, extr_cam in enumerate(range(0, len(self.color_paths))):
                extr_cam = T_droid[i,:]
                R_cam = qvec2rotmat(extr_cam[[7, 4, 5, 6]])
                T_cam = np.array(extr_cam[1:4])
                pose_cam = np.eye(4)
                pose_cam[:3, :3] = R_cam
                pose_cam[:3, 3] = T_cam
                self.poses_droid.append(pose_cam)
                
                pose_gt = np.eye(4)
                self.poses.append(pose_gt)

                frame = {
                    "file_path": self.color_paths[i],
                    "depth_path": self.depth_paths[i],
                    "transform_matrix": pose_gt.tolist(), 
                }
                frames.append(frame)
            
        
            
        self.frames = frames
        
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        intrinsic = config["Dataset"]["intrinsic"]
        # Camera prameters
        self.fx = intrinsic["fx"]
        self.fy = intrinsic["fy"]
        self.cx = intrinsic["cx"]
        self.cy = intrinsic["cy"]
        self.width = intrinsic["width"]
        self.height = intrinsic["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = intrinsic["distorted"]
        self.dist_coeffs = np.array(
            [
                intrinsic["k1"],
                intrinsic["k2"],
                intrinsic["p1"],
                intrinsic["p2"],
                intrinsic["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in intrinsic.keys() else False
        self.depth_scale = intrinsic["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        pose_droid = self.poses_droid[idx]
        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        pose_droid = torch.from_numpy(pose_droid).to(device=self.device)
        return image, depth, pose, pose_droid

class DoridDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = DroidParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.kf_indices = parser.kf_indices
        self.poses = parser.poses
        self.poses_droid = parser.poses_droid
        self.R = parser.R
        self.T = parser.T
        

def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "droid":
        return DoridDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
