import os
import sys
import shutil
import time
import random
from tqdm import tqdm
from argparse import ArgumentParser
from munch import munchify
import matplotlib.pyplot as plt
from torch import nn
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
from rich import print
from gui import gui_utils,slam_gui
from utils.multiprocessing_utils import clone_obj
from gaussian_splatting.gaussian_renderer import render
from gaussian_renderer_unc import render_unc

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_ate_tracking, eval_rendering, save_gaussians,eval_rendering_mapping
from utils.logging_utils import Log, get_style
from utils.pose_utils import update_pose
from utils.multiprocessing_utils import FakeQueue
from utils.slam_utils import get_loss_mapping, get_loss_tracking, \
    get_median_depth, get_isotropic_loss_1, final_loss,l1_loss,ssim, l1dep_loss,get_isotropic_loss_2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class SLAM:
    def __init__(self, config, save_dir=None):
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True,weights="alex"
        ).to("cuda")
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir

        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.tracking_loss = 0
        self.tracking_iter = 0
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        
        self.kf_indices = []
        self.mapping_iter_count = 0
        self.keyframe_visibility = {}
        self.current_keyframe_window = []
        self.viewpoints = {}

        self.use_every_n_frames = 1

        self.cameras = dict()
        self.device = "cuda:0" 
        self.set_tracking_params()
        self.set_mapping_params()
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(device=self.dataset.device)
        
        mapping = False
        
        plt.ion()
        fig= plt.figure()


        '''
        ***************************************************************************************************
        ***************************************************************************************************
        @Start
        ***************************************************************************************************
        ***************************************************************************************************
        '''
    
        for cur_frame_idx in range(0, self.dataset.num_imgs):
            # tracking
            viewpoint = Camera.init_from_dataset(
                self.dataset, cur_frame_idx, projection_matrix)
            
            self.cameras[cur_frame_idx] = viewpoint

            
            if cur_frame_idx == 0:
                self.viewpoints[cur_frame_idx] = viewpoint
                depth_preprocess = self.initialize(cur_frame_idx, viewpoint, self.config['Training']['use_droid_pose'])
                self.add_new_gaussians(
                    cur_frame_idx, viewpoint, depth_map=depth_preprocess, init=True)
                
                self.current_keyframe_window.append(cur_frame_idx)
            
                current_keyframe_window_dict = {}
                current_keyframe_window_dict[self.current_keyframe_window[0]] = self.current_keyframe_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_keyframe_window]
                
                if self.use_gui:
                    gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
                    gui_process.start()
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_keyframe_window_dict,
                        )
                    )
                    time.sleep(1.5)
                self.initialize_mapping(cur_frame_idx, viewpoint)
                if self.use_gui:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_keyframe_window_dict,
                        )
                    )
                
            else:
                current_keyframe_window_dict = {}
                current_keyframe_window_dict[self.current_keyframe_window[0]] = self.current_keyframe_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_keyframe_window]
                
                style = get_style("GS-SLAM")
                if self.config['Training']['prev_pose']:
                    print(f"[{style}]Frame[/{style}] {cur_frame_idx}: \n    [bold red]tracking: assuming that the current pose is the one from the previous frame..[/bold red]")
                elif self.config['Training']['use_droid_pose']:
                    print(f"[{style}]Frame[/{style}] {cur_frame_idx}: \n    [bold red]tracking: droid-slam estimated pose..[/bold red]")
                
                elif self.config['Training']['forward_pose'] and cur_frame_idx >= 2:
                    print(f"[{style}]Frame[/{style}] {cur_frame_idx}: \n    [bold red]tracking: assuming constant motion..[/bold red]")
                     
                render_pkg = self.tracking(cur_frame_idx, viewpoint, keyframes)
                
                if self.config['Eval']['eval_tracking'] and \
                        cur_frame_idx > self.config['Eval']['eval_ate_after'] and \
                        cur_frame_idx % self.config['Eval']['eval_ate_every'] == 0:                           
                    eval_all  = self.config['Eval']['eval_ate_all']
                    eval_ate_tracking(self.cameras, self.kf_indices, eval_all=eval_all, correct_scale=False, fig_plot = fig)

                last_keyframe_idx = self.current_keyframe_window[0]
                visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(viewpoint, cur_frame_idx, last_keyframe_idx,visibility,self.keyframe_visibility,
                                             self.config['Dataset']['use_droid_keyframe'],
                                             self.kf_indices_droid)
                    
                if create_kf:
                    # self.current_keyframe_window: current keyframe windows
                    self.current_keyframe_window, _ = self.update_keyframe_window(
                        cur_frame_idx,
                        visibility,
                        self.keyframe_visibility,
                        self.current_keyframe_window,
                    )
                    self.viewpoints[cur_frame_idx] = viewpoint
                    depth_map = self.preprocess(cur_frame_idx, init=False)
                    
                    # self.current_keyframe_window: dict {current window: [keyframe windows without current window:]}
                    current_keyframe_window_dict = {}
                    current_keyframe_window_dict[self.current_keyframe_window[0]] = self.current_keyframe_window[1:]
                    
                    # keyframes: extract the camera infos of keyframes
                    keyframes = [self.cameras[kf_idx] for kf_idx in self.current_keyframe_window]
                    
                    mapping = True
                else:
                    if self.config['Dataset']['save_only_kf'] and cur_frame_idx !=0:
                        self.cameras[cur_frame_idx].depth = None
                        self.cameras[cur_frame_idx].original_image = None
                    
                    
            # mapping
            
            if mapping:
                if self.use_gui:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_keyframe_window_dict,
                        )
                    )
                self.viewpoints[cur_frame_idx] = viewpoint
                self.add_new_gaussians(cur_frame_idx, viewpoint, depth_map=depth_map)
                frames_to_optimize = self.config["Training"]["pose_window"]
                iter_per_kf = self.mapping_params.mapping_itr_num
                opt_params = []
                
                for cam_idx in range(len(self.current_keyframe_window)):
                    if self.current_keyframe_window[cam_idx] == 0:
                        continue
                    viewpoint = self.viewpoints[self.current_keyframe_window[cam_idx]]
                    if cam_idx < frames_to_optimize:
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_rot_delta],
                                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                                "name": "rot_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_trans_delta],
                                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                                "name": "trans_{}".format(viewpoint.uid),
                            }
                        )
                    
                self.keyframe_optimizers = torch.optim.Adam(opt_params)
                self.keyframe_mapping(iter_per_kf, render_uncertainty=False)

                if self.config['Eval']['eval_tracking'] and \
                    cur_frame_idx > self.config['Eval']['eval_ate_after'] :
                    eval_ate_tracking(self.cameras, self.kf_indices, 
                                      eval_all=self.config['Eval']['eval_ate_all'], correct_scale=False, fig_plot = fig)
                
                if self.config['Eval']['eval_mapping']:
                    eval_rendering_mapping(
                        self.cameras,
                        1,
                        self.gaussians,
                        self.dataset,
                        self.pipeline_params,
                        self.background,
                        self.kf_indices,
                        cal_lpips=cal_lpips)
                mapping = False
                    
           
        end.record()
        plt.close()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")
 
        self.final_optimization(render_uncertainty=False)
        
        eval_ate(
            self.cameras,
            self.kf_indices,
            self.save_dir,
            0,
            final=True,
            correct_scale=False,
        )

        eval_rendering(
                self.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=self.kf_indices,
                iteration="final",
                projection_matrix=self.projection_matrix
        )
        
        save_gaussians(self.gaussians, self.save_dir, "final", final=True)

        if self.use_gui:
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_keyframe_window_dict,
                    )
                )
        
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")
        
        Log("Processing finished !")   
            
    '''
    ***************************************************************************************************
    ***************************************************************************************************
    @Function
    ***************************************************************************************************
    ***************************************************************************************************
    '''

    def run(self):
        pass
    
    def set_tracking_params(self):
        self.tracking_params = munchify(dict())
        self.tracking_params.save_dir = self.config["Results"]["save_dir"]
        self.tracking_params.save_results = self.config["Results"]["save_results"]
        self.tracking_params.save_trj = self.config["Results"]["save_trj"]
        self.tracking_params.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_params.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.tracking_params.tracking_itr_num_droid = self.config["Training"]["tracking_itr_num_droid"]
        self.tracking_params.kf_max_interval = self.config["Training"]["kf_max_interval"]
        self.tracking_params.window_size = self.config["Training"]["window_size"]
    
    
    def set_mapping_params(self):
        self.mapping_params = munchify(dict())
        self.mapping_params.save_results = self.config["Results"]["save_results"]

        self.mapping_params.init_itr_num = self.config["Training"]["init_itr_num"]
        self.mapping_params.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.mapping_params.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.mapping_params.init_min_opacity = self.config["Training"]["init_min_opacity"]
        
        self.mapping_params.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.mapping_params.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.mapping_params.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.mapping_params.gaussian_th = self.config["Training"]["gaussian_th"]
        self.mapping_params.init_gaussian_extent = self.config["Training"]["init_gaussian_extent"]
        self.mapping_params.gaussian_extent = self.config["Training"]["gaussian_extent"]
        
        self.mapping_params.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.mapping_params.size_threshold = self.config["Training"]["size_threshold"]
        self.mapping_params.window_size = self.config["Training"]["window_size"]
        self.mapping_params.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
    

    
    def initialize(self, cur_frame_idx, viewpoint, use_droid_pose):
        # self.kf_indices = []
        self.mapping_iter_count = 0
        self.keyframe_visibility = {}
        self.current_keyframe_window = []

        # Initialise the frame at the droid pose
        if not use_droid_pose:
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        else:
            viewpoint.update_RT(viewpoint.R_droid, viewpoint.T_droid)

        self.kf_indices_droid = self.dataset.kf_indices
        
        depth_preprocess = self.preprocess(cur_frame_idx, init=True)
        return depth_preprocess
        
    def preprocess(self, cur_frame_idx, init=False):
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        return depth[0].numpy()
    
    def add_new_gaussians(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map, 
            render=render,
            pipeline_params = self.pipeline_params, 
            background = self.background,
        )
        
    def initialize_mapping(self, cur_frame_idx, viewpoint):
        Log("Map initialization: ")
        for mapping_iteration in tqdm(range(self.mapping_params.init_itr_num)):
            self.mapping_iter_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            
            (
                image, depth,
                viewspace_point_tensor,visibility_filter,
                radii,opacity,n_touched) = (
                render_pkg["render"],render_pkg["depth"],
                render_pkg["viewspace_points"],render_pkg["visibility_filter"],
                render_pkg["radii"],render_pkg["opacity"],render_pkg["n_touched"],
            )
                
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.mapping_params.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.mapping_params.init_min_opacity,
                        self.mapping_params.init_gaussian_extent,
                        None,
                    )

                if self.mapping_iter_count == self.mapping_params.init_gaussian_reset or (
                    self.mapping_iter_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                
        self.keyframe_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Map initialization finished !")
        return render_pkg
    
    def tracking(self, cur_frame_idx, viewpoint, keyframes=None, render_uncertainty=False):
        if self.config['Training']['prev_pose']:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            # current pos = prev
            viewpoint.update_RT(prev.R, prev.T)
            tracking_itr_num = self.tracking_params.tracking_itr_num
            
        elif self.config['Training']['use_droid_pose']:
            viewpoint.update_RT(viewpoint.R_droid, viewpoint.T_droid)
            tracking_itr_num = self.tracking_params.tracking_itr_num_droid
            
        elif self.config['Training']['forward_pose'] and cur_frame_idx >= 2:
            prev1 = self.cameras[cur_frame_idx - 1]
            prev2 = self.cameras[cur_frame_idx - 2]
            delta = prev1.world_view_transform.transpose(0,1) @ prev2.world_view_transform.transpose(0,1).inverse()
            curr = delta @ prev1.world_view_transform.transpose(0,1)
            R = curr[:3, :3]
            T = curr[:3, 3]
            viewpoint.update_RT(R, T)
            tracking_itr_num = self.tracking_params.tracking_itr_num_droid
            
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            # current pos = prev
            viewpoint.update_RT(prev.R, prev.T)
            Log(f"Frame {cur_frame_idx}: tracking: assuming that the current pose is the one from the previous frame..")
            tracking_itr_num = self.tracking_params.tracking_itr_num
            
        if render_uncertainty:
            viewpoint.viewmatrix = viewpoint.world_view_transform.detach()
            viewpoint.viewmatrix = nn.Parameter(viewpoint.viewmatrix)
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        
        # if render_uncertainty:
        #     opt_params.append(
        #         {
        #             "params": [viewpoint.viewmatrix],
        #             "lr": 0.001,
        #             "name": "trans_{}".format(viewpoint.uid),
        #         }
        #     )
        
        with torch.no_grad():
            render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            # if self.tracking_loss != 0:
            #     loss_tracking_mean = self.tracking_loss/self.tracking_iter
            #     loss_difference = loss_tracking - loss_tracking_mean
            #     if loss_difference > 0:
            #         ratio = min(torch.exp(loss_difference / loss_tracking_mean), 10)
            #         tracking_itr_num = int(tracking_itr_num * ratio)

        
        pose_optimizer = torch.optim.Adam(opt_params)
        t = tqdm(range(tracking_itr_num))
        for tracking_itr in t:
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )

            self.tracking_iter += 1
            self.tracking_loss += loss_tracking.item()
            
            loss_tracking.backward()
 
            with torch.no_grad():
                pose_optimizer.step()
                if render_uncertainty:
                    viewpoint.R = (viewpoint.viewmatrix.transpose(0,1)[:3, :3])
                    viewpoint.T = (viewpoint.viewmatrix.transpose(0,1)[:3, 3])
                converged = update_pose(viewpoint)

            
            if tracking_itr % int(max(tracking_itr_num, 6)/5) == 0 and self.use_gui:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        keyframes = keyframes,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                    )
                )
            t.set_description(f"        tracking loss = {loss_tracking.item()}")
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        torch.cuda.empty_cache()
        return render_pkg
    
    def is_keyframe(self, viewpoint, cur_frame_idx,last_keyframe_idx,cur_frame_visibility_filter,keyframe_visibility,
                    use_droid_keyframe = False,
                    kf_indices_droid = None):
        
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]
        depth_error_percent_th = self.config["Training"]["depth_error_percent"]
        depth_error_th = self.config["Training"]["depth_error_th"]
        
        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        with torch.no_grad():
            depth = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )["depth"]
            gt_depth = torch.from_numpy(viewpoint.depth).to(
                dtype=torch.float32, device= "cuda"
            )[None]
            depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
            
            depth_error = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
            # depth_error =  torch.where(gt_depth * depth_pixel_mask != 0, depth_error / (gt_depth*depth_pixel_mask), torch.tensor(0.0).cuda())
            depth_error_percent =  (depth_error > depth_error_th).sum() / (gt_depth > 0.01).sum()

            
        union = torch.logical_or(
            cur_frame_visibility_filter, keyframe_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, keyframe_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union    
        if use_droid_keyframe:
            if cur_frame_idx in kf_indices_droid:
                print("[bold bright_magenta]    keyframe detected based on droid slam[/bold bright_magenta]")
                return True
        if (point_ratio_2 < kf_overlap and dist_check2):
            print("[bold bright_magenta]    keyframe detected because of low overlapping[/bold bright_magenta]")
            return True
        elif depth_error_percent > depth_error_percent_th:
            print("[bold bright_magenta]    keyframe detected because of large depth error[/bold bright_magenta]")
            return True
        elif dist_check:
            print("[bold bright_magenta]    keyframe detected because of large translation[/bold bright_magenta]")
            return True
        elif cur_frame_idx > self.kf_indices[-1] + self.config['Training']['kf_max_interval']:
            print("[bold bright_magenta]    keyframe detected because of large interval[/bold bright_magenta]")
            return True
        else:
            return False     

    def update_keyframe_window(
        self, cur_frame_idx, cur_frame_visibility_filter, keyframe_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]

            intersection = torch.logical_and(
                cur_frame_visibility_filter, keyframe_visibility[kf_idx]
            ).count_nonzero()
            
            union = torch.logical_or(
                cur_frame_visibility_filter, keyframe_visibility[kf_idx]
            ).count_nonzero()
            
            ratio = intersection / union
            remove_threshold = self.config["Training"]["kf_cutoff"]

            if ratio <= remove_threshold:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
            
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
   
    def keyframe_mapping(self, iter_per_kf, render_uncertainty=False):
        if len(self.current_keyframe_window) == 0:
            return
        
        frames_to_optimize = self.config["Training"]["pose_window"]
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in self.current_keyframe_window]
        random_viewpoint_stack = []
        n_seen = torch.zeros([self.gaussians.get_xyz.shape[0]]).cuda()

        current_window_set = set(self.current_keyframe_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        t = tqdm(range(iter_per_kf),desc="mapping...")
        for iter in t:
            self.mapping_iter_count += 1
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(self.current_keyframe_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )

                # viewspace_point_tensor: used to compute the accumulated gradient to every Gaussian in the visible frustum
                # radii: radius of each 2D gaussian on the screen
                # visibility_filter: radii > 0, visibility of each gaussian
                # opacity: opacity[0-1] of each pixel
                # n_touched: how many pixels touched each Gaussian
                
                (image,viewspace_point_tensor,visibility_filter,radii,depth,opacity,n_touched) = (
                    render_pkg["render"],render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],render_pkg["radii"],
                    render_pkg["depth"],render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                
                
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                
                
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                
                # normal gaussian splatting
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (image,viewspace_point_tensor,visibility_filter,radii,depth,opacity,n_touched) = (
                    render_pkg["render"],render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],render_pkg["radii"],
                    render_pkg["depth"],render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                
                
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

            scaling = self.gaussians.get_scaling
            loss_mapping += self.config['Training']['mapping_isotropic_loss_ratio']*get_isotropic_loss_2(scaling)
            t.set_description(f"        mapping loss = {loss_mapping.item()}")
            
            loss_mapping.backward()            
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.keyframe_visibility = {}
                for idx in range((len(self.current_keyframe_window))):
                    kf_idx = self.current_keyframe_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.keyframe_visibility[kf_idx] = (n_touched > 0).long()

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.mapping_iter_count % self.mapping_params.gaussian_update_every
                    == self.mapping_params.gaussian_update_offset and iter != (iter_per_kf-1)
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.mapping_params.gaussian_th,
                        self.mapping_params.gaussian_extent,
                        self.mapping_params.size_threshold,
                    )

                ## Opacity reset
                if (self.mapping_iter_count % self.mapping_params.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.mapping_iter_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                for cam_idx in range(min(frames_to_optimize, len(self.current_keyframe_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

    def final_optimization(self, render_uncertainty = False):

        Log("Final global optimization ..")
        opt_params = []
        for viewpoint in self.viewpoints.values():
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(viewpoint.uid),
                }
            )
        self.keyframe_optimizers = torch.optim.Adam(opt_params)
        t = tqdm(range(1, self.opt_params.iterations + 1))
        viewpoint_idx_stack = None
        ema_loss_for_log = 0.0
        dep_loss_for_log = 0.0
        for iteration in t:
            
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii, viewspace_point_tensor,depth = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["viewspace_points"],
                render_pkg["depth"],
            )

            loss = get_loss_mapping(config, image, depth, viewpoint_cam, None, final_opt=True)
            scaling = self.gaussians.get_scaling
            loss += self.config['opt_params']['isotropic_loss_ratio']*get_isotropic_loss_1(scaling)
            loss.backward()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                t.set_description(f"        Loss: {ema_loss_for_log}")

            if iteration % 100 == 0:
                if self.use_gui:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians)
                    )
            )
            
            with torch.no_grad():
                if iteration < self.opt_params.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                            radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > self.opt_params.densify_from_iter and iteration % self.opt_params.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt_params.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt_params.densify_grad_threshold, 
                                                        0.005, self.mapping_params.gaussian_extent, size_threshold)
                    if iteration % self.opt_params.opacity_reset_interval == 0:
                        self.gaussians.reset_opacity()
                    
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
        Log("[blod red]global optimization done..[/blod red]") 
    
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    
    args = parser.parse_args(sys.argv[1:])
    
    mp.set_start_method("spawn")
    
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)
    
    config = load_config(args.config)  
    
    if config["Results"]["save_results"]:
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-2]
        )
        mkdir_p(save_dir)
        config["save_dir"] = save_dir
        n = args.config.split("/")
        shutil.copy(config['inherit_from'], os.path.join(save_dir, n[-1]))

    slam = SLAM(config, save_dir=save_dir)

    slam.run()

    # All done
    Log("Done.")
