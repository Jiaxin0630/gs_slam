import torch
from tqdm import tqdm
from torch import nn
from gui import gui_utils
from gaussian_splatting.gaussian_renderer import render
from gaussian_renderer_unc import render_unc
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.logging_utils import Log
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 



def tracking(self, cur_frame_idx, viewpoint, keyframes=None, render_uncertainty=False):
    if self.configs['Training']['prev_pose']:
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
    
    if render_uncertainty:
        opt_params.append(
            {
                "params": [viewpoint.viewmatrix],
                "lr": 0.001,
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
    
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
        if self.tracking_loss != 0:
            loss_tracking_mean = self.tracking_loss/self.tracking_iter
            loss_difference = loss_tracking - loss_tracking_mean
            if loss_difference > 0:
                ratio = min(torch.exp(loss_difference / loss_tracking_mean), 10)
                tracking_itr_num = int(tracking_itr_num * ratio)

    
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
        
        if render_uncertainty:
            render_unc_pkg = render_unc(viewpoint, self.gaussians, self.pipeline_params, self.background, 
                        viewpoint.viewmatrix,viewpoint.projection_matrix,
                        torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=image.device).unsqueeze(0),
                        scaling_modifier=1.0,override_color=None,
                        mask=None,track_off=False,map_off=True)
            
            (image, depth, opacity, depth_median, depth_var) = (
                render_unc_pkg["render"],render_unc_pkg["depth"],render_unc_pkg["opacity"],
                render_unc_pkg["depth_median"],render_unc_pkg["depth_var"]
            )
        
            loss_tracking + get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking += 0.05*torch.abs(depth_var).sum() / (depth_var.shape[1] * depth_var.shape[2] )
            self.tracking_iter += 1
            self.tracking_loss += loss_tracking.item()

        self.tracking_iter += 1
        self.tracking_loss += loss_tracking.item()
        
        loss_tracking.backward()

        with torch.no_grad():
            pose_optimizer.step()
            if render_uncertainty:
                viewpoint.R = (viewpoint.viewmatrix.transpose(0,1)[:3, :3])
                viewpoint.T = (viewpoint.viewmatrix.transpose(0,1)[:3, 3])
            converged = update_pose(viewpoint)

        if tracking_itr % int(tracking_itr_num/5) == 0:
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