import os
import sys
import shutil
import time
import yaml
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from rich import print
from argparse import ArgumentParser
from gui import gui_utils,slam_gui
from gs_slam.slam import SLAM
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.camera import Camera
from utils.config import load_config
from utils.logging import Log, get_style
from utils.multiprocessing import clone_obj
from utils.eval import eval_ate, eval_ate_tracking, eval_rendering, save_gaussians,eval_rendering_mapping

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
        
    mapping = False    
    plt.ion()
    fig= plt.figure()
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True,weights="alex"
        ).to("cuda")
    
    # run 3DGS-SLAM
    for cur_frame_idx in range(0, slam.dataset.num_imgs):
        viewpoint = Camera.init_from_dataset(
            slam.dataset, cur_frame_idx, slam.projection_matrix)
        slam.cameras[cur_frame_idx] = viewpoint

        # tracking
        if cur_frame_idx == 0:
            slam.viewpoints[cur_frame_idx] = viewpoint
            
            slam.initialize(cur_frame_idx, viewpoint, slam.config['Training']['use_droid_pose'])
            
            slam.add_new_gaussians(
                cur_frame_idx, viewpoint, depth_map=viewpoint.depth, init=True)
            
            slam.current_keyframe_window.append(cur_frame_idx)
            current_keyframe_window_dict = {}
            current_keyframe_window_dict[0] = []
            keyframes = [slam.cameras[0]]
            
            if slam.use_gui:
                gui_process = mp.Process(target=slam_gui.run, args=(slam.params_gui,))
                gui_process.start()
                slam.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(slam.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_keyframe_window_dict,
                    )
                )
                time.sleep(1.5)
            
            slam.initialize_mapping(cur_frame_idx, viewpoint)
            
            if slam.use_gui:
                slam.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(slam.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_keyframe_window_dict,
                    )
                )
            
        else:
            current_keyframe_window_dict = {}
            current_keyframe_window_dict[slam.current_keyframe_window[0]] = slam.current_keyframe_window[1:]
            keyframes = [slam.cameras[kf_idx] for kf_idx in slam.current_keyframe_window]
            
            style = get_style("GS-SLAM")
            if slam.config['Training']['prev_pose']:
                print(f"[{style}]Frame[/{style}] {cur_frame_idx}: \n    [bold red]tracking: assuming that the current pose is the one from the previous frame..[/bold red]")
            elif slam.config['Training']['use_droid_pose']:
                print(f"[{style}]Frame[/{style}] {cur_frame_idx}: \n    [bold red]tracking: droid-slam estimated pose..[/bold red]")
            
            elif slam.config['Training']['forward_pose'] and cur_frame_idx >= 2:
                print(f"[{style}]Frame[/{style}] {cur_frame_idx}: \n    [bold red]tracking: assuming constant motion..[/bold red]")
                    
            render_pkg = slam.tracking(cur_frame_idx, viewpoint, keyframes)
            
            if slam.config['Eval']['eval_tracking'] and \
                    cur_frame_idx > slam.config['Eval']['eval_ate_after'] and \
                    cur_frame_idx % slam.config['Eval']['eval_ate_every'] == 0:                           
                eval_all  = slam.config['Eval']['eval_ate_all']
                eval_ate_tracking(slam.cameras, slam.kf_indices, eval_all=eval_all, correct_scale=False, fig_plot = fig)

            last_keyframe_idx = slam.current_keyframe_window[0]
            visibility = (render_pkg["n_touched"] > 0).long()
            create_kf = slam.is_keyframe(viewpoint, cur_frame_idx, last_keyframe_idx,visibility,slam.keyframe_visibility,
                                            slam.config['Dataset']['use_droid_keyframe'],
                                            slam.kf_indices_droid)
                
            if create_kf:
                # slam.current_keyframe_window: current keyframe windows
                slam.current_keyframe_window, _ = slam.update_keyframe_window(
                    viewpoint,
                    cur_frame_idx,
                    visibility,
                    slam.keyframe_visibility,
                    slam.current_keyframe_window,
                )
                slam.viewpoints[cur_frame_idx] = viewpoint
                slam.kf_indices.append(cur_frame_idx)
                
                # slam.current_keyframe_window: dict {current window: [keyframe windows without current window:]}
                current_keyframe_window_dict = {}
                current_keyframe_window_dict[slam.current_keyframe_window[0]] = slam.current_keyframe_window[1:]
                
                # keyframes: extract the camera infos of keyframes
                keyframes = [slam.cameras[kf_idx] for kf_idx in slam.current_keyframe_window]
                
                mapping = True
            else:
                if slam.config['Dataset']['save_only_kf'] and cur_frame_idx !=0:
                    slam.cameras[cur_frame_idx].depth = None
                    slam.cameras[cur_frame_idx].original_image = None
                
                
        # mapping
        if mapping:
            if slam.use_gui:
                slam.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(slam.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_keyframe_window_dict,
                    )
                )
            slam.viewpoints[cur_frame_idx] = viewpoint
            slam.add_new_gaussians(cur_frame_idx, viewpoint, depth_map=viewpoint.depth)
            frames_to_optimize = slam.config["Training"]["pose_window"]
            iter_per_kf = slam.mapping_params.mapping_itr_num
            opt_params = []
            
            for cam_idx in range(len(slam.current_keyframe_window)):
                if slam.current_keyframe_window[cam_idx] == 0:
                    continue
                viewpoint = slam.viewpoints[slam.current_keyframe_window[cam_idx]]
                if cam_idx < frames_to_optimize:
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_rot_delta],
                            "lr": slam.config["Training"]["lr"]["cam_rot_delta"],
                            "name": "rot_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_trans_delta],
                            "lr": slam.config["Training"]["lr"]["cam_trans_delta"],
                            "name": "trans_{}".format(viewpoint.uid),
                        }
                    )
                
            slam.keyframe_optimizers = torch.optim.Adam(opt_params)
            slam.keyframe_mapping(iter_per_kf, render_uncertainty=False)

            if slam.config['Eval']['eval_tracking'] and \
                cur_frame_idx > slam.config['Eval']['eval_ate_after'] :
                eval_ate_tracking(slam.cameras, slam.kf_indices, 
                                    eval_all=slam.config['Eval']['eval_ate_all'], correct_scale=False, fig_plot = fig)
            
            if slam.config['Eval']['eval_mapping']:
                eval_rendering_mapping(
                    slam.cameras,
                    1,
                    slam.gaussians,
                    slam.dataset,
                    slam.pipeline_params,
                    slam.background,
                    slam.kf_indices,
                    cal_lpips=cal_lpips)
            mapping = False
            
    end.record()
    plt.close()
    torch.cuda.synchronize()
    
    FPS = len(slam.cameras) / (start.elapsed_time(end) * 0.001)
    Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
    Log("Total FPS", FPS, tag="Eval")

    slam.final_optimization(render_uncertainty=False)
    
    eval_ate(
        slam.cameras,
        slam.kf_indices,
        slam.save_dir,
        0,
        final=True,
        correct_scale=False,
    )

    eval_rendering(
            slam.cameras,
            slam.gaussians,
            slam.dataset,
            slam.save_dir,
            slam.pipeline_params,
            slam.background,
            kf_indices=slam.kf_indices,
            iteration="final",
            projection_matrix=slam.projection_matrix
    )
    
    save_gaussians(slam.gaussians, slam.save_dir, "final", final=True)

    if slam.use_gui:
        slam.q_main2vis.put(
            gui_utils.GaussianPacket(
                    gaussians=clone_obj(slam.gaussians),
                    current_frame=viewpoint,
                    keyframes=keyframes,
                    kf_window=current_keyframe_window_dict,
                )
            )
        
        slam.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
        gui_process.join()
        Log("GUI Stopped and joined the main thread")
        
    Log("Processing finished !")   
    






