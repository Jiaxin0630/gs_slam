import json
import os
import sys
import cv2
import evo
import numpy as np
import torch
from utils.camera_utils import Camera
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision
import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.math_utils import qvec2rotmat, rotmat2qvec
from utils.slam_utils import l1dep_loss

def evaluate_evo(poses_gt, poses_est, plot_dir, label, correct_scale=False, trj_data = None):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=correct_scale
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    
    with open(
        os.path.join(plot_dir, "trajectory_est_ate_metric.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)
        
        
    ape_error_np = np.array(ape_metric.get_result().np_arrays['error_array'])
    np.savetxt(os.path.join(plot_dir, "trajectory_est_ate.txt"), ape_error_np)
    Log("RMSE ATE \[cm]", ape_stat*100, tag="Eval")

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat*100} [cm]")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "trajectory_est.png"), dpi=200)
    plt.close()
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(trj_data['trj_id'], ape_error_np*100, '-o', color='blue', linewidth=1.5, 
             markersize=2, markerfacecolor='orange',markeredgecolor='orange')
    plt.xlabel('Frame id')
    plt.ylabel('[cm]')
    plt.title(f"ATE RMSE: {ape_stat*100} [cm]", fontsize=15, color="blue")
    plt.savefig(os.path.join(plot_dir, "trajectory_ate.png"), dpi=200)
    plt.close()
    return ape_stat

def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.detach().cpu().numpy()
        pose[0:3, 3] = T.detach().cpu().numpy()
        return pose


def evaluate_evo_tracking(poses_gt, poses_est, correct_scale=False, fig=None):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=correct_scale
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[cm]", ape_stat*100, tag="    tracking results")

    plot_mode = evo.tools.plot.PlotMode.xy
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat*100} cm")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "black", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()

    return ape_stat

def eval_ate_tracking(frames, kf_ids, eval_all, correct_scale=True, fig_plot=None):
    fig= fig_plot
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    if eval_all or (not eval_all and len(kf_ids) < 5):
        for cam in frames.values():
            pose_est = np.linalg.inv(gen_pose_matrix(cam.R, cam.T))
            pose_gt = np.linalg.inv(gen_pose_matrix(cam.R_gt, cam.T_gt))
            trj_id.append(cam.uid)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)
    else:
        for kf_id in kf_ids:
            cam = frames[kf_id]
            pose_est = np.linalg.inv(gen_pose_matrix(cam.R, cam.T))
            pose_gt = np.linalg.inv(gen_pose_matrix(cam.R_gt, cam.T_gt))
            trj_id.append(cam.uid)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)
            

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    ate = evaluate_evo_tracking(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        correct_scale=correct_scale,
        fig=fig
    )
    
    plt.draw()
    plt.pause(0.1)
    plt.clf()
    return ate

def eval_ate(frames, kf_ids, save_dir, iterations, final=False, correct_scale=False):
    Log("start evaluating ate..")
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []
    pose_est_np = []
    for cam in frames.values():
        pose_est = np.linalg.inv(gen_pose_matrix(cam.R, cam.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(cam.R_gt, cam.T_gt))

        trj_id.append(cam.uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)
        
        quat = rotmat2qvec(cam.R.detach().clone().cpu().numpy())
        quat = np.append(quat[1:], quat[0])
        pose_est_np.append(np.append(np.append(cam.uid, cam.T.detach().clone().cpu().numpy()), quat))
        

    pose_est_np = np.array(pose_est_np)
    np.savetxt(os.path.join(save_dir, "trajectory_est_w2c.txt"), pose_est_np,  fmt='%.10f')
    
    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        correct_scale=correct_scale,
        trj_data = trj_data
    )
    return ate



def eval_rendering_mapping(
    frames,
    interval,
    gaussians,
    dataset,
    pipe,
    background,
    kf_indices,
    cal_lpips = None
):
    img_pred, img_gt, saved_frame_idx = [], [], []
    psnr_array, ssim_array, lpips_array = [], [], []
    for idx in kf_indices[::interval]:
        frame = frames[idx]
        gt_image, _, _,_ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_ssim"]}', tag="    mapping results")


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    projection_matrix = None
):
    Log("start evaluating rendering quality..")
    with torch.no_grad():
        plt.close()
        interval = 1
        save_dir = os.path.join(save_dir, "rendering")
        depth_path = os.path.join(save_dir, "depth")
        render_path = os.path.join(save_dir, "render")
        if not depth_path:
            os.makedirs(depth_path)
        if not render_path:
            os.makedirs(render_path)
        saved_frame_idx, cam_uid = [], []
        end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
        psnr_array, ssim_array, lpips_array, l1dep_loss_array = [], [], [], []
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")
        
        mkdir_p(render_path)
        for idx in range(0, end_idx, interval):
            saved_frame_idx.append(idx)
            
            frame = frames[idx]
            cam_uid.append(frame.uid)
            gt_image, gt_depth, _,_ = dataset[idx]

            results = render(frame, gaussians, pipe, background)
            rendering = results["render"]
            image = torch.clamp(rendering, 0.0, 1.0)
            
            depth = results["depth"]
            
            gt_depth = torch.from_numpy(gt_depth).to(
                dtype=torch.float32, device=rendering.device)[None]
            
            depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
            
            L1_depth_loss = l1dep_loss(depth*depth_pixel_mask, gt_depth*depth_pixel_mask)
            
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            
            gt_depth_scaled = gt_depth / (depth.max() + 1e-5)
            depth_scaled = depth / (depth.max() + 1e-5)
        
            gt_depth_normalized = cv2.normalize(gt_depth.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            gt_depth_colorized = cv2.applyColorMap(gt_depth_normalized, cv2.COLORMAP_VIRIDIS)

            depth_normalized = cv2.normalize(depth.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            
            
            mask = gt_image > 0

            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
            lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

            psnr_array.append(psnr_score.item())
            ssim_array.append(ssim_score.item())
            lpips_array.append(lpips_score.item())
            l1dep_loss_array.append(L1_depth_loss.item())
            
            if iteration == "final":
                cv2.imwrite(f"{depth_path}/{idx:05d}.png", (depth_colorized))
                out1 = torch.cat((gt_image, rendering), dim=2)
                out2 = torch.cat((torch.from_numpy(gt_depth_colorized).permute(2,0,1).cuda()/ 255.0, 
                                  torch.from_numpy(depth_colorized).permute(2,0,1).cuda()/ 255.0), dim=2)
                out = torch.cat((out1, out2), dim=1)
                torchvision.utils.save_image(out, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                
                

        images_to_video(render_path, os.path.join(save_dir, "render.mp4"))
        
        output = dict()
        output["mean_psnr"] = float(np.mean(psnr_array))
        output["mean_ssim"] = float(np.mean(ssim_array))
        output["mean_lpips"] = float(np.mean(lpips_array))
        output["mean_l1dep_loss"] = float(np.mean(l1dep_loss_array))

        Log(
            f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, mean_l1dep_loss[cm]: {output["mean_l1dep_loss"]*100}', tag="Eval",
        )

        psnr_np = np.array(psnr_array)
        ssim_np = np.array(ssim_array)
        lpips_np = np.array(lpips_array)
        l1dep_loss_np = np.array(l1dep_loss_array)
        cam_uid_np = np.array(cam_uid)
        
        with open(os.path.join(save_dir, "render_metric_array.txt"), 'w') as f:
            f.write("PSNR SSIM LPIPS L1Depth[cm]\n")
            np.savetxt(f, np.column_stack((
                psnr_np, ssim_np, lpips_np, 100*l1dep_loss_np)), fmt='%0.6f')
            
        fig, axs = plt.subplots(4, 1, figsize=(30, 16))
        axs[0].plot(cam_uid_np, psnr_np,color="darkviolet",linewidth=1.5)
        axs[0].set_title('psnr',fontsize=25,color="darkviolet",fontweight='bold')
        axs[0].grid(True, which='both', linestyle=':', linewidth=0.5, color='green')

        axs[1].plot(cam_uid_np, ssim_np,color="dodgerblue",linewidth=1.5)
        axs[1].set_title('ssim',fontsize=25,color="dodgerblue",fontweight='bold')
        axs[1].grid(True, which='both', linestyle=':', linewidth=0.5, color='green')
        
        axs[2].plot(cam_uid_np, lpips_np,color="crimson",linewidth=1.5)
        axs[2].set_title('lpips',fontsize=25,color="crimson",fontweight='bold')
        axs[2].grid(True, which='both', linestyle=':', linewidth=0.5, color='green')
        
        axs[3].plot(cam_uid_np, l1dep_loss_np*100,color="orangered",linewidth=1.5)
        axs[3].set_title('L1 depth loss[cm]',fontsize=25,color="orangered",fontweight='bold')
        axs[3].grid(True, which='both', linestyle=':', linewidth=0.5, color='green')
        plt.subplots_adjust(hspace=0.3)
        axs[3].set_xlabel("frame")
        plt.savefig(os.path.join(save_dir, "render_metric.png"), dpi=300)
        
        json.dump(
            output,
            open(os.path.join(save_dir, "render_metric.json"), "w", encoding="utf-8"),
            indent=4,
        )
        return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply_all(os.path.join(point_cloud_path, "point_cloud.ply"))
    
def images_to_video(image_folder, video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    images.sort() 

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print("Failed to read the first image.")
        return

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        if img is None:
            print(f"Failed to read image {image}. Skipping.")
            continue
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

