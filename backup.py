def keyframe_mapping(self, frames_to_optimize, iter_per_kf):
        if len(self.current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
        random_viewpoint_stack = []
        
        n_seen = torch.zeros([self.gaussians.get_xyz.shape[0]]).cuda()
        
        scale_grad_acm = torch.zeros_like(self.gaussians.get_scaling)
        scaling_old = None
        
        current_window_set = set(self.current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for iter in tqdm(range(iter_per_kf),desc="mapping..."):
            self.mapping_iter_count += 1
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(self.current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (image,viewspace_point_tensor,visibility_filter,radii,depth,opacity,n_touched) = (
                    render_pkg["render"],render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],render_pkg["radii"],
                    render_pkg["depth"],render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                    
                n_seen[n_touched>0] += 1
                
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (image,viewspace_point_tensor,visibility_filter,radii,depth,opacity,n_touched) = (
                    render_pkg["render"],render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],render_pkg["radii"],
                    render_pkg["depth"],render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                
                n_seen[n_touched>0] += 1
                
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            
            if scaling_old is not None:
                ratio = scale_grad_acm[n_touched>0] / n_seen[n_touched>0].unsqueeze(-1)
                loss_mapping += (ratio * torch.abs(scaling_old[n_touched>0] - self.gaussians._scaling[n_touched>0])).sum()
            
            scaling_old =  self.gaussians._scaling.detach()
            
           
            loss_mapping.backward()
            scale_grad_acm += torch.norm(self.gaussians._scaling.grad,dim=-1,keepdim=True)
            
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(self.current_window))):
                    kf_idx = self.current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

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
                    == self.mapping_params.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.mapping_params.gaussian_th,
                        self.mapping_params.gaussian_extent,
                        self.mapping_params.size_threshold,
                    )
                    n_seen = torch.zeros([self.gaussians.get_xyz.shape[0]]).cuda()
                    scale_grad_acm = torch.zeros_like(self.gaussians.get_scaling)
                    scaling_old = None

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