import math
import torch
from gaussian_splatting import Camera, GaussianModel
from .diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings


def feature_fusion(self: GaussianModel, viewpoint_camera: Camera, feature_map: torch.Tensor, fusion_alpha_threshold: float = 0.):
    '''Copy of the forward method from gaussian_splatting.GaussianModel, only change the rasterizer'''
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device=self._xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=viewpoint_camera.bg_color.to(self._xyz.device),
        scale_modifier=self.scale_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=self.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=self.debug,
        antialiasing=self.antialiasing,
        fusion_alpha_threshold=fusion_alpha_threshold,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = self.get_xyz
    means2D = screenspace_points
    opacity = self.get_opacity

    scales = self.get_scaling
    rotations = self.get_rotation

    shs = self.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth_image, features, features_alpha, pixhit, features_idx = rasterizer(
        feature_map=feature_map,
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)
    rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }
    return out, features, features_alpha, pixhit, features_idx
