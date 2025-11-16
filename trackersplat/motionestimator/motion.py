from typing import NamedTuple
import torch


class Motion(NamedTuple):
    fixed_mask: torch.Tensor = None
    motion_mask_cov: torch.Tensor = None
    motion_mask_mean: torch.Tensor = None
    rotation_quaternion: torch.Tensor = None
    scaling_modifier_log: torch.Tensor = None
    translation_vector: torch.Tensor = None
    confidence_fix: torch.Tensor = None
    confidence_cov: torch.Tensor = None
    confidence_mean: torch.Tensor = None
    update_baseframe: bool = False

    opacity_modifier_log: torch.Tensor = None
    features_dc_modifier: torch.Tensor = None
    features_rest_modifier: torch.Tensor = None

    def to(self, device: torch.device) -> 'Motion':
        return self._replace(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self._asdict().items()})

    def validate(self):
        if self.fixed_mask is not None:
            assert self.fixed_mask.dtype == torch.bool and self.fixed_mask.dim() == 1
            if self.confidence_fix is not None:
                assert self.confidence_fix.dim() == 1 and self.confidence_fix.size(0) == self.fixed_mask.sum()
        else:
            assert self.confidence_fix is None

        if self.motion_mask_cov is not None:
            assert self.motion_mask_cov.dtype == torch.bool and self.motion_mask_cov.dim() == 1
            if self.confidence_cov is not None:
                assert self.confidence_cov.dim() == 1 and self.confidence_cov.size(0) == self.motion_mask_cov.sum()
            if self.rotation_quaternion is not None:
                assert self.rotation_quaternion.dim() == 2 and self.rotation_quaternion.size(0) == self.motion_mask_cov.sum() and self.rotation_quaternion.size(1) == 4
            elif self.scaling_modifier_log is not None:
                assert self.scaling_modifier_log.dim() == 2 and self.scaling_modifier_log.size(0) == self.motion_mask_cov.sum() and self.scaling_modifier_log.size(1) == 3
        else:
            assert self.confidence_cov is None

        if self.motion_mask_mean is not None:
            assert self.motion_mask_mean.dtype == torch.bool and self.motion_mask_mean.dim() == 1
            if self.confidence_mean is not None:
                assert self.confidence_mean.dim() == 1 and self.confidence_mean.size(0) == self.motion_mask_mean.sum()
            if self.translation_vector is not None:
                assert self.translation_vector.dim() == 2 and self.translation_vector.size(0) == self.motion_mask_mean.sum() and self.translation_vector.size(1) == 3
        else:
            assert self.confidence_mean is None
