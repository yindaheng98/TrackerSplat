from abc import ABCMeta, abstractmethod
from gaussian_splatting import GaussianModel
from trackersplat import Motion, MotionCompensater


class PatchCompensater(MotionCompensater, metaclass=ABCMeta):

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = super().compensate(baseframe, motion)
        return self.patch(currframe)

    @abstractmethod
    def patch(self, gaussians: GaussianModel) -> GaussianModel:
        pass
