"""
Copy from SIGA2025VVC-Dataset/render_ftgs.py
"""
import os
from os.path import dirname, exists, join
import argparse
from argparse import Namespace
import numpy as np
import cv2


class DotDict(dict):
    def __init__(self, mapping=None, /, **kwargs):
        if mapping is None:
            mapping = {}
        elif type(mapping) is Namespace:
            mapping = vars(mapping)

        super().__init__(mapping, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if type(value) is dict:
                value = DotDict(value)
            return value
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<DotDict " + dict.__repr__(self) + ">"

    def todict(self):
        return {k: v for k, v in self.items()}


dotdict = DotDict


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split(".")[0])
        self.second_version = int(version.split(".")[1])

        if isWrite:
            os.makedirs(dirname(filename), exist_ok=True)
            self.fs = open(filename, "w")
            self.fs.write("%YAML:1.0\r\n")
            self.fs.write("---\r\n")
        else:
            assert exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        try:
            if self.isWrite:
                self.fs.close()
            else:
                cv2.FileStorage.release(self.fs)
        except Exception:
            pass

    def _write(self, out):
        self.fs.write(out + "\r\n")

    def write(self, key, value, dt="mat"):
        if dt == "mat":
            self._write("{}: !!opencv-matrix".format(key))
            self._write("  rows: {}".format(value.shape[0]))
            self._write("  cols: {}".format(value.shape[1]))
            self._write("  dt: d")
            self._write("  data: [{}]".format(", ".join(["{:.10f}".format(i) for i in value.reshape(-1)])))
        elif dt == "list":
            self._write("{}:".format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == "real":
            if isinstance(value, np.ndarray):
                value = value.item()
            self._write("{}: {:.10f}".format(key, value))  # as accurate as possible
        else:
            raise NotImplementedError

    def read(self, key, dt="mat"):
        if dt == "mat":
            output = self.fs.getNode(key).mat()
        elif dt == "list":
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == "":
                    val = str(int(n.at(i).real()))
                if val != "none":
                    results.append(val)
            output = results
        elif dt == "real":
            output = self.fs.getNode(key).real()
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera_new(data_root, intri_name="intri.yml", extri_name="extri.yml"):
    return read_camera(join(data_root, intri_name), join(data_root, extri_name))


def read_camera(intri_path: str, extri_path: str = None, cam_names=[]) -> dotdict:
    if extri_path is None:
        extri_path = join(intri_path, "extri.yml")
        intri_path = join(intri_path, "intri.yml")
    assert exists(intri_path), intri_path
    assert exists(extri_path), extri_path

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cams = dotdict()
    cam_names = sorted(intri.read("names", dt="list"))
    for cam in cam_names:
        # Intrinsics
        cams[cam] = dotdict()
        cams[cam].K = intri.read("K_{}".format(cam))
        cams[cam].H = int(intri.read("H_{}".format(cam), dt="real")) or -1
        cams[cam].W = int(intri.read("W_{}".format(cam), dt="real")) or -1
        cams[cam].invK = np.linalg.inv(cams[cam]["K"])

        # Extrinsics
        Tvec = extri.read("T_{}".format(cam))
        Rvec = extri.read("R_{}".format(cam))
        if Rvec is not None:
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read("Rot_{}".format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam].R = R
        cams[cam].T = Tvec
        cams[cam].C = -R.T @ Tvec.squeeze()
        cams[cam].RT = RT
        cams[cam].Rvec = Rvec
        cams[cam].P = cams[cam].K @ cams[cam].RT

        # Distortion
        D = intri.read("D_{}".format(cam))
        if D is None:
            D = intri.read("dist_{}".format(cam))
        cams[cam].D = D

        # Time input
        cams[cam].t = extri.read("t_{}".format(cam), dt="real") or 0  # temporal index, might all be 0
        cams[cam].v = extri.read("v_{}".format(cam), dt="real") or 0  # temporal index, might all be 0

        # Bounds, could be overwritten
        cams[cam].n = extri.read("n_{}".format(cam), dt="real") or 0.0001  # temporal index, might all be 0
        cams[cam].f = extri.read("f_{}".format(cam), dt="real") or 1e6  # temporal index, might all be 0
        cams[cam].bounds = extri.read("bounds_{}".format(cam))
        cams[cam].bounds = (
            np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if cams[cam].bounds is None else cams[cam].bounds
        )

        # CCM
        cams[cam].ccm = intri.read("ccm_{}".format(cam))
        cams[cam].ccm = np.eye(3) if cams[cam].ccm is None else cams[cam].ccm

    return dotdict(cams)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the video folder")

if __name__ == "__main__":
    args = parser.parse_args()
    cameras = read_camera_new(args.path, intri_name="train_intri.yml", extri_name="train_extri.yml")
    print(cameras)
