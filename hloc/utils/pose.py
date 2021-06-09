#from __future__ import annotations
from os import X_OK
from typing import List
from pathlib import Path
from collections import namedtuple
from scipy.spatial.transform import Rotation
import numpy as np

class Transform:
    R: Rotation
    t: np.array

    def __init__(self, t, q, quat_convention='xyzw'):
        self.t = np.array(t, dtype=float)
        q = np.array(q, dtype=float)
        if quat_convention == 'xyzw':
            pass
        elif quat_convention == 'wxyz':
            q = np.roll(q,-1)
        else:
            raise("Unsupported quaternion format: ", quat_convention)
        q = q * np.sign(q[-1])
        self.R = Rotation.from_quat(q)
        
    def inv(self):
        R = self.R.inv()
        t = -R.apply(self.t)
        return Transform(t, R.as_quat(), quat_convention='xyzw')
    
    def as3by4(self):
        R = self.R.as_matrix()
        return np.concatenate((R, self.t[:,None]),axis=1)
        
    def __mul__(self, other):
        t = self.t + self.R.apply(other.t)
        R =  self.R * other.R
        return Transform(t, R.as_quat(), quat_convention='xyzw')
        
    def __repr__(self):
        return "\t".join([str(x) for x in [self.R.as_quat(), self.t]])

    def normalize(self):
        return UnscaledTransform(t=self.t, q=self.R.as_quat())
    
    def distance_to(self, other):
        t_err = np.linalg.norm(self.t - other.t)
        R_err = (self.R*other.R.inv()).magnitude()
        return t_err, np.rad2deg(R_err)

class UnscaledTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = self.t / np.linalg.norm(self.t)

    def distance_to(self, other):
        t_err = np.arccos(np.clip(np.dot(self.t, other.t), -1.0, 1.0))
        R_err = (self.R*other.R.inv()).magnitude()
        return np.rad2deg(t_err), np.rad2deg(R_err)

class PoseDatabase:
    def __init__(self, sources: List[Path]):
        lines = []
        for path in sources:
            
            if path.is_dir():
                for file in path.glob('*.txt'):
                    lines += file.open().readlines()
            else:
                 lines += path.open().readlines()
        self.poses = {'ground': Transform(t=[0,0,0], q=[0,0,0,1])}
        for line in lines:
            fields = line.strip().split()
            image_name = fields[0]
            R_CW = Rotation.from_quat(np.roll(fields[1:5],-1))
            C_W = np.array(fields[5:8])
            T_WC = Transform(t=C_W, q=R_CW.inv().as_quat())
            self.poses[image_name] = T_WC

    def relative_pose(self, k0, k1):
        p0 = self.poses[k0]
        p1 = self.poses[k1]
        return p0.inv() * p1

    def relative_poses(self, k0, k1):
        p0 = [self.poses[x] for x in k0]
        p1 = [self.poses[x] for x in k1]
        return [x.inv() * y for x,y in zip(p0,p1)]

    def closest_in(self, database, output_file:Path=None):
        sfm_k = list(database.poses.keys())[1:]
        sfm_t = np.stack([pose.t for pose in list(database.poses.values())[1:]])
        sfm_R = np.stack([pose.R for pose in list(database.poses.values())[1:]])
        retrieval = {}
        for key in self.poses:
            if key == 'ground': continue
            t = self.poses[key].t
            R = self.poses[key].R
            dist = np.linalg.norm(sfm_t - t, axis=(1,))
            top = np.argsort(dist)[:30]
            rot_err = np.array([rot_err.magnitude() for rot_err in sfm_R[top] * R.inv()])
            top = top[rot_err < 1.0][:10]
            retrieval[key] = [sfm_k[t] for t in top]
        if output_file:
            pairs = []
            for key in retrieval:
                for found in retrieval[key]:
                    pairs.append(f"{key} {found}\n")
            with open(output_file, 'w') as file:
                file.writelines(pairs)
            return pairs
        else:
            return retrieval