import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap
from scipy.spatial.transform import Rotation

from .utils.read_write_model import read_model
from .utils.parsers import (
    SubQuery, parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair, parse_generalized_queries)
from .utils.query import *

from typing import List

def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(
                j for i in observed if i != -1 for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def pose_from_cluster(query, retrieval_ids, db_images, points3D,
                      feature_file, match_file, thresh):
    camera_dicts = []
    rel_camera_poses = []
    for subquery in query:
        qname = subquery.name
        #qinfo = subquery.info
        kpq = feature_file[qname]['keypoints'].__array__()
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        num_matches = 0
        db_ids = retrieval_ids[qname]
        for i, db_id in enumerate(db_ids):
            db_name = db_images[db_id].name
            points3D_ids = db_images[db_id].point3D_ids

            pair = names_to_pair(qname, db_name)
            matches = match_file[pair]['matches0'].__array__()
            valid = np.where(matches > -1)[0]
            valid = valid[points3D_ids[matches[valid]] != -1]
            num_matches += len(valid)

            for idx in valid:
                id_3D = points3D_ids[matches[idx]]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)

        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mkpq = kpq[mkp_idxs]
        mkpq += 0.5  # COLMAP coordinates

        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        mp3d = [points3D[j].xyz for j in mp3d_ids]
        mp3d = np.array(mp3d).reshape(-1, 3)

        # mostly for logging and post-processing
        mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                           for i in idxs for j in kp_idx_to_3D[i]]

        #camera_model, width, height, params = qinfo
        camera_dicts.append({
            'model': subquery.camera_model,
            'width': subquery.width,
            'height': subquery.height,
            'params': subquery.params,
        })
        subquery.mkpq = mkpq  # 2d
        subquery.mp3d = mp3d  # 3d
        subquery.num_matches = len(mkpq)
        subquery.mp3d_ids = mp3d_ids
        subquery.mkp_idxs = mkp_idxs
        subquery.mkp_to_3D_to_db = mkp_to_3D_to_db
        R = Rotation.from_quat(subquery.extrinsics[:4]).as_matrix()

        t = subquery.extrinsics[-3:]
        rel_camera_poses.append(np.append(R,np.atleast_2d(t).transpose(),axis=1)) 

    mkpq = np.concatenate([sq.mkpq for sq in query])
    mp3d = np.concatenate([sq.mp3d for sq in query])
    cam_idxs = np.concatenate([cam_idx*np.ones(query[cam_idx].num_matches,dtype=int)
            for cam_idx in range(len(query))])
    
    #for obj in [mkpq, mp3d, cam_idxs,
    #        rel_camera_poses, camera_dicts]:
    #    print(type(obj[0]))
    print(cam_idxs)

    np.savetxt("mkpq.csv", mkpq, delimiter=" ")
    np.savetxt("mp3d.csv", mp3d, delimiter=" ")
    np.savetxt("cam_idxs.csv", cam_idxs, delimiter=" ")

    ret = pycolmap.generalized_absolute_pose_estimation(mkpq, mp3d, cam_idxs,
            rel_camera_poses, camera_dicts, max_error_px=thresh)
    return ret


def main(queries: List[QueryFrameSequence], 
        features: Path, 
        matches: Path, 
        results: Path,
        ransac_thresh=12, covisibility_clustering=False):

    assert features.exists(), features
    assert matches.exists(), matches

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'loc': {},
    }
    logging.info('Starting localization...')
    for query in tqdm(queries):
        qname, qinfo, subqueries

        ret = pose_from_cluster(
            subqueries, retrieval_ids, db_images, points3D,
            feature_file, match_file, thresh=ransac_thresh)
        # logging.info(f'# inliers: {ret["num_inliers"]}')

        if ret['success']:
            poses[qname] = (ret['qvec'], ret['tvec'])
        else:
            closest = retrieval_names[subqueries[0].name]
            poses[qname] = (closest.qvec, closest.tvec)
        logs['loc'][qname] = {
            'db': retrieval_ids,
            # 'PnP_ret': ret,
            # 'keypoints_query': mkpq,
            # 'points3D_xyz': mp3d,
            # 'points3D_ids': mp3d_ids,
            # 'num_matches': num_matches,
            # 'keypoint_index_to_db': map_,
        }
    
    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            f.write(f'{name} {qvec} {tvec}\n')
