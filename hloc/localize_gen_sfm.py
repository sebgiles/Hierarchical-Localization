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

from .utils.read_write_model import read_model, qvec2rotmat, rotmat2qvec
from .utils.parsers import (
    SubQuery, parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair, parse_generalized_queries)


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

    # Uncomment to save to file for COLMAP unit test
    # np.savetxt("mkpq.csv", mkpq, delimiter=" ")
    # np.savetxt("mp3d.csv", mp3d, delimiter=" ")
    # np.savetxt("cam_idxs.csv", cam_idxs, delimiter=" ")

    ret = pycolmap.generalized_absolute_pose_estimation(mkpq, mp3d, cam_idxs,
            rel_camera_poses, camera_dicts, max_error_px=thresh)
    return ret


def main(reference_sfm, queries, retrieval, features, matches, results,
         ransac_thresh=12, covisibility_clustering=False):

    assert reference_sfm.exists(), reference_sfm
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_generalized_queries(queries)
    retrieval_dict = parse_retrieval(retrieval)

    logging.info('Reading 3D model...')
    _, db_images, points3D = read_model(str(reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')
    for qname, qinfo, subqueries in tqdm(queries):
        retrieval_names = {sq.name:retrieval_dict[sq.name] for sq in subqueries}
        retrieval_ids = {}
        for sq_name in retrieval_names:
            sq_retrieval_ids = []
            for name in retrieval_names[sq_name]:
                if name not in db_name_to_id:
                    logging.warning(f'Image {n} was retrieved but not in database')
                    continue
                sq_retrieval_ids.append(db_name_to_id[name])
            retrieval_ids[sq_name] = sq_retrieval_ids

        ret = pose_from_cluster(
            subqueries, retrieval_ids, db_images, points3D,
            feature_file, match_file, thresh=ransac_thresh)
        # logging.info(f'# inliers: {ret["num_inliers"]}')

        if ret['success']:
            for i in range(len(subqueries)):
                # messy way to do this, need to find a better way

                q_10 = ret['rig_qvecs'][i]
                t_10 = ret['rig_tvecs'][i]
                r_10 = qvec2rotmat(q_10)
                r_01 = np.transpose(r_10)
                t_01 = -np.matmul(r_01, t_10)

                r0 = qvec2rotmat(ret['qvec'])
                t0 = ret['tvec']
                t1 = t0 - t_01

                r_1w = np.matmul(r_10, r0)
                t_1w = np.matmul(r_10, t1)
                q_1w = rotmat2qvec(r_1w)
                poses[subqueries[i].name] = (q_1w, t_1w)

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

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
