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
from .utils.pose import Transform

from .utils.read_write_model import read_model
from .utils.parsers import (
    SubQuery, parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair, parse_generalized_queries)
from .utils.query import *

from typing import List


def get_points_3d(feature_file: h5py.File, match_file: h5py.File,
                        db_images, db_points, retr_ids, img_name):

    kpq = feature_file[img_name]['keypoints'].__array__()
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(retr_ids):
        db_name = db_images[db_id].name
        points3D_ids = db_images[db_id].point3D_ids

        pair = names_to_pair(img_name, db_name)
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
    mp3d = [db_points[j].xyz for j in mp3d_ids]
    mp3d = np.array(mp3d).reshape(-1, 3)

    return mp3d, mkpq


def get_points_2d(features: h5py.File, matches: h5py.File,
                    img0_name: str, img1_name: str, n_best: int=10000):
    pair = matches[names_to_pair(img0_name, img1_name)]
    matches01 = pair['matches0'].__array__()
    ranking = -pair['matching_scores0'].__array__().argsort()
    matches01 = matches01[ranking]
    matched = (matches01>-1)
    features0 = features[img0_name]['keypoints'].__array__()[ranking]
    features1 = features[img1_name]['keypoints'].__array__()
    mkpq0 = features0[matched][:n_best]
    mkpq1 = features1[matches01[matched]][:n_best]
    mkpq0 += 0.5  # to COLMAP coordinates
    mkpq1 += 0.5  # to COLMAP coordinates
    return mkpq0, mkpq1


def main(   reference_sfm,
            features: Path,
            matches_absolute: Path,
            matches_relative: Path,
            retrieval: Path,
            queries: List[QueryFrameSequence],
            results: Path,
            ransac_thresh=12,
            covisibility_clustering=False,
            overwrite=False):

    assert reference_sfm.exists(), reference_sfm
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches_absolute.exists(), matches_absolute
    assert matches_relative.exists(), matches_relative

    retrieval_dict = parse_retrieval(retrieval)

    logging.info('Reading 3D model...')
    _, db_images, points3D = read_model(str(reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(features, 'r')
    abs_match_file = h5py.File(matches_absolute, 'r')
    rel_match_file = h5py.File(matches_relative, 'r')

    poses = {}
    output_file = h5py.File(str(results), 'a')

    logging.info('Starting localization...')
    for query in tqdm(queries):
        cam_id = query.cams[0]
        qimages = [q.images[cam_id] for q in query.frames]
        qnames = [img.filename for img in qimages]
        qname = qnames[0]
        camera = CAMERAS[cam_id]

        if qname in output_file:
            if overwrite:
                del output_file[qname]
            else:
                continue

        retrieval_names = {nm: retrieval_dict[nm] for nm in qnames}
        retrieval_ids = {nm:
            [db_name_to_id[retr_name] for retr_name in retrieval_dict[nm]]
                for nm in qnames}

        rel1_points2D_0, rel0_points2D_1 = get_points_2d(
            features=feature_file,
            matches=rel_match_file,
            img0_name=qnames[0],
            img1_name=qnames[1]
        )

        points3D_0, map_points2D_0 = get_points_3d(
            feature_file=feature_file,
            match_file=abs_match_file,
            db_images=db_images,
            db_points=points3D,
            retr_ids=retrieval_ids[qnames[0]],
            img_name=qnames[0],
        )

        points3D_1, map_points2D_1 = get_points_3d(
            feature_file=feature_file,
            match_file=abs_match_file,
            db_images=db_images,
            db_points=points3D,
            retr_ids=retrieval_ids[qnames[1]],
            img_name=qnames[1],
        )

        ret = pycolmap.sequence_pose_estimation(
            points3D_0,
            points3D_1,
            map_points2D_0,
            map_points2D_1,
            rel1_points2D_0,
            rel0_points2D_1,
            camera,
            max_error_px = 12.0
        )

        if ret['success']:
            poses[qname] = (ret['qvec_0'], ret['tvec_0'])
            R_CG = Rotation.from_quat(np.roll(ret['qvec_0'],-1))
            C_G = -R_CG.inv().apply(ret['tvec_0'])
            T_GC = Transform(t=C_G, q=R_CG.inv().as_quat())
            grp = output_file.create_group(qname)
            grp.create_dataset('q', data=T_GC.R.as_quat())
            grp.create_dataset('t', data=T_GC.t)

    output_file.close()
    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
