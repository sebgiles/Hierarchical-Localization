from pathlib import Path
import logging
import numpy as np
from collections import defaultdict


def parse_image_with_intrinsics(data):
    assert len(data) == 12
    name, camera_model, width, height = data[:4]
    intrinsics = np.array(data[4:12], float)
    info = (camera_model, int(width), int(height), intrinsics)
    return (name, info, )    


def parse_image_with_intrinsics_and_extrinsics(data):
    name, info = parse_image_with_intrinsics(data[:12])
    extrinsics = np.array(data[12:19], float)
    return (name, info, extrinsics)   

# identity quaternion and null translation
IDENTITY_EXTRINSICS = np.array([1,0,0,0,0,0,0], float)

def parse_generalized_query(fields):
    name, info_0 = parse_image_with_intrinsics(fields[:12])
    images = [(name, info_0, IDENTITY_EXTRINSICS,),]
    done = 12
    while len(fields) > done:
        assert len(fields) >= done + 19
        image = parse_image_with_intrinsics_and_extrinsics(fields[done:])
        images.append(image)
        done += 19
        
    # first image gives the query name
    # info_0 is kept for compatibility with the single view pipeline
    return (name, info_0, images)


def parse_generalized_queries(paths):
    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

        logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
        for data in raw_data:
            data = data.strip('\n ').split(' ')
            query = parse_generalized_query(data)
            results.append(query)

    assert len(results) > 0
    return results


def parse_image_lists_with_intrinsics(paths):
    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

        logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
        for data in raw_data:
            data = data.strip("\n ").split(' ')
            results.append(parse_image_with_intrinsics(data))

    assert len(results) > 0
    return results


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split(' ')
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))
