from threading import setprofile
import numpy as np
from .parsers import parse_retrieval
from pathlib import Path
import os
import random
from typing import List, Sequence
import datetime


CAMERAS = [
    {
        'model': 'OPENCV',
        'width': 1024,
        'height': 768,
        'params': np.array([ 868.993378, 866.063001, 525.942323, 420.042529,
                            -0.399431, 0.188924, 0.000153, 0.000571 ]),
        'extrinsics': np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]
        ])
    },
    {
        'model': 'OPENCV',
        'width': 1024,
        'height': 768,
        'params': np.array([873.382641, 876.489513, 529.324138, 397.272397,
                             -0.397066,   0.181925,   0.000176,  -0.000579]),
        'extrinsics': np.array([
            [ 0.02879946,  0.02008295, -0.99938344, -1.30507738],
            [-0.00422425,  0.99979167,  0.01996942,  0.08334538],
            [ 0.99957628,  0.00364653,  0.0288783 , -1.23681136]
        ])
    },
]


def check_unique(values: list):
    values.sort()
    for e1, e2 in zip(values[:-1],values[1:]):
        assert e1 != e2


class QueryImage:
    def __init__(self, query: str):
        filename = query.split(' ')[0].strip()
        self.filename = filename
        self.time_us = int(filename[13:-6])
        self.time_ms = int(filename[13:-9])
        self.frame_number = int(filename[4:9])
        self.sequence_number = None
        self.camera_id = int(filename[11])
        self.date = datetime.datetime.fromtimestamp(self.time_us/1e6)
        self.mkpq = None
        self.string_with_intrinsics = None

    def __repr__(self):
        return self.filename


class QueryFrame:
    def __init__(self, image_list: List[QueryImage]):
        image_list.sort(key=lambda x: x.camera_id)
        self.cams = [x.camera_id for x in image_list]
        check_unique(self.cams)
        self.images = {img.camera_id: img for img in image_list}
        self.time_us = image_list[0].time_us
        self.time_ms = image_list[0].time_ms
        self.date = image_list[0].date
        # self.time_ms = np.mean([x.time_ms for x in image_list])
        # self.time_us = np.mean([x.time_us for x in image_list])
        self.time_delta_us = (
            max([x.time_us for x in image_list]) - min([x.time_us for x in image_list]))
        self.frame_number = image_list[0].frame_number
        self.size = len(image_list)

    def __repr__(self):
        date_string = self.date.strftime("%Y-%m-%d")
        out_string = f"Frame {date_string} #{self.frame_number:05} {{\n"
        return out_string + "\n".join([f"\t{x}" for x in self.images.values()]) + "\n}"


class QueryFrameSequence:
    def __init__(self, frames: List[QueryFrame]):
        assert len(frames) > 1
        assert all([frames[1:]])
        frames.sort(key=lambda x: x.time_ms)
        self.date = frames[0].date
        self.start_ms = frames[0].time_ms
        self.end_ms = frames[-1].time_ms
        self.size = len(frames)
        self.cams = frames[0].cams
        self.duration_ms = self.end_ms - self.start_ms
        self.period_avg_ms = self.duration_ms / (self.size - 1)
        #self.period_std_ms = np.std(np.diff([x.time_ms for x in frames]))
        self.frames = frames

    def __repr__(self):
        text = (
            f'{self.date}\t' +
            f'duration: {(self.duration_ms/1e3):.1f}s\t' +
            f'{self.size:} frames\t' +
            f'period: {self.period_avg_ms:.1f} ms'
        )
        return text

    def get_image_names(self, cam=0):
        return [x.images[cam].filename for x in self.frames]

class QueryDatabase:
    def __init__(self, images_path: Path):
        image_names = None
        strings_with_instrinsics = None
        if images_path.is_file():
            
            with open(images_path, 'r') as f:
                strings_with_instrinsics = f.readlines()
            image_names = [s.split()[0] for s in strings_with_instrinsics]
        else:
            image_names = os.listdir(images_path)
        self.images = [QueryImage(x) for x in image_names]
        if strings_with_instrinsics:
            for image, swi in zip(self.images, strings_with_instrinsics):
                image.string_with_intrinsics = swi.strip()
        self.frames = self.__to_frames(self.images)
        self.sequences = self.__to_sequences(self.frames)


    def get_sequence_queries(self, step_size: int, max_size:int = None):
        out = []
        for seq in self.sequences:
            for cam in seq.cams:
                if len(seq.frames) < step_size: 
                    continue
                for i, f0 in enumerate(seq.frames[:-step_size]):
                    f1 = seq.frames[i+step_size]
                    qfs = QueryFrameSequence([f0, f1])
                    out.append(qfs)
        if max_size is not None:
            out = out[:max_size]
        return SequenceQuerySet(out)


    # n is how many queries you want to generate
    # step_size is number of time steps (~100ms) between frames
    def get_random_sequence_queries(self, query_count=1, step_size=1, step_count=2, 
        required_cams=[], seed=None):
        random.seed(seed)
        step_span = step_count*step_size
        valid_sequences = []
        for seq in self.sequences:
            if seq.size > step_span:
                valid_sequences.append(seq)
        out = []
        while len(out) < query_count:
            # NOTE: distribution is not uniform
            seq_number = random.randint(0, len(valid_sequences)-1)
            seq = valid_sequences[seq_number]
            frame_number = random.randint(0, seq.size-step_span-1)
            frames = seq.frames[frame_number:frame_number+step_span:step_size]
            qfs = QueryFrameSequence(frames)
            duplicate = any([qfs.start_ms == x.start_ms for x in out])
            has_all_cams = all([cam in qfs.cams for cam in required_cams])
            if has_all_cams and not duplicate:
                out.append(qfs)

        return SequenceQuerySet(out)

    @classmethod
    def __to_frames(cls, images: List[QueryImage],
                    max_delta_us=200) -> List[QueryFrame]:
        frames = []
        # Assume images in the same frame are adjacent after sorting by time
        images.sort(key=lambda x: x.time_us)
        frame = [images[0]]
        for image in images[1:]:
            if frame[0].frame_number != image.frame_number:
                f = QueryFrame(frame)
                # NOTE: Discards whole frame if images are out of sync
                if abs(f.time_delta_us) < max_delta_us:
                    frames.append(f)
                frame = []
            frame.append(image)
        frames.append(QueryFrame(frame))
        return frames

    # From inspection "normal" capture period appears to be 65 to 135 ms
    # this is measured between average timestamps of the images in each frame.
    # We discard short sequences and those with unstable frame rate.
    @classmethod
    def __to_sequences(cls, frames: QueryFrame, max_delta_ms=200,
                        min_seq_duration_ms=1000) -> List[QueryFrameSequence]:
        sequences = []
        frames.sort(key=lambda x: x.time_us)
        sequence = [frames[0]]
        for frame in frames[1:]:
            interrupt = (
                frame.time_us - sequence[-1].time_us > max_delta_ms*1000
                or frame.frame_number != sequence[-1].frame_number + 1
                or frame.cams != sequence[-1].cams
            )
            if interrupt:
                if len(sequence) > 1:
                    qfs = QueryFrameSequence(sequence)
                    if qfs.duration_ms >= min_seq_duration_ms:
                        sequences.append(qfs)
                sequence = []
            sequence.append(frame)
        if len(sequence) > 1:
            qfs = QueryFrameSequence(sequence)
            if qfs.duration_ms >= min_seq_duration_ms:
                sequences.append(qfs)

        return sequences

    def __repr__(self):
        text_dict = {
            '#images': len(self.images),
            '#frames': len(self.frames),
            '#single frames':
                np.sum(np.array([x.size for x in self.frames])==1),
            '#sequences': len(self.sequences),
        }
        return repr(text_dict)


class SequenceQuerySet:
    def __init__(self, sequence_queries: List[QueryFrameSequence]):
        self.sequences = sequence_queries

    def write_pairs_for_matching(self, output_file: Path, overwrite=False, cam_ids:int=None):
        assert overwrite or not output_file.exists()
        lines = []
        for seq in self.sequences:
            for f0, f1 in zip(seq.frames[:-1], seq.frames[1:]):
                cams = cam_ids or seq.cams
                for cam_id in cams:
                    lines.append(
                        f"{f0.images[cam_id].filename} {f1.images[cam_id].filename}\n")
                    lines.append(
                        f"{f1.images[cam_id].filename} {f0.images[cam_id].filename}\n")

        with open(output_file, 'w') as f:
            f.writelines(lines)
