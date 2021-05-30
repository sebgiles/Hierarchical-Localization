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

    def __repr__(self):
        return self.filename


class QueryFrame:
    def __init__(self, images: List[QueryImage]):
        images.sort(key=lambda x: x.camera_id)
        check_unique([x.camera_id for x in images])
        self.images = images
        self.time_us = images[0].time_us
        self.time_ms = images[0].time_ms
        self.date = images[0].date
        # self.time_ms = np.mean([x.time_ms for x in images])
        # self.time_us = np.mean([x.time_us for x in images])
        self.time_delta_us = (
            max([x.time_us for x in images]) - min([x.time_us for x in images]))
        self.frame_number = images[0].frame_number
        self.size = len(images)

    def __repr__(self):
        date_string = self.date.strftime("%Y-%m-%d")
        out_string = f"Frame {date_string} #{self.frame_number:05} {{\n"
        return out_string + "\n".join([f"\t{x}" for x in self.images]) + "\n}"


class QueryFrameSequence:
    def __init__(self, frames: List[QueryFrame]):
        assert len(frames) > 1
        frames.sort(key=lambda x: x.time_ms)
        self.date = frames[0].date
        self.start_ms = frames[0].time_ms
        self.end_ms = frames[-1].time_ms
        self.size = len(frames)
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
        return (x.images[cam].filename for x in self.frames)
        
class QueryDatabase:
    def __init__(self, images_path: Path):
        image_names = None
        if images_path.is_file():
            # Take images with valid retrieval
            image_names = list(parse_retrieval(images_path).keys())
        else:
            image_names = os.listdir(images_path)
        self.images = [QueryImage(x) for x in image_names]
        self.frames = self.__to_frames(self.images)
        self.sequences = self.__to_sequences(self.frames)


    # NOTE: only returns pairs for now
    # n is how many queries you want to generate
    # step_size is number of time steps (~100ms) between frames
    def get_sequence_queries(self, query_count=1, step_size=1, step_count=2,
        seed=None):
        random.seed(seed)
        step_span = step_count*step_size
        valid_sequences = []
        for seq in self.sequences:
            if seq.size > step_span:
                valid_sequences.append(seq)
        out = []
        for i in range(query_count):
            # NOTE: distribution is not uniform
            seq_number = random.randint(0, len(valid_sequences)-1)
            seq = valid_sequences[seq_number]
            frame_number = random.randint(0, seq.size-step_span-1)
            frames = seq.frames[frame_number:frame_number+step_span:step_size]
            out.append(QueryFrameSequence(frames))
        # Remove duplicates (should be replaces tp satisfy query_count)
        out.sort(key=lambda x: x.start_ms)
        i = 1
        while i < len(out):
            if out[i].start_ms == out[i-1].start_ms:
                del out[i]
            else:
                i += 1 

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
            # if frame.time_us - sequence[-1].time_us > max_delta_ms*1000:
            if frame.frame_number != sequence[-1].frame_number + 1:
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

    def write_pairs_for_matching(self, output_file: Path, overwrite=False):
        assert overwrite or not output_file.exists()
        lines = []
        for seq in self.sequences:
            for f0, f1 in zip(seq.frames[:-1], seq.frames[1:]):
                # NOTE: we only take one image from the rigs
                lines.append(
                    f"{f0.images[0].filename} {f1.images[0].filename}\n")
                lines.append(
                    f"{f1.images[0].filename} {f0.images[0].filename}\n")

        with open(output_file, 'w') as f:
            f.writelines(lines)
