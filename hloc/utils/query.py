import numpy as np
from .parsers import parse_retrieval
from pathlib import Path
import os
import random
from typing import List, Sequence

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
        self.mkpq = None

    def __repr__(self):
        text_dict = {
            'frame': self.frame_number,
            'time [ms]':self.time_ms,
            'cam id': self.camera_id,
            'file': self.filename,
        }
        return repr(text_dict)


class QueryFrame:
    def __init__(self, images: List[QueryImage]):
        images.sort(key=lambda x: x.camera_id)
        check_unique([x.camera_id for x in images])
        self.images = images
        self.time_us = images[0].time_us
        self.time_ms = images[0].time_ms
        # self.time_ms = np.mean([x.time_ms for x in images])
        # self.time_us = np.mean([x.time_us for x in images])
        self.time_delta_us = (
            max([x.time_us for x in images]) - min([x.time_us for x in images]))
        self.frame_number = images[0].frame_number
        self.sequence_number = images[0].sequence_number
        self.size = len(images)

    def __repr__(self):
        return " ".join([f"{x.frame_number}" for x in self.images])


class QueryFrameSequence:
    def __init__(self, frames: List[QueryFrame]):
        assert len(frames) > 1
        frames.sort(key=lambda x: x.time_ms)
        self.start_ms = frames[0].time_ms
        self.end_ms = frames[-1].time_ms
        self.frames = frames
        self.size = len(frames)
        self.duration_ms = self.end_ms - self.start_ms
        self.period_avg_ms = self.duration_ms / (self.size - 1)
        self.period_std_ms = np.std(np.diff([x.time_ms for x in frames]))

    def __repr__(self):
        text_dict = {
            'start [ms]':self.start_ms,
            'duration [ms]': self.duration_ms,
            'frame count': self.size,
            'period avg [ms]': self.period_avg_ms,
            'period std [ms]': self.period_std_ms,
        }
        return repr(text_dict)


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
    def get_sequence_queries(self, n=1, gap=1, seed=None):
        random.seed(seed)
        valid_sequences = []
        for seq in self.sequences:
            if seq.size > gap:
                valid_sequences.append(seq)
        out = []
        for i in range(n):
            # NOTE: distribution is not uniform
            seq_number = random.randint(0, len(valid_sequences)-1)
            seq = valid_sequences[seq_number]
            frame_number = random.randint(0, seq.size-gap-1)
            out.append(QueryFrameSequence([ seq.frames[frame_number],
                                            seq.frames[frame_number+gap] ]))
        # Remove duplicates
        out.sort(key=lambda x: x.start_ms)
        i = 1
        while i < len(out):
            if out[i].start_ms == out[i-1].start_ms:
                del out[i]
            else:
                i += 1 

        return SequenceQuerySet(out)

    # We discard frames where the images are out of sync
    @classmethod
    def __to_frames(cls, images: List[QueryImage], 
                    max_delta_us=100) -> List[QueryFrame]:
        frames = []
        images.sort(key=lambda x: x.time_us)
        frame = [images[0]]
        for image in images[1:]:
            if frame[0].frame_number != image.frame_number:
                f = QueryFrame(frame)
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
    def __to_sequences(cls, frames: QueryFrame, max_delta_ms=150, 
                        min_seq_duration_ms=1000) -> List[QueryFrameSequence]:
        sequences = []
        frames.sort(key=lambda x: x.time_us)
        sequence = [frames[0]]
        for frame in frames[1:]:
            if frame.time_us - sequence[-1].time_us > max_delta_ms*1000:
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
