import os
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from joblib import dump, load
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset(file_path):
    """"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def group_dataset(data):
    """group dataset according to video id"""
    grouped_data = {}
    for d in tqdm(data, desc="grouping data into videos"):
        video_id = d['video_id']
        if video_id not in grouped_data.keys():
            grouped_data[video_id] = []
        grouped_data[video_id].append(d)
    
    return grouped_data

def create_2d_joints_sequences(grouped_data, ds_factor=8):
    sequences = []
    for video in tqdm(grouped_data.values(), desc="creating joint sequences"):
        sequence = []
        for frame in video:
            joints_2d = frame["joints_2d"]
            sequence.append(joints_2d)
        
        sequences.append(sequence[::ds_factor])
        
    return sequences

def create_training_sequences(sequences, n=20):
    """creates sequences of n frames"""
    train_sequences = []
    for sequence in tqdm(sequences, desc=f"creating sequences of size {n}"):
        train_seq = []
        for frame in sequence:
            if len(train_seq) == n:
                train_sequences.append(train_seq)
                train_seq = []
            train_seq.append(frame)
    return train_sequences

def unroll_2d_sequences(sequences):
    sequences_updated = []
    for sequence in tqdm(sequences, desc=f"unrolling joint vectors"):
        sequences_updated.append(np.array([
            joints_vec.ravel().reshape(1, -1) for joints_vec in sequence
        ]))
    return sequences_updated

def stack_sequences(sequences):
    """Stacking sequences to make a single matrix"""
    return np.vstack(sequences).squeeze()


class Scaler:
    """Helper class for normalization"""
    def __init__(self, save_filename, mode="train", s_type="normalize"):
        self.f_name = save_filename
        self.mode = mode
        self.s_type = s_type
        self.scaler = self._get_scaler()
    
    def _get_scaler(self):
        if self.mode=="test":
            return load(self.f_name)
        else:
            return MinMaxScaler() if self.s_type=="normalize" else StandardScaler()
    
    def _fit(self, sequences_matrix):
        x_columns = range(0,34,2)
        y_columns = range(1,34,2)
        x_max = max(np.max(sequences_matrix, axis=0)[x_columns])
        y_max = max(np.max(sequences_matrix, axis=0)[y_columns])
        x_min = min(np.min(sequences_matrix, axis=0)[x_columns])
        y_min = min(np.min(sequences_matrix, axis=0)[y_columns])

        min_max_matrix = np.vstack(
            [np.array([x_max, y_max]*17).reshape(1,-1),
            np.array([x_min, y_min]*17).reshape(1,-1)]
        )
        _ = self.scaler.fit_transform(min_max_matrix)
        
    def fit(self, sequences_matrix):
        # scaled_sequences_matrix = self.scaler.fit_transform(sequences_matrix)
        self._fit(sequences_matrix)
        scaled_sequences_matrix = self.transform(sequences_matrix)
        # Saving the scaler so it can be loaded later during testing
        dump(self.scaler, self.f_name)

        return scaled_sequences_matrix
    
    def transform(self, sequences_matrix):
        return self.scaler.transform(sequences_matrix)

def prepare_2d_data(
        data_file,
        mode="train",
        s_fname="scaler.joblib",
        s_type="normalize"
):
    data = load_dataset(data_file)
    grouped_data = group_dataset(data)
    sequences = create_2d_joints_sequences(grouped_data)
    train_sequences = create_training_sequences(sequences)
    train_sequences = unroll_2d_sequences(train_sequences)
    seq_matrix = stack_sequences(train_sequences)
    scaler = Scaler(s_fname, mode=mode, s_type=s_type)
    if mode == "train":
        seq_matrix = scaler.fit(seq_matrix)
    else:
        seq_matrix = scaler.transform(seq_matrix)
    
    return seq_matrix