import math
import os
from typing import Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class UNetDataset(Dataset):
    def __init__(self):
        self._DIR = "../build/bin/bin/"
        self._INPUT_PATH = self._DIR + "input_static_2d"
        self._TARGET_PATH = self._DIR + "output_static_2d"

        self.input_data = None
        self.target_data = None

        self.input_train = None
        self.input_test = None
        self.target_train = None
        self.target_test = None

        self.current_input_data = None
        self.current_target_data = None

    def __len__(self):
        return len(self.current_input_data)

    def __getitem__(self, item):
        return self.current_input_data[item], self.current_target_data[item]

    def set_current(self, train=True):
        if train:
            self.current_input_data = torch.from_numpy(self.input_train)
        else:
            self.current_input_data = torch.from_numpy(self.input_test)

        if train:
            self.current_target_data = torch.from_numpy(self.target_train)
        else:
            self.current_target_data = torch.from_numpy(self.target_test)

    def make_splits(self):
        top = math.ceil(self.input_data.shape[0] * 0.7)
        floor = math.ceil(self.input_data.shape[0] * 0.3)

        self.input_train = self.input_data[:top, :]
        self.input_test = self.input_data[top:floor + top, :]

        self.target_train = self.target_data[:top, :]
        self.target_test = self.target_data[top:floor + top, :]

    def load(self):
        assert os.path.isfile(self._INPUT_PATH), f"Cannot find input path at: {self._INPUT_PATH}"
        assert os.path.isfile(self._TARGET_PATH), f"Cannot find target path at: {self._TARGET_PATH}"

        print("Training datasets found, formatting")
        self.input_data = self._read_file(self._INPUT_PATH)
        self.target_data = self._read_file(self._TARGET_PATH)

        assert self.input_data.shape == self.target_data.shape, "Input/target shapes MUST match"

        print(f"Data loaded successfully Input Data: {self.input_data.shape}, Target Data: {self.target_data.shape}")

        print("Making splits of the input data")
        self.make_splits()

        print("Defaulting current dataset to training inputs")
        self.set_current(train=True)

    def _read_file(self, file_name: str) -> np.array:
        shape = self._decide_shape(file_name)
        lines = list(open(file_name).readlines()[1:])
        for i, line in enumerate(lines):
            lines[i] = list(map(float, line.rstrip().split()))

        lines = np.array(lines)

        entries = []
        for i in range(0, len(lines.T), shape[2]):
            entries.append(lines[:, i:i + shape[2]])

        lines = np.array(entries)
        lines = lines.reshape(shape)
        return lines

    def _decide_shape(self, file_name: str) -> Tuple[Union[int, Any], ...]:
        # Eigen is column major
        rows = 0
        cols = 1
        features = 2
        entries = 3

        # numpy is row major
        np_rows = 2
        np_cols = 3
        np_features = 1
        np_entries = 0

        # Assumes the shape is the first line of the file.
        shape = list(map(int, open(file_name).readline().rstrip().split(' ')))

        rows = shape[rows]
        cols = shape[cols]
        features = shape[features]
        entries = shape[entries]

        shape[np_rows] = rows
        shape[np_cols] = cols
        shape[np_features] = features
        shape[np_entries] = entries

        self.shape = tuple(shape)
        return tuple(self.shape)
