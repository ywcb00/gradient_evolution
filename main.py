from dataset import DatasetID, getDataset
from plot import plotAll
from train import buildAndFit

import os
from pathlib import Path

dataset_id = DatasetID.Mnist

seed = 13

FIGURES_DIR = Path(f'figures_{dataset_id.name}')
os.makedirs(FIGURES_DIR, exist_ok=True)

train, val, test = getDataset(dataset_id, seed)

gradients, individual_gradients, metrics = buildAndFit(dataset_id, train, FIGURES_DIR, seed)

plotAll(dataset_id, gradients, individual_gradients, metrics, FIGURES_DIR)
