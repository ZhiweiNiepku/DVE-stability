import pandas as pd
from .ddg_dataset import DdgData, ddG_Dataset, ProteinDataset
from torch.utils.data import DataLoader

def make_dataloader(train_dataset_path="", test_dataset_path="", batch_size=8, is_training=True):
    if is_training:
        train_data, test_data = DdgData(train_dataset_path), DdgData(test_dataset_path)

        train_ds = ddG_Dataset(train_data.wild_tokens, train_data.mutant_tokens, train_data.positions, train_data.ddGs, train_data.cls)
        test_ds = ddG_Dataset(test_data.wild_tokens, test_data.mutant_tokens, test_data.positions, test_data.ddGs, test_data.cls)

        training_loader = DataLoader(train_ds, batch_size=batch_size, num_workers = 8, shuffle = True)
        testing_loader = DataLoader(test_ds, batch_size=batch_size, num_workers = 8)

        return training_loader, testing_loader
    else:
        test_data = DdgData(test_dataset_path)
        test_ds = ddG_Dataset(test_data.wild_tokens, test_data.mutant_tokens, test_data.positions, test_data.ddGs, test_data.cls)
        testing_loader = DataLoader(test_ds, batch_size=batch_size, num_workers = 8)

        return testing_loader