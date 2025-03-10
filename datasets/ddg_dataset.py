from torch.utils.data import Dataset
import esm
import torch
import pandas as pd

class ProteinDataset(Dataset):
    def __init__(self, df):
        self.df = df
        _, esm1v_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm1v_batch_converter = esm1v_alphabet.get_batch_converter()

    def __getitem__(self, idx):
        _, _, esm1b_batch_tokens1 = self.esm1v_batch_converter([('' , ''.join(self.df.iloc[idx]['wt_seq'])[:1022])])
        _, _, esm1b_batch_tokens2 = self.esm1v_batch_converter([('' , ''.join(self.df.iloc[idx]['mut_seq'])[:1022])])
        pos = self.df.iloc[idx]['pos']
        return esm1b_batch_tokens1.squeeze(dim=0), esm1b_batch_tokens2.squeeze(dim=0), pos+1, torch.FloatTensor([self.df.iloc[idx]['ddg']])

    def __len__(self):
        return len(self.df)

def data_frame_to_list(dataframe):
    header = ["wt_seq", "mut_seq", "ddg", "pos"]
    lenOfDate = len(dataframe)
    listOfdate = []
    for raw in range(lenOfDate):
        listTemp = []
        for name in header:
            listTemp.append(dataframe.iloc[raw][name])
        listOfdate.append(listTemp)
    return listOfdate

class DdgData(object):
    def __init__(self, data_path):
        _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.tokenizer = esm2_alphabet.get_batch_converter()

        dataframe = pd.read_csv(data_path)
        self.wild_tokens, self.mutant_tokens, self.positions, self.ddGs, self.cls = self.__process__(dataframe)
        
    def __process__(self, data):
        data_list = data_frame_to_list(data)
        wild_seqs = []
        mutation_seqs = []
        positions = []
        ddGs = []
        cls = []

        for index, (wild_seq, mutation_seq, ddg, pos) in enumerate(data_list):
            wild_seqs.append(("none", wild_seq[:1022]))
            mutation_seqs.append(("none", mutation_seq[:1022]))
            positions.append(pos + 1)
            ddGs.append(ddg)
            if ddg > 0:
                cls.append(1)
            else:
                cls.append(0)
        wild_labels, wild_strs, wild_tokens = self.tokenizer(wild_seqs)
        mutation_labels, mutation_strs, mutant_tokens = self.tokenizer(mutation_seqs)
        positions = torch.tensor(positions)
        ddGs = torch.tensor(ddGs)
        cls = torch.tensor(cls)

        return wild_tokens, mutant_tokens, positions, ddGs, cls
    
    def __len__(self):
        return len(self.ddGs)