import pandas as pd
import torch
from torch.utils.data import Dataset


class GenerationDataset(Dataset):
    def __init__(self, path_to_set):
        full_dataset = pd.read_csv(path_to_set)
        full_dataset['core_lemma'] = full_dataset.apply(lambda x: 'An image of' + x['core_lemma'].replace('_', ' ') + ' (' + x['definition'] + ')', axis=1)
        
        self.lemmas = list(zip(full_dataset['wordnet_id'], full_dataset['core_lemma']))

    def __len__(self):
        return len(self.lemmas)
     
    def __getitem__(self, idx):
        return self.lemmas[idx]



if __name__ == '__main__':
    our_set = GenerationDataset('/home/vityav/test_t2i/main.csv')
    print(our_set[102])
