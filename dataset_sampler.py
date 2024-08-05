import pandas as pd
import torch
from torch.utils.data import Dataset


class GenerationDataset(Dataset):
    def __init__(self, path_to_set):
        full_dataset = pd.read_csv(path_to_set, delimiter='\t')
        full_dataset['core_lemma'] = full_dataset['core_lemma'].apply(lambda x: '' + 'an image of ' + x.replace('_', ' '))
        self.lemmas = list(zip(full_dataset['wordnet_id'], full_dataset['core_lemma']))


    def __len__(self):
        return len(self.lemmas)
     
    def __getitem__(self, idx):
        return self.lemmas[idx]



if __name__ == '__main__':

    our_set = GenerationDataset('/Users/eneminova/LLM_Taxonomy/text2image/taxo2img_test_set_for_DiffMs_with_images_paths_2.tsv')
    print(our_set[102])
