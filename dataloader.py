import torch
import numpy as np
import torch.nn as nn
from collections import Counter
from analysis import get_poi_sequences
from torch.utils.data import Dataset


class NoiseDistribution:
    def __init__(self, vocab):
        self.probs = np.array([vocab.freqs[w] for w in vocab.words])
        self.probs = np.power(self.probs, 0.75)
        self.probs /= np.sum(self.probs)
    def sample(self, n):
        "Returns the indices of n words randomly sampled from the vocabulary."
        return np.random.choice(a=self.probs.shape[0], size=n, p=self.probs)
        

class NegativeSampling(nn.Module):
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()
    def forward(self, scores):
        batch_size = scores.shape[0]
        n_negative_samples = scores.shape[1] - 1   # TODO average or sum the negative samples? Summing seems to be correct by the paper
        positive = self.log_sigmoid(scores[:,0])
        negatives = torch.sum(self.log_sigmoid(-scores[:,1:]), dim=1)
        return -torch.sum(positive + negatives) / batch_size  # average for batch


class Vocab:
    def __init__(self, poi_ds, min_count=0):
        self.min_count = min_count
        all_tokens = poi_ds.full_sequence_gdf['gr_cat_class_code'].tolist()
        self.freqs = {t:n for t, n in Counter(all_tokens).items() if n >= min_count}
        self.words = sorted(self.freqs.keys())
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        all_paragraphs = poi_ds.full_sequence_gdf['losa_id'].tolist()
        self.paragraphs_freqs = {t:n for t, n in Counter(all_paragraphs).items()}
        self.paragraphs = sorted(self.paragraphs_freqs.keys())
        self.paragraph2idx = {w: i for i, w in enumerate(self.paragraphs)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.idx2paragraph = {i: w for w, i in self.paragraph2idx.items()}
        self.gr_cat_class_code_name_map = {}
        for idx, row in poi_ds.full_sequence_gdf.iterrows():
            self.gr_cat_class_code_name_map[row['gr_cat_class_code']] = row['gr_cat_class']


class CustomPoiDataset(Dataset):
    def __init__(self, **kwargs):
        self.full_sequence_gdf = get_poi_sequences(**kwargs)
        self.seq_ids = sorted(self.full_sequence_gdf['seq_id'].unique().tolist())
        self.full_sequence_gdf['losa_id'] = self.full_sequence_gdf['postcode'].apply(self.add_losa_id)
        self.paragraphs = self.full_sequence_gdf['losa_id'].unique().tolist()
        self.paragraphs = sorted(self.paragraphs)

    def add_losa_id(self, postcode):
        return postcode.replace(' ', '')[:5]
    
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_gdf = self.full_sequence_gdf[self.full_sequence_gdf['seq_id'] == seq_id]
        center_row = seq_gdf[seq_gdf['is_center'] == True]
        context_rows = seq_gdf[seq_gdf['is_center'] == False]
        paragraph = center_row['losa_id'].values[0]
        context_rows = context_rows.sort_values('distance')
        center_class = center_row['gr_cat_class_code'].values[0]
        context = context_rows['gr_cat_class_code'].values
        seq = [center_class] + context.tolist()
        item = {'seq': seq, 'paragraph': paragraph, 'seq_id': seq_id}
        return item
