import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam  
from tqdm import trange
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from dataloader import CustomPoiDataset, Vocab, NegativeSampling, NoiseDistribution

class DistributedMemory(nn.Module):
    def __init__(self, vec_dim, n_docs, n_words, concat=False):
        super(DistributedMemory, self).__init__()
        self.paragraph_matrix = nn.Parameter(torch.randn(n_docs, vec_dim)) # (n_docs, vec_dim)
        self.word_matrix = nn.Parameter(torch.randn(n_words, vec_dim))
        self.outputs = nn.Parameter(torch.zeros(vec_dim, n_words))
        self.concat = concat
    
    def forward(self, doc_ids, context_ids, sample_ids):
        if self.concat:
            word_matrix = self.word_matrix[context_ids,:]
            doc_matrix = self.paragraph_matrix[doc_ids,:].unsqueeze(dim=1)
            inputs = torch.cat([doc_matrix, word_matrix], dim=1)
        else:
            word_matrix = torch.sum(self.word_matrix[context_ids,:], dim=1)
            doc_matrix = self.paragraph_matrix[doc_ids,:]
            inputs = torch.add(doc_matrix, word_matrix).unsqueeze(dim=1)
        outputs = self.outputs[:,sample_ids]
        outputs = torch.bmm(inputs, outputs.permute(1, 0, 2)).squeeze()   
        if self.concat:
            outputs = outputs.sum(dim=1) 
        return outputs

def train(model, dataloader, loss_fn, epochs=40, lr=1e-3, device="cpu", min_delta= None, vocab=None, ds=None):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mid_checkpoint_path = f'models/doc2vec_checkpoint_mid_{timestamp}.pth'
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, factor=0.8, verbose=True)
    training_losses = []
    model.to(device)
    print(f"Training on {device}")
    try:
        tbar =  trange(epochs, desc=f"Epochs , Loss: {0.0}")
        for epoch in tbar:
            epoch_losses = []
            for batch in dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                model.zero_grad()
                logits = model.forward(**batch)
                batch_loss = loss_fn(logits)
                epoch_losses.append(batch_loss.item())
                batch_loss.backward()
                optimizer.step()
            training_losses.append(np.mean(epoch_losses))
            tbar.set_description(f"Epochs , Loss: {training_losses[-1]}")
            scheduler.step(training_losses[-1])
            if min_delta is not None and len(training_losses) > 1:
                if training_losses[-2] - training_losses[-1] < min_delta and training_losses[-1] - training_losses[-2] > 0:
                    print(f"Loss improvement less than {min_delta} - stopping early!")
                    break
            if epoch % 100 == 0:
                save_checkpoint(model, training_losses, vocab, ds, filename=mid_checkpoint_path, add_timestamp=False)

    except KeyboardInterrupt:
        print(f"Interrupted on epoch {epoch}!")
    finally:
        return training_losses

class NCEDataset(Dataset):
    def __init__(self, examples):
        self.examples = list(examples)  # just naively evaluate the whole damn thing - suboptimal!
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]

def example_generator(ds, context_size, noise, n_negative_samples, vocab):
    for item in ds:
        seq = item['seq']
        doc_id = item['paragraph']
        doc_id = vocab.paragraph2idx[doc_id]
        for i in range(context_size, len(seq) - context_size):
            positive_sample = vocab.word2idx[seq[i]]
            sample_ids = noise.sample(n_negative_samples).tolist()
            # Fix a wee bug - ensure negative samples don't accidentally include the positive
            sample_ids = [sample_id if sample_id != positive_sample else -1 for sample_id in sample_ids]
            sample_ids.insert(0, positive_sample)                
            context = seq[i - context_size:i] + seq[i + 1:i + context_size + 1]
            context_ids = [vocab.word2idx[w] for w in context]
            yield {"doc_ids": torch.tensor(doc_id),  # we use plural here because it will be batched
                   "sample_ids": torch.tensor(sample_ids), 
                   "context_ids": torch.tensor(context_ids)}
            
def describe_batch(batch, vocab, convert_codes= False):
    results = []
    for doc_id, context_ids, sample_ids in zip(batch["doc_ids"], batch["context_ids"], batch["sample_ids"]):
        context = [vocab.words[i] for i in context_ids]
        samples = [vocab.words[i] for i in sample_ids]
        if convert_codes:
            context = [vocab.gr_cat_class_code_name_map[code] for code in context]
            samples = [vocab.gr_cat_class_code_name_map[code] for code in samples]
            paragraph = vocab.idx2paragraph[doc_id.item()]
        else:
            paragraph = doc_id.item()
        context.insert(len(context_ids) // 2, "____")
        result = {
            "doc_id": doc_id,
            "paragraph": paragraph,
            "context": " ".join(context), 
            "context_ids": context_ids, 
            "samples": samples, 
            "sample_ids": sample_ids
            }
        results.append(result)
    return results
            

def main(dataset_args=None, vocab_min_count=0, n_negative_samples=5, context_size=10, vec_dim=100, concat=False, lr=1e-3,
            epochs=40, device="cpu", min_delta=None, dataloader_args=None):
    ds = CustomPoiDataset(**dataset_args)
    vocab = Vocab(ds, min_count=vocab_min_count)
    noise = NoiseDistribution(vocab)
    loss_fn = NegativeSampling()
    model = DistributedMemory(vec_dim=vec_dim, n_docs=len(vocab.paragraphs), n_words=len(vocab.words), concat=concat)
    examples = example_generator(ds, context_size, noise, n_negative_samples, vocab)
    dataloader = DataLoader(NCEDataset(examples), **dataloader_args)
    training_losses = train(model, dataloader, loss_fn, epochs=epochs, lr=lr, device=device, min_delta=min_delta, vocab=vocab, ds=ds)
    return model, training_losses, vocab, ds

def save_checkpoint(model, training_losses, vocab, ds, filename="doc2vec_checkpoint.pth", add_timestamp=True):
    if add_timestamp:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename, ext = filename.split(".")
        filename = f"{filename}_{timestamp}.{ext}"
    torch.save({
        "model_state_dict": model.state_dict(),
        "training_losses": training_losses,
        "vocab": vocab,
        "ds": ds
    }, filename)
    print(f"Checkpoint saved as {filename}")
    return filename

def load_checkpoint(filename="doc2vec_checkpoint.pth"):
    checkpoint = torch.load(filename)
    model = DistributedMemory(vec_dim=checkpoint["model_state_dict"]["paragraph_matrix"].shape[1], 
                              n_docs=checkpoint["model_state_dict"]["paragraph_matrix"].shape[0], 
                              n_words=checkpoint["model_state_dict"]["word_matrix"].shape[0])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["training_losses"], checkpoint["vocab"], checkpoint["ds"]

if __name__ == '__main__':
    # Example usage
    model, training_losses, vocab, ds = main() # ... lots of args
    ckpt_path = save_checkpoint(model, training_losses, vocab, ds)
    # ... later
    model, training_losses, vocab, ds = load_checkpoint(ckpt_path)