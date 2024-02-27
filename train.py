import argparse 
import torch
from doc2vec import main, save_checkpoint


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recreate", type=int, default=0)
    parser.add_argument("--max_sequence_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--drop_last", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--vec_dim", type=int, default=100)
    parser.add_argument("--vocab_min_count", type=int, default=0)
    parser.add_argument("--n_negative_samples", type=int, default=5)
    parser.add_argument("--context_size", type=int, default=10) 
    parser.add_argument("--concat", type=int, default=0)

    args = parser.parse_args()
    
    pois_dataset_args = {
        'force_recreate': bool(args.force_recreate),
        'max_sequence_size': args.max_sequence_size
    }

    dataloader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': bool(args.shuffle),
        'drop_last': bool(args.drop_last)
    }
    
    device = args.device
    epochs = args.epochs
    lr = args.lr
    min_delta = args.min_delta
    vec_dim = args.vec_dim
    vocab_min_count = args.vocab_min_count
    n_negative_samples = args.n_negative_samples
    context_size = args.context_size
    concat = bool(args.concat)

    model, training_losses, vocab, ds = main(
        dataset_args=pois_dataset_args, 
        dataloader_args=dataloader_args, 
        concat=concat,
        vec_dim=vec_dim,  
        vocab_min_count=vocab_min_count, 
        n_negative_samples=n_negative_samples,
        context_size=context_size, 
        device=device, 
        epochs=epochs, 
        lr=lr, 
        min_delta=min_delta
    )
    
    checkpoint_path = f'models/doc2vec_checkpoint.pt'
    ckpt_path = save_checkpoint(
        model=model, 
        training_losses=training_losses, 
        vocab=vocab, 
        ds=ds, 
        filename=checkpoint_path, 
        add_timestamp=True
    )
    
    print(f"Checkpoint saved as {ckpt_path}")