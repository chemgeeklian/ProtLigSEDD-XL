#!/usr/bin/env python3
"""
Simple PyTorch finetuning script for louaaron/sedd-medium using the repo's SEDD model and loss.

Usage example:
  python pt_finetune_sedd.py \
    --data_path /lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/input_data/processed_pubchem_subset_1k.pt \
    --model_name louaaron/sedd-medium \
    --tokenizer ibm/MoLFormer-XL-both-10pct \
    --output_dir ./sedd_finetune_out \
    --batch_size 8 --epochs 3 --lr 1e-4

This script intentionally uses a straightforward training loop (no DDP). It's intended for quick fine-tuning
or debugging on small datasets. For large-scale training please use the repo's `train.py`/DDP pipeline.
"""
import argparse
import os
import math
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, GPT2TokenizerFast
import data
from datasets import load_dataset

import load_model
import losses
import noise_lib
import graph_lib
from model.ema import ExponentialMovingAverage


def load_smiles_from_pt(path: str) -> List[str]:
    obj = torch.load(path, map_location='cpu')
    smiles = []
    if isinstance(obj, (list, tuple)):
        it = obj
    elif isinstance(obj, dict):
        # common pattern: dict with list values
        # if one key 'ligand_smiles' or 'smiles', use that
        if 'ligand_smiles' in obj and isinstance(obj['ligand_smiles'], (list,tuple)):
            return list(obj['ligand_smiles'])
        if 'smiles' in obj and isinstance(obj['smiles'], (list,tuple)):
            return list(obj['smiles'])
        # otherwise iterate values
        it = list(obj.values())
    else:
        # fallback: attempt to iterate
        try:
            it = list(obj)
        except Exception:
            raise RuntimeError(f'Unsupported object loaded from {path}: {type(obj)}')

    for sample in it:
        if isinstance(sample, dict):
            if 'ligand_smiles' in sample:
                smiles.append(sample['ligand_smiles'])
            elif 'smiles' in sample:
                smiles.append(sample['smiles'])
            else:
                # take first string field
                found = False
                for v in sample.values():
                    if isinstance(v, str):
                        smiles.append(v)
                        found = True
                        break
                if not found:
                    smiles.append(str(sample))
        elif isinstance(sample, str):
            smiles.append(sample)
        else:
            smiles.append(str(sample))

    return smiles


def resize_model_vocab(model, new_vocab_size: int):
    """Resize model's embedding and final linear to new_vocab_size, copying overlapping rows."""
    # embedding
    old_vocab, dim = model.vocab_embed.embedding.shape
    if old_vocab == new_vocab_size:
        return
    device = model.vocab_embed.embedding.device
    new_emb = nn.Parameter(torch.empty((new_vocab_size, dim), device=device))
    torch.nn.init.kaiming_uniform_(new_emb, a=math.sqrt(5))
    ncopy = min(old_vocab, new_vocab_size)
    new_emb.data[:ncopy] = model.vocab_embed.embedding.data[:ncopy]
    model.vocab_embed.embedding = new_emb

    # final linear
    old_out, hidden = model.output_layer.linear.weight.shape
    # create new linear
    new_linear = nn.Linear(hidden, new_vocab_size, bias=True).to(device)
    new_linear.weight.data.zero_()
    new_linear.bias.data.zero_()
    ncopy2 = min(old_out, new_vocab_size)
    new_linear.weight.data[:ncopy2] = model.output_layer.linear.weight.data[:ncopy2]
    new_linear.bias.data[:ncopy2] = model.output_layer.linear.bias.data[:ncopy2]
    model.output_layer.linear = new_linear

    # update config tokens if present
    try:
        model.config.tokens = new_vocab_size
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, default=None,
                        help='Optional local .pt file path. If not set, the HF dataset specified by --dataset will be used.')
    parser.add_argument('--dataset', type=str, default='sagawa/pubchem-10m-canonicalized',
                        help='HuggingFace dataset id to use when --data_path is not provided')
    parser.add_argument('--cache_dir', type=str, default='data', help='cache dir for datasets.load_dataset')
    parser.add_argument('--num_proc', type=int, default=4, help='num_proc for dataset.map tokenization')
    parser.add_argument('--model_name', default='louaaron/sedd-medium')
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--output_dir', default='./sedd_finetune_out')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # tokenizer
    if args.tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    else:
        print('Loading tokenizer:', args.tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        # Ensure pad token exists (use EOS if needed)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # load model + graph + noise
    print('Loading model:', args.model_name)
    model, graph, noise = load_model.load_model_hf(args.model_name, device)
    model.train()

    # determine sequence length
    seq_len = args.max_length if args.max_length is not None else getattr(model.config, 'model', {}).get('length', None)
    # fallback to model.config.model.length path or model.config.model.length
    if seq_len is None:
        try:
            seq_len = model.config.model.length
        except Exception:
            seq_len = 1024
    print('Using seq_len =', seq_len)

    # Prepare dataset: either use local .pt or HF dataset
    if args.data_path is not None:
        print('Loading SMILES from', args.data_path)
        smiles = load_smiles_from_pt(args.data_path)
        print('Loaded', len(smiles), 'entries')
        smiles = [s for s in smiles if isinstance(s, str) and len(s) > 0]

        print('Tokenizing local list...')
        encoded = tokenizer(smiles, truncation=True, padding='max_length', max_length=seq_len, return_tensors='pt')
        input_ids = encoded['input_ids']
        dataset_for_training = None
        print('Tokenized shape:', input_ids.shape)
    else:
        print('Loading HF dataset:', args.dataset)
        # Use the repo's data.get_dataset which handles tokenization and chunking
        # mode 'train' used here
        dataset_for_training = data.get_dataset(args.dataset, 'train', cache_dir=args.cache_dir, block_size=seq_len, num_proc=args.num_proc, tokenizer=tokenizer)
        print('Dataset prepared with', len(dataset_for_training), 'examples (chunks)')

    # If tokenizer vocab differs, resize model vocab and final layer
    try:
        tok_vocab = int(tokenizer.vocab_size)
    except Exception:
        tok_vocab = input_ids.max().item() + 1
    try:
        model_vocab = int(model.config.tokens)
    except Exception:
        # fallbacks
        try:
            model_vocab = model.vocab_embed.embedding.shape[0]
        except Exception:
            model_vocab = None

    if model_vocab is not None and tok_vocab != model_vocab:
        print(f'Resizing model vocab {model_vocab} -> {tok_vocab}')
        resize_model_vocab(model, tok_vocab)

    # build dataloader
    if args.data_path is not None:
        ds = TensorDataset(input_ids)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    else:
        # dataset_for_training is a datasets.Dataset already chunked and with torch format
        # It contains column names like 'input_ids'
        # define a collate_fn that pads variable-length input_ids
        def collate_fn(batch):
            # batch: list of dicts
            input_ids_list = [b['input_ids'] for b in batch]
            # ensure tensors
            input_ids_list = [torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in input_ids_list]
            # if all equal length, stack; else pad
            lengths = [x.size(0) for x in input_ids_list]
            if len(set(lengths)) == 1:
                input_ids = torch.stack(input_ids_list, dim=0)
            else:
                pad_id = getattr(tokenizer, 'pad_token_id', None)
                if pad_id is None:
                    pad_id = 0
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            # collect smiles/text if present
            smiles = []
            for b in batch:
                if 'smiles' in b:
                    smiles.append(b['smiles'])
                elif 'text' in b:
                    smiles.append(b['text'])
                else:
                    smiles.append(None)
            return {'input_ids': input_ids, 'smiles': smiles}

        loader = DataLoader(dataset_for_training, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # optimizer & ema
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    loss_fn = losses.get_loss_fn(noise, graph, train=True)

    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for batch_idx, batch in enumerate(loader):
            # batch can be either a tensor batch from local TensorDataset or a dict from collate_fn
            if args.data_path is not None:
                input_ids = batch[0].to(device)
            else:
                # batch is a dict from collate_fn
                input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, input_ids).mean()
            loss.backward()
            optimizer.step()
            ema.update(model.parameters())

            global_step += 1
            if global_step % 10 == 0:
                print(f'Epoch {epoch} step {global_step} loss {loss.item():.6f}')

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f'checkpoint_{global_step}.pth')
                print('Saving checkpoint to', ckpt_path)
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': global_step}
                torch.save(state, ckpt_path)

        print(f'Epoch {epoch} finished in {time.time()-t0:.1f}s')

    # final save
    final_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
    print('Saving final checkpoint to', final_path)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': global_step}, final_path)


if __name__ == '__main__':
    main()


'''
python pt_finetune_sedd.py \
  --data_path /lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/input_data/processed_pubchem_subset_1k.pt \
  --model_name louaaron/sedd-medium \
  --output_dir ./sedd_finetune_out \
  --batch_size 8 \
  --epochs 3 \
  --lr 1e-4 \
  --max_length 128

python pt_finetune_sedd.py \
  --dataset sagawa/pubchem-10m-canonicalized \
  --model_name louaaron/sedd-medium \
  --output_dir ./sedd_finetune_out \
  --batch_size 8 \
  --epochs 3 \
  --lr 2e-5 \
  --max_length 128
  '''