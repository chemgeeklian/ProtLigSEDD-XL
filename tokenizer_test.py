from transformers import AutoTokenizer; 

t=AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct',trust_remote_code=True); 
#print('vocab_size', t.vocab_size); 
#print(t('c1ccccc1'))

import torch

# Path to your .pt file
data_path = "/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/input_data/processed_pubchem_subset_1k.pt"

# Load the dataset
data = torch.load(data_path, weights_only=False)  # weights_only=False keeps full objects

print(f"Type: {type(data)}")
print(f"Dataset length: {len(data)}")

# If it's a list/dict of samples with 'smiles' or 'ligand_smiles' keys
for i, sample in enumerate(data[:10]):  # print first 10 as example
    if isinstance(sample, dict):
        if 'ligand_smiles' in sample:
            print(f"Sample {i}: {sample['ligand_smiles']}")
        elif 'smiles' in sample:
            print(f"Sample {i}: {sample['smiles']}")
        else:
            print(f"Sample {i}: keys = {sample.keys()}")
    else:
        print(f"Sample {i}: {sample}")
