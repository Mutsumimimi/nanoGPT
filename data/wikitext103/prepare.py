"""
Prepare WikiText-103 dataset for GPT-2 BPE training.
Downloads via HuggingFace datasets, tokenizes with tiktoken GPT-2 BPE,
saves train.bin, val.bin, and meta.pkl.
"""
import os
import pickle
import tiktoken
import numpy as np
from datasets import load_dataset

# download WikiText-103
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# encode with GPT-2 BPE tokenizer
enc = tiktoken.get_encoding('gpt2')

def tokenize(examples):
    return {'ids': [enc.encode_ordinary(text) for text in examples['text']]}

dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])

# flatten into single list per split
def flatten_and_save(split, name):
    ids = []
    for sample in dataset[split]['ids']:
        ids.extend(sample)
        ids.append(enc.eot_token)  # add EOS between articles
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(os.path.dirname(__file__), f'{name}.bin'))
    print(f"{name} has {len(ids):,} tokens")

flatten_and_save('train', 'train')
flatten_and_save('validation', 'val')

# save meta
meta = {'vocab_size': enc.n_vocab}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"vocab_size: {enc.n_vocab}")
print("done")
