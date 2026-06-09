"""
Prepare WikiText-103 dataset for GPT-2 BPE training.
Reads .parquet files downloaded via hf download, tokenizes
with tiktoken GPT-2 BPE, saves train.bin, val.bin, and meta.pkl.
"""
import os
import pickle
import glob
import tiktoken
import numpy as np
import pyarrow.parquet as pq

out_dir = os.path.dirname(__file__)
raw_dir = os.path.join(out_dir, 'wikitext-103-raw-v1')

splits = {
    'train': 'train',
    'val':   'validation',
}

enc = tiktoken.get_encoding('gpt2')

for name, prefix in splits.items():
    parquet_files = sorted(glob.glob(os.path.join(raw_dir, f'{prefix}-*.parquet')))

    if not parquet_files:
        raise FileNotFoundError(f'no {prefix}-*.parquet files found in {raw_dir}')

    ids = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        for text in table.column('text').to_pylist():
            if text and text.strip():
                ids.extend(enc.encode_ordinary(text))
                ids.append(enc.eot_token)

    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(out_dir, f'{name}.bin'))
    print(f'{name}.bin: {len(ids):,} tokens')

meta = {'vocab_size': enc.n_vocab}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f'vocab_size: {enc.n_vocab}')
print('done')
