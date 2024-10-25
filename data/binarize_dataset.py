import json
import os
import sys
from pathlib import Path
import argparse
import glob
import re
import subprocess


import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from prepare_dataset_utils.tokenizer import Tokenizer
import prepare_dataset_utils.packed_dataset as packed_dataset



def prepare_sample(source_path: Path,  destination_path: Path, checkpoint_dir: Path,chunk_size: int) -> None:
    """Prepare the  mentioned datasets using the original tokenizer."""

    destination_path.mkdir(parents=True, exist_ok=True)

    filenames_sample = glob.glob(os.path.join(source_path, '**', '*.jsonl'), recursive=True)
    
    tokenizer = Tokenizer(checkpoint_dir)

    for name in filenames_sample:
        filepath = name
        prefix = re.sub(r"/","_",os.path.relpath(filepath,source_path) )
        prefix = re.sub(r".jsonl","",prefix)
        
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()

def prepare(source_path : Path, destination_path : Path, checkpoint_dir: Path, chunk_size : int) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""

    prepare_fn = prepare_sample
    prepare_fn(source_path=source_path, destination_path=destination_path, checkpoint_dir=checkpoint_dir, chunk_size = chunk_size
    )



if __name__ == "__main__":
    
    
    parser= argparse.ArgumentParser()
    parser.add_argument("--src_path",type=str,)
    parser.add_argument("--dest_path",type=str)
    parser.add_argument("--vocab_path",type=str)
    parser.add_argument("--chunk_size",type=int)
    
    
    args = parser.parse_args()
    src_path = Path(args.src_path)
    dest_path = Path(args.dest_path)
    vocab_path = Path(args.vocab_path)
    chunk_size = (args.chunk_size +1) * 1024

    prepare(src_path,dest_path,vocab_path,chunk_size)
  
  
  
    
