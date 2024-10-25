import prepare_dataset_utils.packed_dataset as packed_dataset
import argparse
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import glob
import os
from prepare_dataset_utils.tokenizer import Tokenizer
import torch

def de_binarize(filenames):
    print(filenames)
    
    checkpoint_dir =  Path("./prepare_dataset_utils/tokenizer.json")
    tokenizer = Tokenizer(checkpoint_dir)
    n_chunks = 1
    block_size = 4096
    dummy_out= packed_dataset.PackedDataset(filenames,n_chunks,block_size,seed=12345, shuffle=False)
    
    j=0
    global_tokens = 0
    for i in dummy_out:
        print (torch.unique(i,return_counts = True))
        # print(i)
        # for j in i:
        #     if j != 50256:
        #         global_tokens += 1
        #         print("global_tokens  ",global_tokens)
       
        # # global_tokens +=  i.shape[0] 
        # j = j+1
    print("global_tokens  ",global_tokens)
    
    
    # loader = DataLoader(dummy_out,batch_size=256,shuffle=False, drop_last=True)
    # global_tokens = 0
    # j=0
    # # print("len(loader.dataset)" , len(loader.dataset))
    # for i in loader:
    #     print(i)
    #     print(i.shape)
    #     print(j)
    #     j= j+1
    #     global_tokens += i.shape[0] * i.shape[1]
    #     # print("content",i.shape)
    # print("global_tokens  ",global_tokens)
        

if __name__ == "__main__":
    
    parser= argparse.ArgumentParser()
    parser.add_argument("--src_path",type=str,)
    parser.add_argument("--dest_path",type=str)
    
    
    args = parser.parse_args()
    src_path = Path (args.src_path)
    dest_path =Path( args.dest_path)
    
    filenames = glob.glob(os.path.join(src_path, '**', '*.bin'), recursive=True)
    
    print(filenames)
    
    # filenames = ["dataset_out/bigger_0000000000.bin","dataset_out/bigger_0000000001.bin","dataset_out/bigger_0000000002.bin","dataset_out/bigger_0000000003.bin"]

    
    de_binarize(filenames)
    