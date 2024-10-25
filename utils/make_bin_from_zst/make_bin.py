from pathlib import Path
import pyarrow.parquet as pq
from litdata import optimize, TokensLoader
from functools import partial
from binarize_scripts.prepare_dataset_utils.tokenizer import Tokenizer
import zstandard as zstd
import json





# 1. Define a function to convert the text within the parquet files into tokens
def tokenize_fn1(filepath, tokenizer=None):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
        for text in batch.to_pandas()["content"]:
            encoded_text = tokenizer.encode(text, bos=False, eos=True)
            # # decoded_text = tokenizer.decode(text)
            print("original_text: ", text)
            print("encoded_text: ", encoded_text)
            # # print("decoded_text: ",decoded_text)
            yield tokenizer.encode(text, bos=False, eos=True)


def tokenize_fn(filepath, tokenizer=None):
    with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            text = json.loads(row)["text"]

            text_ids = tokenizer.encode(text, bos=False, eos=True)
            # print("original_text: ",text)
            # print("decoded_text: ",text_ids)
            yield text_ids


if __name__ == "__main__":
    # 2. Generate the inputs

    input_dir = "slimpajama_1b_zst/val"
 
    inputs = [str(file) for file in Path(f"{input_dir}").rglob("*.zst")]
    print(inputs)

    # 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
    outputs = optimize(
        fn=partial(
            tokenize_fn,
            tokenizer=Tokenizer(
                "../slimpajama_val_test_trained_bpe_tok.json"
            ),
        ),
        inputs=[str(f) for f in inputs],
        output_dir="slimpajama_1b_bin/val",
        chunk_size=(2049 * 8012),
        # chunk_size=((4 + 1)* 8012),
        item_loader=TokensLoader(),
    )
