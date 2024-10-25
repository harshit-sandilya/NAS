import os
import glob
import json
import zstandard as zstd
import argparse

def read_zst_file(file_path,filter_datatype):
    dctx = zstd.ZstdDecompressor()
    with open(file_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            buffer = b''
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    
                    
                    if(json.loads(line)['meta']['redpajama_set_name'] == filter_datatype):
                        yield json.loads(line)
            if buffer:
                if(json.loads(buffer)['meta']['redpajama_set_name'] == filter_datatype):
                        yield json.loads(buffer)

def write_jsonl_file(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_chunk(input_dir, output_dir, base_path, chunk_name,filter_datatype):
    # Create the output path
    output_chunk_dir = os.path.join(output_dir, base_path)
    
    os.makedirs(output_chunk_dir, exist_ok=True)
    # print("output_chunk_dir     : ",output_chunk_dir,   " base_path :  ",base_path," chunk_name     :",chunk_name )
        
    output_file = os.path.join(output_chunk_dir, f"{base_path}_{chunk_name}.jsonl")
    if(base_path == "."):
        output_file = os.path.join(output_chunk_dir, f"{chunk_name}.jsonl")
    # Recursively find all .jsonl.zst files in the chunk
    zst_files = glob.glob(os.path.join(input_dir, chunk_name, "*.zst"))
    
    # Read and merge .zst files
    merged_data = []
    for file in zst_files:
        merged_data.extend(read_zst_file(file,filter_datatype))
    # print("writing to ", output_file)
    # Write merged data to a single jsonl file
    write_jsonl_file(merged_data, output_file)
    print(f"Merged {len(zst_files)} files into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge .jsonl.zst files into a single .jsonl file.")
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing .jsonl.zst files')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for the merged .jsonl files')
    args = parser.parse_args()
    filter_datatype =  "RedPajamaCommonCrawl"
    # filter_datatype = "RedPajamaC4"
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the directory structure
    for base_dir, subdirs, _ in os.walk(input_dir):
        for subdir in subdirs:
            if subdir.startswith('chunk'):
                # Determine base path name for output file
                # print("base_dir  ",base_dir)
                # print("input_dir  ",input_dir)
                relative_path = os.path.relpath(base_dir, input_dir)
                # print("relative_path  ",relative_path)
                base_path = relative_path.replace(os.sep, '_')
                # print("base_path             ", base_path)
                # Process each chunk directory
                process_chunk(base_dir, output_dir, base_path, subdir,filter_datatype)


if __name__ == "__main__":
    main()