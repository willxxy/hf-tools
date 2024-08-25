from transformers import EncoderDecoderModel

### HOW TO CREATE SHARED ENCODER DECODER MODEL
shared_bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased", tie_encoder_decoder=True)



### HOW TO COMBINE SAFE TENSORS INTO ONE .PT FILE

import torch
from safetensors import safe_open
import os

def combine_safetensors_to_pt(check_point_root, output_file, num_files=4):
    combined_tensor = {}
    checkpoint_keys = set()

    for i in range(1, num_files + 1):
        safe_tensors_path = os.path.join(check_point_root, f'model-0000{i}-of-00004.safetensors')
        print(f"Processing file {i} of {num_files}: {safe_tensors_path}")
        
        try:
            with safe_open(safe_tensors_path, framework='pt') as f:
                for key in f.keys():
                    if key in checkpoint_keys:
                        print(f"Warning: Duplicate key '{key}' found in file {i}. Using the latest version.")
                    combined_tensor[key] = f.get_tensor(key)
                    checkpoint_keys.add(key)
        except Exception as e:
            print(f"Error processing file {safe_tensors_path}: {str(e)}")

    print(f"Total number of keys: {len(combined_tensor)}")
    print(f"Number of unique keys: {len(checkpoint_keys)}")

    print(f"Saving combined model to {output_file}")
    torch.save(combined_tensor, output_file)
    print("Done!")

# Usage
check_point_root = './openvla-7b+kinova+b4+lr-1e-05+lora-r64+dropout-500'
output_file = "combined_model.pt"
combine_safetensors_to_pt(check_point_root, f'{check_point_root}/{output_file}')
