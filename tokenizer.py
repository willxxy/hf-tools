from transformers import AutoTokenizer AutoModel, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
tokenizer = AutoTokenizer('path/to/checkpoint')
model = AutoModel('path/to/checkpoint')

### PRINTS ALL OF THE SPECIAL TOKENS
print(tokenizer.special_tokens_map)

### ADDS NEW TOKENS TO SPECIAL TOKENS MAP
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

### ADDS CUSTOM TOKENS
tokenizer.add_tokens(custom_tokens)

### RESIZES MODEL TOKEN EMBEDDINGS
model.resize_token_embeddings(len(tokenizer))


### LORA CONFIGS
lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", 'k_proj', 'v_proj', 
                        'gate_proj', 'up_proj', 'down_proj'],
        bias = 'none',
        lora_dropout=0.01,
        task_type = TaskType.CAUSAL_LM,
        )

### BYTESANDBINARIES CONFIGS
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

### ADDING CACHE PATH TO MODEL
tokenizer = AutoTokenizer('path/to/checkpoint', cache_dir = 'path/to/cache')
model = AutoModel('path/to/checkpoint', cache_dir = 'path/to/cache')

### ADDING BNB
