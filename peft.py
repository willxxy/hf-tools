from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login

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
tokenizer = AutoTokenizer('path/to/checkpoint', cache_dir = 'path/to/cache', quantization_config = bnb_config)
model = AutoModel('path/to/checkpoint', cache_dir = 'path/to/cache', quantization_config = bnb_config)

### PRIVATE MODELS ACCESS witgh API KEY
print('Loading API key')
with open('./api_keys.txt', 'r') as file:
    file_contents = file.readlines()
api_key = file_contents[0]

login(token = api_key)
