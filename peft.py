from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login

### LORA CONFIGS
# This is usually the best resulting target modules (all the projection layers)
target_modules = ["q_proj", "o_proj", 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj']
# If for some reason you run into a bug where these modules do not exist do this as the target_modules
target_modules = None # --> This will automatically select the modules for you.
# This is also an alternative that targets all linear layers
target_modules= 'all-linear'

lora_config = LoraConfig(
        r=8,
        target_modules=target_modules,
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
# Note that if you add the bnb config, you do not need to explicitly put the model on device.
tokenizer = AutoTokenizer('path/to/checkpoint', cache_dir = 'path/to/cache', quantization_config = bnb_config)
model = AutoModel('path/to/checkpoint', cache_dir = 'path/to/cache', quantization_config = bnb_config)

### WRAPPING MODEL TO PEFT WITH LORA CONFIGS
model = get_peft_model(model, lora_config)

### PRINT TRAINIABLE PARAMETERS AFTER WRAPPING MODEL
model.print_trainable_parameters()


### For peft wrapped models you have to access atrributes like 
print(model.base_model.model)
print(model.base_model.model.model.embed_tokens)
etc...
 



