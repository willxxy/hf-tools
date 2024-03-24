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
