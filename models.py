from transformers import EncoderDecoderModel

### HOW TO CREATE SHARED ENCODER DECODER MODEL
shared_bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased", tie_encoder_decoder=True)
