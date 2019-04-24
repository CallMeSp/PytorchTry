import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('data')

# Tokenized input
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
text_3 = [text_1,text_2]
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
tokenized_text_3 = [tokenizer.tokenize(sent) for sent in text_3]
# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
indexed_tokens_3 = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text_3]

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])
tokens_tensor_3 = torch.tensor([indexed_tokens_3])
print(tokens_tensor_1)
print(tokens_tensor_2)
print(tokens_tensor_3)
# Load pre-trained model (weights)


model = TransfoXLModel.from_pretrained('data')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to('cuda')
tokens_tensor_2 = tokens_tensor_2.to('cuda')
model.to('cuda')

with torch.no_grad():
    # Predict hidden states features for each layer
    hidden_states_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

# Load pre-trained model (weights)
model = TransfoXLLMHeadModel.from_pretrained('data')
model.eval()

# If you have a GPU, put everything on cuda
if torch.cuda.is_available():
    tokens_tensor_1 = tokens_tensor_1.to('cuda')
    tokens_tensor_2 = tokens_tensor_2.to('cuda')
    model.to('cuda')


with torch.no_grad():
    # Predict all tokens
    predictions_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)
    print(predictions_1)
    print(predictions_2)

# get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'who'


exit(0)