from GPT2 import GPT2, GPTConfig
from transformers import GPT2LMHeadModel

sd = GPT2LMHeadModel.from_pretrained('gpt2').state_dict()
model = GPT2(GPTConfig())

for item in sd.items():
    print(item[0], item[1].shape, sep=': ')

sd = model.state_dict()
print()
for item in sd.items():
    print(item[0], item[1].shape, sep=': ')