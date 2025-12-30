from GPT.model import GPT2
from prettyPrint import centerPrint, divPrint
import torch
import tiktoken

model_name = 'gpt2-medium'

model = GPT2.from_pretrained(model_name)
tokenizer = tiktoken.get_encoding('gpt2')

text = ''
with open('title.txt', 'r') as file:
    text = file.read()

centerPrint(text)
divPrint()
print(f'Parameter Count: {model.param_count()}')
# os.system('figlet Menu')

while True: 
    inp = input('Enter prompt: ')
    if inp.lower().lstrip() != 'exit':
        enc = torch.tensor(tokenizer.encode(inp), device='mps')
        op = model.generate(enc, 80)
        op = tokenizer.decode(op)
        print(op)
    else:
        break
