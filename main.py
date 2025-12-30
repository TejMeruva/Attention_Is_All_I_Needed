from GPT.model import GPT2
from prettyPrint import centerPrint, divPrint
import torch
import os
import tiktoken
import joblib

model_name = 'gpt2'

model = GPT2.from_pretrained(model_name)
tokenizer = tiktoken.get_encoding('gpt2')

models = [GPT2.from_pretrained('gpt2'), 
          GPT2.from_pretrained('gpt2-medium'),
          joblib.load('GPT2_124M_01.pkl'),
          GPT2()
          ]

text = ''
with open('title.txt', 'r') as file:
    text = file.read()

centerPrint(text)
divPrint()
# print(f'Parameter Count: {model.param_count()}')
print()
os.system('figlet Menu')
menus = ''
menus += '1: Choose Model\n'
menus += '2: Chat\n'
menus += '3: Exit'
print(menus)

while True: 
    inp = input('>>> ')
    match inp:
        case '1':
            options = ''
            options += '0: GPT2 (HuggingFace)\n'
            options += '1: GPT2-medium (HuggingFace)\n'
            options += '2: GPT2 (self-trained)\n'
            options += '3: GPT2 (untrained)'
            print(options)
            ind = int(input('Model Choice: '))
            model = models[ind]
            print(f'model switched!')
            print(f'parameters: {round(model.param_count()/1.e6, 2)}M')
        case '2':
            prompt = input('Enter seed: ')
            if inp.lower().lstrip() != 'exit':
                enc = torch.tensor(tokenizer.encode(prompt), device='mps')
                op = model.generate(enc, 100)
                op = tokenizer.decode(op)
                print(op)
        case '3':
            break
        case _:
            print('Invalid Menu Index!')
