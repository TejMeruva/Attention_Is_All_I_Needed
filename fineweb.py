from datasets import load_dataset
import tiktoken
import multiprocessing as mp
import os
import numpy as np
from tqdm import tqdm

'''
- dowloads the FineWeb-edu (10B) dataset.
- tokenizes it and saves it into shards
'''

# tokenize
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc))
    return np.array(tokens).astype(np.uint16)


def main():
    # constants
    shard_size = 13107200  # micro_batch * 8 * 100
    n_proc = os.cpu_count()
    save_folder = 'fineweb_data'

    # download dataset
    data = load_dataset(
        path='HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split='train'
    )

    # test = np.arange(0, 1000).reshape(-1, 4).tolist()
    # # print(test)

    with mp.Pool(n_proc) as pool:
        shard_tokens = np.zeros((shard_size, ))
        pos = 0
        shard_ind = 0
        prog_bar = tqdm(total=len(shard_tokens), unit='tokens',
                        desc=f'shard{shard_ind}')
        for tokens in pool.imap(tokenize, data['text'][:int(5e5)], 16):
            inc = len(tokens)
            if pos + inc <= len(shard_tokens):  # writing in same shard
                shard_tokens[pos:pos+inc] = tokens
                pos += inc
                prog_bar.update(inc)
            else:  # creating new shard
                # finishing old shard
                inc = len(shard_tokens) - pos
                shard_tokens[pos: pos+inc] = tokens[:inc]
                # assert prog_bar is None, f'len of tokens to be added {len(tokens)} as pos {pos} is more that {len(shard_tokens)}'
                prog_bar.update(inc)

                # saving old shard
                np.save(
                    file=os.path.join(save_folder, f'shard{shard_ind}'),
                    arr=shard_tokens
                )
                shard_ind += 1

                # starting new shard
                rem = pos + len(tokens) - len(shard_tokens)
                shard_tokens[:rem] = tokens[inc:]
                pos = rem
                prog_bar = tqdm(total=len(shard_tokens),
                                unit='tokens', desc=f'shard{shard_ind}')

        else:
            # saving the remaining tokens in the last shard
            np.save(
                file=os.path.join(save_folder, f'shard{shard_ind}'),
                arr=shard_tokens[:pos]
            )


if __name__ == "__main__":
    mp.freeze_support()   # optional but recommended
    main()
