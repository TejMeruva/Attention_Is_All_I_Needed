**16 December 2025**
- installed `GithHub Desktop` (had been using CLI like a hobo all these months)
- found a tutorial repo
- tried writing with attention first (didn't work out)

**17 December 2025**
- finished the position encoding and embedding module
- learnt to keep track of the `tensor sizes` as was mentioned in the tutorial as a common tip.
- made a note in `Notability`
- wrote comments frequently to keep track of what's going on.
- learnt what `droupout` is
- wrote the `transformerAttention` class and learnt about attention.
- finished writing the FeedForward Network, assumed that different ffn is used for different position, turns out not so.
- Finished the `Encoder` module.
- Finished the `Decoder` module.
- Learnt what a `causal mask`  is.
- finished combining the encoder and decoder (did by mylsef)
- finished the `full transformer`!
- i am not satisfied--i need to train it right now
- i google how to train a transformer to be a chatbot
- apparently you don't even need the encoder stack.
- i will have to make a new file altogether.
- 24M parameters: crazy
- learnt why `right shifting` is needed.

**18 Decemeber 2025**
- tried the Colab TPU 
- couldn't use the local transformer module
- copy pasted the code instead
- learnt about `tokenizer` from `transformers`.
- the fuck is an `attention mask`?

**20 December 2025**
- welp Imma do it again--to learn it better. This time i am not gonna take help.
- i used Notability to keep track of the tensor shapes. 
- i kwpt using a terminal, using pyton in interactive mode to see what outputs tensor operation were giving me.
- finished `Embedder`

**21 December 2025**
- finished the Transformer `Attention` and `Encoder`.
- realized that taking a break made it so easy that I was able to write the attention class all by myself.
- finished `Decoder`

**22 December 2025**
- discovered [Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)
- discovered [GPT-2 Architecture](https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b)

**23 December 2025**
- almost gave up--thinking that i won't be able to train it. 
- got convinced to follow the video by Andrej Karpathy.
- got the video idea:
  ```
  I finished PyTorch Basics
  ...
  Let's build a transformer.
  ...
  I learnt about the transformer
  ...
  Let's build GPT.
  ```
- watched AK video till 59:00 and decided to write out GPT-2.
- implementing the GPT-2 model myself.
  - learnt how unlike th original transformer, in GPT-2, even the positional encoding is embedding that is learnt.
  - learnt about dataclasses (insane)
  - better than writing down all the arguments millions of times.
  - using the structure of `GPT2LMHeadModel` from HuggingFace.
  - finished entrie GPT2-small (124M). only training left. 
  - initially got 163M params instead of 124M 
  - was because of one line: 
    - self.transformer.wte.weight = self.lm_head.weight
  - then got really close
    - mine: 124439808 
    - his: 124475904
  - GPTed it (funny how i am a transformer how to make it)
  - found difference in vocab size.

**29 Decemeber 2025**
-  corrected the GPT model a bit 
-  loded the weights 
-  output was gibberish comapred to AK
-  gpted more and more.
-  found that one number. one number in the GPT2Attention module was 3 instead of 2. (in the concatenation of the heads.)
-  

**30 December 2025**
- rewrote `from_pretrained()` myself.
- made a CLI.
- tried using gpt2-xl (fucking 6.43GB)
- tried gpt2-medium instead.
- now I have to train the model from scratch instead of using the pre-trained weights.
- followed AK's video chapter by chapter. Trying to learn and do everything by myself.
- trained using mini shakespeare 
- got loss of 6.34
- still blurts garbage, better than untrained though.
- learnt about TF32.
-  learnt about tflops and how the mps comapres to cuda.
- using mps, FP32: ~4-5s per batch
- using mps, TF32, ~3s per batch
- learnt about bfloat implementaion using torch.autocast.
- using bfloat got ~4s per epoch.
- i got insanely higher tokens per second (+300) by reducing, number of tokens in each batch. (1024 -> 256)
- found sweet spot at 512 tokens in each idx with 8 idx's.
- using torch.compile
- removed .contiguous() from dataset.py. got the time down to ~3.5s per iteration and 1130 tok/sec.
- got 1050 tok/sec even when using 1024 n_embed when using flash attention (crazy) (4 idx per batch, 1024 tokens per idx)
- got 1060 tok/sec using 50304 vocab_size
- 330 tok/sec to 830 tok/sec (2.5x faster overall)
- used the hyper parameters used by GPT3.
- got 1000 tok/sec without using torch.compile
- Ak was getting 173k tok/sec.
- almost 170x my speed.
- got the lowest loss of 1.9 and tok rate of 1100 using gradient accululation.
- learnt and implemented weight decay. It is necessary to prevent the over dependance on one weight.
- implemented correct gradient accumulation acc to AK.