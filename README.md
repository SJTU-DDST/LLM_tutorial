# LLM_tutorial
The original transformer architecture includes encoder and decoder.  For now, `torch_tutorial/transformer_decoder.py` implements the basic architecture of the decoder. 

Basically, the architecture is composed by one tokenizer, one input embedding, Decoderblock, and language model head. For decoderblock, it can be seen in the `forward` function that it is composed by one attention layer, one layer normalization after residual connection, one feedforward layer, and another layer normalization with residual connection. 

