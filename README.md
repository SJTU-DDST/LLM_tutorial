# LLM_tutorial
The original transformer architecture includes encoder and decoder.  For now, `torch_tutorial/transformer_decoder.py` implements the basic architecture of the decoder. 

Basically, the architecture is composed by one tokenizor, one input embedding, Decoderblock, one softmax layer, and language model head. 

For one tokenizor, it will map the words into token id. 

For input embedding, it will convert the token id to a dence vector.

For decoderblock, it can be seen in the `forward` function that it is composed by one attention layer, one layer normalization after residual connection, one feedforward layer, and another layer normalization with residual connection. 

The language model head is a MLP. It map the hidden size of each word into the vocabulary size.

The softmax will calculate the probability of each word and we will choose one with the max probability. 

## Mask
Something special is that, we should mask the future token to avoid the model to see later information in advanced. For example, the prompt is `Hello World` and we ask the model to generate the word after `World`. In training part, since we want to let model have the ability to generate words without information in the future, the model shouldn't see any word after `World`. \
` mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)`\
This code generate a lower triangular matrix. All elements under the diagonal (including diagonal) are 1 and the others are `-inf`. 

## Position Encoding
After we got the input embedding, we should concat the input position information to it because the model should know the order of words within one sentence. \
`x = self.embedding(x) + self.pos_embed[:, :seq_len, :]` 