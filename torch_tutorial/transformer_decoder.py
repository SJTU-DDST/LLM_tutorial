import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        # x.shape = (2, 16, 64)
        # attn_out.shape = (2, 16, 64)
        
        x = self.ln1(x + attn_out)
        # x.shape = (2, 16, 64)
        ff_out = self.ff(x)
        # ff_out.shape = (2, 16, 64)
        x = self.ln2(x + ff_out)
        # x.shape = (2, 16, 64)
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, num_layers=2, dim_ff=128, max_len=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dim_ff) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # (seq, seq) 下三角mask，防止看到未来token
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        # x: (batch, seq)
        b, seq_len = x.shape # (2, 16)
        x = self.embedding(x) + self.pos_embed[:, :seq_len, :] # (2, 16, 64)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)  # (16, 16)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        # x: (2, 16, 64)
        logits = self.lm_head(x)  # (batch, seq, vocab_size)
        return logits

batch_size = 2
seq_len = 16

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
batch = tokenizer(
    ["Hello world!", "This is a test."],
    return_tensors="pt",    
    padding=True,
    truncation=True,
    max_length=seq_len,
)

vocab_size = tokenizer.vocab_size
print("vocab size:", vocab_size)

input_ids = batch["input_ids"]
print(input_ids)

model = SimpleGPT(vocab_size)
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
out = model(input_ids)
print("logits shape:", out.shape)  # (2, 16, 100)
