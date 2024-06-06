import torch.nn as nn
import torch
import math

# TODO1


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.d_v = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.output = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, dim = x.shape

        # get query key value

        Q = self.query(x)  # (batch_size, num_image_tokens, dim)
        K = self.key(x)  # (batch_size, num_image_tokens, dim)
        V = self.value(x)   # (batch_size, num_image_tokens, dim)

        Q = Q.view(batch_size, num_image_tokens,
                   self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, num_image_tokens,
                   self.num_heads, self.d_k).permute(0, 2, 3, 1)
        V = V.view(batch_size, num_image_tokens,
                   self.num_heads, self.d_v).permute(0, 2, 1, 3)

        scale_score = torch.matmul(Q, K)/(self.d_k**0.5)
        att_weight = nn.functional.softmax(scale_score, dim=-1)
        att_weight = self.attn_drop(att_weight)

        output = torch.matmul(att_weight,V).permute(0,2,1,3).contiguous().view(batch_size,num_image_tokens,self.dim)
        output = self.output(output)
        return output
        # raise Exception('TODO1!')


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )

    def forward(self, input):
        return super().forward(input)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )

    def forward(self, input):
        return super().forward(input)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)

        x = x + attn
        x = self.LayerNorm1(x)

        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
