import torch
import torch.nn as nn
from typing import Type
from torch import Tensor
import math
from typing import Tuple
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class Attention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        downsample_rate: int = 1,   # 下采样
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # qkv获取
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # B,N_heads,N_tokens,C_per_head
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B,N_heads,N_tokens,C_per_head
        # Scale
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        # # B,N_tokens,C
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,         # 输入channel
        hidden_dim: int,        # 中间channel
        output_dim: int,        # 输出channel
        num_layers: int,        # fc的层数
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int, 
                 num_heads:int,
                 mlp_dim:int,
                 activation:Type[nn.Module] = nn.ReLU,
                 skip_first_layer_pe: bool = False) ->None:
        super().__init__()
        self.self_attn = Attention(embedding_dim,num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_exp_to_image = Attention(
            embedding_dim,num_heads
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim,mlp_dim,activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_exp = Attention(
            embedding_dim,num_heads
        )
        self.skip_first_layer_pe = skip_first_layer_pe
    def forward(self,exp_q_emb,img_emb,exp_first_emb) ->Tuple[Tensor,Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=exp_q_emb,k=exp_q_emb,v=exp_q_emb)
        else:
            q = exp_q_emb + exp_first_emb
            attn_out = self.self_attn(q = q,k=q,v=exp_q_emb)
            queries = exp_q_emb + attn_out
        queries = self.norm1(queries)

        q = queries + exp_first_emb
        k = img_emb
        attn_out = self.cross_attn_exp_to_image(q=q,k=k,v=img_emb)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + exp_first_emb
        k = img_emb
        attn_out = self.cross_attn_image_to_exp(q=k,k=q,v=queries)
        keys = img_emb + attn_out
        keys = self.norm4(keys)
        return queries,keys

class CrossTransformer(nn.Module):
    def __init__(self,
                 depth:int,
                 embedding_dim:int,
                 num_heads:int,
                 mlp_dim:int,
                 activation:Type[nn.Module]=nn.ReLU) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    skip_first_layer_pe=(i==0)
                )
            )

        self.final_attn_exp_to_img = Attention(embedding_dim,num_heads)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self,img_emb,orig_emb):
        bs,c,h,w = img_emb.shape
        img_emb = img_emb.flatten(2).permute(0,2,1)

        queries = orig_emb
        keys = img_emb
        for layer in self.layers:
            queries,keys = layer(
                exp_q_emb = queries,
                img_emb = keys,
                exp_first_emb = orig_emb
            )
        q = queries + orig_emb
        k = keys

        attn_out = self.final_attn_exp_to_img(q = q,k = k,v = keys)

        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        queries = queries.squeeze()
        return queries

class Exp_Guide(nn.Module):
    def __init__(self,
                 attention_dim,
                 depth,
                 num_heads,
                 mlp_dim,
                 exp_dim,
                 final_mlp_depth,
                 final_mlp_hidden_dim):
        super().__init__()
        self.crosstransformer = CrossTransformer(depth,attention_dim,num_heads,mlp_dim)
        self.exp_dim = exp_dim
        self.exp_prediction_head = MLP(attention_dim,final_mlp_hidden_dim,exp_dim,final_mlp_depth)

    def forward(self,img_emb,exp_emb):
        exp_prediction = self.crosstransformer(img_emb,exp_emb)
        exp_prediction = self.exp_prediction_head(exp_prediction)
        return exp_prediction

if __name__ == '__main__':

    model = Exp_Guide(368,2,8,512,785,3,1024)
    a = torch.randn(2,368,7,7)
    b = torch.randn(2,1,368)
    out = model(a,b)
    print(out.shape)