import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil  else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        conv_seq_len=configs.seq_len
        for l in range(configs.e_layers-1):
            conv_seq_len=(conv_seq_len+2)//2

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * conv_seq_len, configs.num_class)

    def classification(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]
