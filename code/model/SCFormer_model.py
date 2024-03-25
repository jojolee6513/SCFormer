import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch import nn, Tensor
from typing import Optional
from .wd_loss import SinkhornDistance

DEVICE = 'cuda:0'
torch.cuda.set_device(0)
#####################################

# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)
def apply_rotary(x, sinusoidal_pos):
    sin, cos = sinusoidal_pos
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
    # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
    y = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)
    return y[:, :, 0:-1]
#######################################
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension, init_weights=True):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        # if not same_shape:
        #     strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            # nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
            # x = self.conv3(x)
        # return F.relu(out + x)
        return F.relu(out + x)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # tgt2 = self.self_attn(q=q, k=k, v=tgt, mask=tgt_mask)[1]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def pos_add(self, x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        sin = sin.to(DEVICE)
        cos = cos.to(DEVICE)
        x = x.permute(1, 0, 2)
        zero = torch.zeros([x.shape[0], 64, 1]).to(DEVICE)
        x = torch.cat((x, zero), dim=-1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        x = torch.stack([y1, y2], dim=-1).flatten(-2, -1)
        x = x[:, :, :-1]
        return x.permute(1, 0, 2)

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        q = k = self.pos_add(tgt2, query_pos)
        ########
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def pos_add(self, x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        sin = sin.to(DEVICE)
        cos = cos.to(DEVICE)
        x = x.permute(1, 0, 2)
        zero = torch.zeros([x.shape[0], 64, 1]).to(DEVICE)
        x = torch.cat((x, zero), dim=-1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        x = torch.stack([y1, y2], dim=-1).flatten(-2, -1)
        x = x[:, :, :-1]
        return x.permute(1, 0, 2)

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        #################################
        tgt2_pos = self.pos_add(tgt2, query_pos)
        memory_pos = self.pos_add(memory, pos)
        tgt2 = self.multihead_attn(query=tgt2_pos,
                                   key=memory_pos,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        ################################
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


embed_positions = RoFormerSinusoidalPositionalEmbedding(128, 50)
sinusoidal_pos = embed_positions(64, 50)[
     None, :, :
].chunk(2, dim=-1)

class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2
    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim=64,
            enforce_input_project=True,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # self.pe_layer = PositionEmbeddingSine(num_pos_feats=32, normalize=True)
        # self.embed_positions = RoFormerSinusoidalPositionalEmbedding(128, 50)
        # self.sinusoidal_pos = self.embed_positions(64, 50)[
        #      None, :, :
        # ].chunk(2, dim=-1)

        ######random mask
        self.random_mask_bands = np.arange(0, 64)
        np.random.seed(123)
        np.random.shuffle(self.random_mask_bands)


        # define Transformer decoder here
        self.inchannels = in_channels
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        # mapping layer
        self.target_mapping = Mapping(103, 100)  # PU 103  PC:102 sa: 204 IP:200
        self.source_mapping = Mapping(128, 100)  # chikusei 128
        self.resdentialBlock = Block(in_channel=100, out_channel=64, same_shape=False)

        self.wd_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        self.mmd = MMD_loss(kernel_type='linear')
        self.mse = nn.MSELoss()

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes)

        # decoder
        self.classifier = nn.Sequential(
            nn.Linear(49*64, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x=None, y=None, y_label=None, domain='source', wd_s=None, wd_t=None, episode=None):
        if self.training:
            src = []
            pos = []
            if x is not None:
                if domain == 'source':
                    x = self.source_mapping(x)
                    y = self.source_mapping(y)
                else:
                    x = self.target_mapping(x)
                    y = self.target_mapping(y)

                x = self.resdentialBlock(x)
                y = self.resdentialBlock(y)

                src.append(x.flatten(2))
                src.append(y.flatten(2))

            else:

                wd_s = self.source_mapping(wd_s)
                wd_t = self.target_mapping(wd_t)
                wd_s = self.resdentialBlock(wd_s)
                wd_t = self.resdentialBlock(wd_t)

                # pos.append(self.pe_layer(wd_s.size(), None).flatten(2))
                src.append(wd_s.flatten(2))
                src.append(wd_t.flatten(2))

            # flatten NxCxHxW to L(S)xNxE-----HWxNxC
            src[0] = src[0].permute(1, 0, 2)
            src[1] = src[1].permute(1, 0, 2)
            # pos[-1] = pos[-1].permute(1, 0, 2)
            output_x = src[0]
            output_y = src[1]

            # mask = torch.zeros([16, 64])
            # mask[]

            MSE_loss_sup, MSE_loss_que = 0, 0
            for i in range(self.num_layers):
                attn_mask = torch.zeros([self.inchannels, self.inchannels])
                # Sequentail
                # attn_mask[i*int(self.inchannels/self.num_layers):(i+1)*int(self.inchannels/self.num_layers), :] = 1
                # Random
                for k in range(16 * i, 16 * (i + 1)):
                    attn_mask[self.random_mask_bands[k], :] = 1
                #################################################
                attn_mask = attn_mask.transpose(1, 0)
                attn_mask = (attn_mask == 1).bool().cuda()  # 64*64

                copy_sup = output_x[16 * i:16 * (i + 1), :, :].clone()
                copy_qury = output_y[16*i:16*(i+1), :, :].clone()

                output_x = self.transformer_cross_attention_layers[i](
                    output_x, output_x,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,
                    pos=sinusoidal_pos, query_pos=sinusoidal_pos,
                )

                output_y = self.transformer_cross_attention_layers[i](
                    output_y, output_y,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,
                    pos=sinusoidal_pos, query_pos=sinusoidal_pos,
                )

                output_x = self.transformer_self_attention_layers[i](
                    output_x, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=sinusoidal_pos,
                )

                output_y = self.transformer_self_attention_layers[i](
                    output_y, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=sinusoidal_pos,
                )
                # FFN
                output_x = self.transformer_ffn_layers[i](output_x)
                output_y = self.transformer_ffn_layers[i](output_y)
                MSE_loss_sup += self.mse(copy_sup, output_x[16 * i:16 * (i + 1), :, :])
                MSE_loss_que += self.mse(copy_qury, output_y[16*i:16*(i+1), :, :])

            if x is not None:
                output_x = output_x.permute(1, 0, 2)
                output_y = output_y.permute(1, 0, 2)

                # inter-domain shift
                I = np.identity(int(output_x.size(1)))
                I = np.expand_dims(I, axis=0)
                I = np.tile(I, (output_x.size(0), 1, 1))
                I = torch.from_numpy(I).cuda()
                # S = super_pixel_fea
                S = output_x[:, :, int(output_x.size(2)/2)]
                S = S.unsqueeze(-1)
                S_t = S.transpose(2, 1)
                S1 = S_t @ S
                S1_I = torch.linalg.inv(S1)
                S_star = S1_I @ S_t
                P = I - (S @ S_star) + 1e-5
                R = output_y.double()
                R = R[:, :, int(R.size(2)/2)]

                intra_domain_loss_list = []

                #
                for i in range(R.size(0)):
                    label = y_label[i]
                    intra_domain_loss_list.append(torch.norm((P[label]) @ (R[i])))

                intra_domain_loss = max(intra_domain_loss_list)

                # classification
                output_x = output_x.reshape(
                    (output_x.shape[0], output_x.shape[1]*output_x.shape[2]))
                output_y = output_y.reshape(
                    (output_y.shape[0], output_y.shape[1] * output_y.shape[2]))
                output_x = self.classifier(output_x)
                output_y = self.classifier(output_y)

                return output_x, output_y, intra_domain_loss,

            else:
                # domain shift loss
                output_x = output_x.permute(1, 0, 2)
                output_y = output_y.permute(1, 0, 2)
                output_x = output_x.reshape(
                    (output_x.shape[0], output_x.shape[1] * output_x.shape[2]))
                output_y = output_y.reshape(
                    (output_y.shape[0], output_y.shape[1] * output_y.shape[2]))
                loss, _, _ = self.wd_loss(output_x, output_y)
                return loss, #(MSE_loss_sup/4 + MSE_loss_que/4)

        else:
            # test
            src = []
            pos = []
            x = self.target_mapping(x)
            x = self.resdentialBlock(x)

            src.append(x.flatten(2))
            # flatten NxCxHxW to HWxNxC
            src[0] = src[0].permute(1, 0, 2)
            output_x = src[0]
            for i in range(self.num_layers):
                attn_mask = torch.zeros([self.inchannels, self.inchannels])
                # S
                # attn_mask[i*int(self.inchannels/self.num_layers):(i+1)*int(self.inchannels/self.num_layers)] = 1
                # R
                for k in range(16 * i, 16 * (i + 1)):
                    attn_mask[self.random_mask_bands[k], :] = 1
                attn_mask = attn_mask.transpose(1, 0)
                attn_mask = (attn_mask == 1).bool().cuda()

                output_x = self.transformer_cross_attention_layers[i](
                    output_x, output_x,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,
                    pos=sinusoidal_pos, query_pos=sinusoidal_pos
                )
                output_x = self.transformer_self_attention_layers[i](
                    output_x, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=sinusoidal_pos,
                )
                # FFN
                output_x = self.transformer_ffn_layers[i](output_x)

            output_x = output_x.permute(1, 0, 2)
            output_x = output_x.reshape(
                (output_x.shape[0], output_x.shape[1] * output_x.shape[2]))
            output_x = self.classifier(output_x)
            return output_x


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss


def sp_center_iter(supp_feat, supp_mask, sp_init_center, n_iter):
    '''
            :param supp_feat: A Tensor of support feature, (C, H, W)
            :param supp_mask: A Tensor of support mask, (1, H, W)
            :param sp_init_center: A Tensor of initial sp center, (C + xy, num_sp)
            :param n_iter: The number of iterations
            :return: sp_center: The centroid of superpixels (prototypes)
           
    '''

    supp_mask = supp_mask.unsqueeze(0)
    c_xy, num_sp = sp_init_center.size()
    _, h, w = supp_feat.size()
    h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
    w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda()
    supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)
    # supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()] # (C + xy) x num_roi
    sup_label = supp_mask[0, int(supp_mask.size(0)/2), int(supp_mask.size(1)/2)]
    supp_feat_roi = supp_feat[:, (supp_mask == sup_label).squeeze()]  # (C + xy) x num_roi

    num_roi = supp_feat_roi.size(1)
    supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
    sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

    for i in range(n_iter):
        # Compute association between each pixel in RoI and superpixel
        if i == 0:
           sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
        else:
           sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)

        assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
        dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
        feat_dist = dist[:-2, :, :].sum(0)
        spat_dist = dist[-2:, :, :].sum(0)
        total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
        p2sp_assoc = torch.neg(total_dist).exp()  # e^(-total_dist)
        p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

        sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp
        sp_center = sp_center.sum(1)

    return sp_center[:-2, :]

# if __name__ == '__main__':
#     torch.cuda.set_device(0)
#     torch.manual_seed(123)
#
#     support = torch.randn((9, 128, 9, 9)).cuda()
#     query = torch.randn((171, 128, 9, 9)).cuda()
#     source = torch.randn((128, 128, 9, 9)).cuda()
#     target = torch.randn((128, 103, 9, 9)).cuda()
#     target_label = torch.randint(0, 8, [171,]).cuda()
#
#     encoder = MultiScaleMaskedTransformerDecoder(
#         in_channels=64,
#         mask_classification=True,
#         num_classes=9,
#         hidden_dim=49,
#         num_queries=1,
#         nheads=7,
#         dim_feedforward=49 * 4,
#         dec_layers=4,
#         pre_norm=True,
#     )
#     encoder.cuda()
#     encoder.train()
#     result = encoder(x=support, y=query, y_label=target_label, episode=str(1))
#     wd_loss = encoder(wd_s=source, wd_t=target)
#     print(wd_loss)

