import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.Utils import aeq


class BatchConv1d(nn.Module):

    def __init__(self, q_size, k_size, kernel_width = 3, use_mask=False):

        super(BatchConv1d, self).__init__()
        self.q_size = q_size
        self.k_size = k_size
        self.kernel_width = kernel_width
        self.padding_width = (self.kernel_width - 1) / 2
        if self.kernel_width != 0:
            # (batch_size * q_len) * q_size -> (batch_size * q_len) * k_size * kernel_width
            self.q_to_kernel = nn.Linear(q_size, k_size * self.kernel_width, bias=True)
            self.q_to_bias = nn.Linear(q_size, 1, bias=True)
            #nn.init.xavier_normal(self.q_to_kernel.weight)
            self.bias_b = nn.Parameter(torch.FloatTensor(1))
            nn.init.normal(self.bias_b)
            #self.bias_b.fill_(0.0)
            self.use_mask = use_mask
            if self.use_mask is True:
                self.kernel_mask = torch.zeros(kernel_width)
                self.kernel_mask[:(self.kernel_width + 1) / 2] = 1
                self.kernel_mask = Variable(self.kernel_mask, requires_grad=False).cuda()

    def forward(self, q, k):

        batch_size, q_len, q_size = q.size()
        batch_size, k_len, k_size = k.size()

        if self.kernel_width == 0:
            scores = torch.matmul(q, k.transpose(1, 2))
            return scores
        # conv1d(input, weight)
        #### original parameter dimensions ####
        # input.size = batch_size * in_channels * input_width
        # weight.size = out_channels * in_channels * kernel_width
        # output.size = batch_size * out_channels * output_width
        #### use groups to make every sentence in batch should have its own kernel #####
        # groups = batch_size
        # input.size = 1 * (batch_size * k_size) * kv_len
        # weight.size = batch_size *  k_size * kernel_width
        # bias.size = batch_size
        # output.size = 1 * batch_size * kv_len
        # q: (B*n_head, L_q, d_k), k: (B*n_head, L_k, d_k)
        inp = k.transpose(2, 1).contiguous().view(1, batch_size * k_size, k_len)
        #q_flat = q.view(batch_size * q_len, q_size)
        #kernel = self.q_to_kernel(q_flat).view(batch_size * q_len, k_size, self.kernel_width)
        #bias   = self.q_to_bias  (q_flat).view(batch_size * q_len)
        kernel = self.q_to_kernel(q).view(batch_size * q_len, k_size, self.kernel_width)
        bias   = self.q_to_bias  (q).view(batch_size * q_len)
        if self.use_mask is True: kernel = kernel * self.kernel_mask[None, None, :]
        conv_res = nn.functional.conv1d(inp,
                            kernel,
                            bias=bias,
                            groups=batch_size,
                            padding=self.padding_width) # (1, batch_size * q_len, k_len)
        conv_res_b = conv_res + self.bias_b
        a_ij = conv_res_b.view(batch_size, q_len, k_len) # kv_len * batch_size

        return a_ij


class Phraselize(nn.Module):

    def __init__(self, phrase_len, word_dim):
        #TODO: assert phrase_len % 2 == 1
        super(Phraselize, self).__init__()
        self.phrase_len = phrase_len
        self.word_dim   = word_dim
        self.phrase_dim = phrase_len * word_dim
        if 0 != phrase_len:
            self.linear_phrase = nn.Linear(self.phrase_dim, self.word_dim)
        else:
            self.linear_phrase = None

    def forward(self, inputs):
        if 0 == self.phrase_len:
            return inputs
        else:
            batch, length, dim = inputs.size()
            gram = list(range(self.phrase_len))
            for i in range(self.phrase_len):
                #gram[i] = inputs.data.new_zeros(batch, length+self.phrase_len, dim)
                gram[i] = Variable(inputs.data.new(
                    batch, length+self.phrase_len, dim).fill_(0.0))
                gram[i][:, i:i+length, :] = inputs
            grams = torch.stack(gram, dim=-1)
            grams = grams[:, :length, :, :].contiguous()
            grams = grams.view(batch, length, self.phrase_dim)
            phrase = self.linear_phrase(grams)
            return phrase

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """
    def __init__(self, head_count, model_dim, dropout=0.1,
                 use_attcnn=0, use_mask=False):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.use_mask = use_mask
        self.use_attcnn = use_attcnn
        if 1 == self.use_attcnn:
            multi_kernel_width = True
            if multi_kernel_width:
                self.kws = [0,0,0,0,3,3,5,7]
                self.kernels = nn.ModuleList(
                    [
                        BatchConv1d(self.dim_per_head, self.dim_per_head,
                                    kw, use_mask=use_mask)
                        for kw in self.kws
                    ]
                )
            else:
                self.kernel_width = 5
                self.bconv1d = BatchConv1d(d_k, d_k, kernel_width,
                                           use_mask=use_mask)
                self.kernels = nn.ModuleList([self.bconv1d] * head_count)
        elif 2 == self.use_attcnn:
            self.que_lens = [0,0,0,3,3,3,5,5]
            self.key_lens = [0,0,3,0,3,3,5,5]
            self.val_lens = [0,0,0,0,0,3,0,5]
            self.que_phrases = nn.ModuleList(
                [ Phraselize(l, self.dim_per_head) for l in self.que_lens ]
            )
            self.key_phrases = nn.ModuleList(
                [ Phraselize(l, self.dim_per_head) for l in self.key_lens ]
            )
            self.val_phrases = nn.ModuleList(
                [ Phraselize(l, self.dim_per_head) for l in self.val_lens ]
            )

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.model_dim % 8, 0)
        if mask is not None:
            batch_, q_len_, k_len_ = mask.size()
            aeq(batch_, batch)
            aeq(k_len_, k_len)
            aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        if 1 == self.use_attcnn:
            attns = list(range(self.head_count))
            for i in range(self.head_count):
                attns[i] = self.kernels[i](query_up[:, i, :, :],
                                             key_up[:, i, :, :])
            scores = torch.stack(attns, dim=1)
        elif 2 == self.use_attcnn:
            q_ps = list(range(self.head_count))
            k_ps = list(range(self.head_count))
            v_ps = list(range(self.head_count))
            for i in range(self.head_count):
                q_ps[i] = self.que_phrases[i](query_up[:, i, :, :])
                k_ps[i] = self.key_phrases[i](key_up[:, i, :, :])
                v_ps[i] = self.key_phrases[i](value_up[:, i, :, :])
            q_p = torch.stack(q_ps, dim=1)
            k_p = torch.stack(k_ps, dim=1)
            v_p = torch.stack(v_ps, dim=1)
            value_up = v_p
            scores = torch.matmul(q_p, k_p.transpose(2, 3))
        else:
            scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)
        # CHECK
        batch_, q_len_, d_ = output.size()
        aeq(q_len, q_len_)
        aeq(batch, batch_)
        aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()
        # END CHECK
        return output, top_attn
