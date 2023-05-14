"""
The code is based on the child-sum Tree-LSTM from https://github.com/JetBrains-Research/embeddings-for-trees
However, the encoder part has been changed. My version is a version that considers the order of the children
nodes of the tree. So it is not a child-sum LSTM anymore.
"""

from abc import abstractmethod
from multiprocessing import context
from typing import Union, Tuple, Optional, List, Dict
from argparse import ArgumentParser

import torch
from torch import nn
import dgl


class LuongAttention(nn.Module):
    """Implement Luong Global Attention with general score function
    https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, units: int):
        super().__init__()
        self.attn = nn.Linear(units, units, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate attention weights
        :param hidden: [batch size; units]
        :param encoder_outputs: [batch size; seq len; units]
        :param mask: [batch size; seq len]
        :return: [batch size; seq len]
        """
        batch_size, seq_len = mask.shape
        # [batch size; units]
        attended_hidden = self.attn(hidden)
        # [batch size; seq len]
        score = torch.bmm(
            encoder_outputs, attended_hidden.view(batch_size, -1, 1)
        ).squeeze(-1)
        score += mask

        # [batch size; seq len]
        weights = torch.softmax(score, dim=1)
        return weights


def segment_sizes_to_slices(sizes: torch.Tensor) -> List:
    cum_sums = torch.cumsum(sizes, dim=0)
    slices = [slice(0, cum_sums[0])]
    slices += [slice(start, end) for start, end in zip(cum_sums[:-1], cum_sums[1:])]
    return slices


def cut_into_segments(
    data: torch.Tensor,
    sample_sizes: torch.LongTensor,
    mask_value: float = -1e9,
    only_data=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut data from common tensor into samples part with padding to longest sequence.
    Also return mask with value on pad places.
    :param data: [n contexts; units]
    :param sample_sizes: [batch size]
    :param mask_value:
    :return: [batch size; max context len; units], [batch size; max context len]
    """

    batch_size = len(sample_sizes)
    max_context_len = max(sample_sizes)

    batched_contexts = data.new_zeros((batch_size, max_context_len, data.shape[-1]))
    attention_mask = data.new_zeros((batch_size, max_context_len))

    context_slices = segment_sizes_to_slices(sample_sizes)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, sample_sizes)):
        batched_contexts[i, :cur_size] = data[cur_slice]
        attention_mask[i, cur_size:] = mask_value

    if only_data is True:
        return batched_contexts
    return batched_contexts, attention_mask


class Encoder(nn.Module):
    def __init__(self, config, vocab_db):
        super().__init__()
        self.vocab_db = vocab_db
        self._encoder_size = config.encoder_size

        self._dropout = nn.Dropout(config.encoder_dropout)

        self._W_iou = nn.Linear(config.embedding_size, 3 * config.encoder_size)
        self._U_iou = nn.Linear(
            config.encoder_size, 3 * config.encoder_size, bias=False
        )
        self._W_f = nn.Linear(config.embedding_size, config.encoder_size)
        self._U_f = nn.Linear(config.encoder_size, config.encoder_size, bias=False)
        self._out_linear = nn.Linear(config.encoder_size, config.decoder_size)
        self._norm = nn.LayerNorm(config.decoder_size)
        self._tanh = nn.Tanh()
        self.encoder_layers = config.encoder_num_layers
        self._aggregate_lstm_uh = nn.LSTM(
            config.encoder_size * 3,
            config.encoder_size * 3,
            num_layers=config.encoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )
        self._aggregate_lstm_fc = nn.LSTM(
            config.encoder_size,
            config.encoder_size,
            num_layers=config.encoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )

    def message_func(self, edges: dgl.udf.EdgeBatch) -> Dict:

        h_f = self._U_f(edges.src["h"])
        x_f = edges.dst["x_f"]
        f = torch.sigmoid(x_f + h_f)
        return {"Uh": self._U_iou(edges.src["h"]), "fc": edges.src["c"] * f}

    def reduce_func(self, nodes: dgl.udf.NodeBatch) -> Dict:

        token = self._W_iou(nodes.data["embed"]).unsqueeze(1)
        state_shape = [self.encoder_layers, token.shape[0], token.shape[2]]
        h0_uh = torch.zeros(state_shape)
        c0_uh = torch.zeros(state_shape)
        seq = nodes.mailbox["Uh"]

        seq = torch.cat([token, seq, token], dim=1)

        Uh, (hn_uh, cn) = self._aggregate_lstm_uh(seq, (h0_uh, c0_uh))

        token = self._W_f(nodes.data["embed"]).unsqueeze(1)
        state_shape = [self.encoder_layers, token.shape[0], token.shape[2]]
        h0_fc = torch.zeros(state_shape)
        c0_fc = torch.zeros(state_shape)
        seq = nodes.mailbox["fc"]
        seq = torch.cat([token, seq, token], dim=1)

        Fc, (hn_fc, cn) = self._aggregate_lstm_fc(seq, (h0_fc, c0_fc))
        # return {"Uh_sum": hn_uh[-1], "fc_sum": hn_fc[-1]}
        return {"Uh_sum": Uh[:, -1, :], "fc_sum": Fc[:, -1, :]}

    def apply_node_func(self, nodes: dgl.udf.NodeBatch) -> Dict:
        iou = nodes.data["x_iou"] + nodes.data["Uh_sum"]
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + nodes.data["fc_sum"]
        h = o * torch.tanh(c)

        return {"h": h, "c": c}

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:

        x = self._dropout(graph.ndata["embed"])

        # init matrices for message propagation
        number_of_nodes = graph.number_of_nodes()
        graph.ndata["x_iou"] = self._W_iou(x)
        graph.ndata["x_f"] = self._W_f(x)
        graph.ndata["h"] = graph.ndata["embed"].new_zeros(
            (number_of_nodes, self._encoder_size)
        )
        graph.ndata["c"] = graph.ndata["embed"].new_zeros(
            (number_of_nodes, self._encoder_size)
        )
        graph.ndata["Uh_sum"] = graph.ndata["embed"].new_zeros(
            (number_of_nodes, 3 * self._encoder_size)
        )
        graph.ndata["fc_sum"] = graph.ndata["embed"].new_zeros(
            (number_of_nodes, self._encoder_size)
        )

        # propagate nodes
        dgl.prop_nodes_topo(
            graph,
            message_func=self.message_func,
            reduce_func=self.reduce_func,
            apply_node_func=self.apply_node_func,
        )

        # [n nodes; encoder size]
        h = graph.ndata.pop("h")
        # [n nodes; decoder size]
        out = self._tanh(self._norm(self._out_linear(h)))
        return out


DecoderState = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class BaseDecoderStep(nn.Module):
    @abstractmethod
    def get_initial_state(
        self, encoder_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> DecoderState:
        """Use this function to initialize decoder state, e.g. h and c for LSTM decoder.

        :param encoder_output: [batch size; max seq len; encoder size] -- encoder output
        :param attention_mask: [batch size; max seq len] -- mask with zeros on non pad elements
        :return: decoder state: single tensor of tuple of tensors
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(
        self,
        input_token: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_state: DecoderState,
    ) -> Tuple[torch.Tensor, torch.Tensor, DecoderState]:
        """Perform one decoder step based on input token and encoder output.

        :param input_token: [batch size] -- tokens from previous step or from target sequence
        :param encoder_output: [batch size; max seq len; encoder size] -- encoder output
        :param attention_mask: [batch size; max seq len] -- mask with zeros on non pad elements
        :param decoder_state: decoder state from previous or initial step
        :return:
            [batch size; vocab size] -- logits
            [batch size; encoder seq length] -- attention weights
            new decoder state
        """
        raise NotImplementedError()


class LSTMDecoderStep(BaseDecoderStep):
    def __init__(self, config, output_size: int, pad_idx: Optional[int] = None):
        super().__init__()
        self._decoder_num_layers = config.decoder_num_layers

        self._target_embedding = nn.Embedding(
            output_size, config.embedding_size, padding_idx=pad_idx
        )

        self._attention = LuongAttention(config.decoder_size)

        self._decoder_lstm = nn.LSTM(
            config.embedding_size,
            config.decoder_size,
            num_layers=config.decoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )
        self._dropout_rnn = nn.Dropout(config.rnn_dropout)

        self._concat_layer = nn.Linear(
            config.decoder_size * 2, config.decoder_size, bias=False
        )
        self._norm = nn.LayerNorm(config.decoder_size)
        self._projection_layer = nn.Linear(config.decoder_size, output_size, bias=False)

    def get_initial_state(
        self, encoder_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> DecoderState:
        initial_state: torch.Tensor = encoder_output[:, 0, :]
        initial_state = initial_state.unsqueeze(0).repeat(
            self._decoder_num_layers, 1, 1
        )
        return initial_state, initial_state

    def forward(
        self,
        input_token: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_state: DecoderState,
    ) -> Tuple[torch.Tensor, torch.Tensor, DecoderState]:
        h_prev, c_prev = decoder_state

        # [batch size; 1; embedding size]
        embedded = self._target_embedding(input_token).unsqueeze(1)

        # hidden -- [n layers; batch size; decoder size]
        # output -- [batch size; 1; decoder size]
        rnn_output, (h_prev, c_prev) = self._decoder_lstm(embedded, (h_prev, c_prev))
        rnn_output = self._dropout_rnn(rnn_output)

        # [batch size; context size]
        attn_weights = self._attention(h_prev[-1], encoder_output, attention_mask)

        # [batch size; 1; decoder size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_output)

        # [batch size; 2 * decoder size]
        concat_input = torch.cat([rnn_output, context], dim=2).squeeze(1)

        # [batch size; decoder size]
        concat = self._concat_layer(concat_input)
        concat = self._norm(concat)
        concat = torch.tanh(concat)

        # [batch size; vocab size]
        output = self._projection_layer(concat)

        return output, attn_weights, (h_prev, c_prev)


class Decoder(nn.Module):

    _negative_value = -1e9

    def __init__(
        self,
        decoder_step: BaseDecoderStep,
        output_size: int,
        sos_token: int,
        teacher_forcing: float = 0.0,
    ):
        super().__init__()
        self._decoder_step = decoder_step
        self._teacher_forcing = teacher_forcing
        self._out_size = output_size
        self._sos_token = sos_token

    def forward(
        self,
        encoder_output: torch.Tensor,
        segment_sizes: torch.LongTensor,
        output_size: int,
        target_sequence: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate output sequence based on encoder output

        :param encoder_output: [n sequences; encoder size] -- stacked encoder output
        :param segment_sizes: [batch size] -- size of each segment in encoder output
        :param output_size: size of output sequence
        :param target_sequence: [batch size; max seq len] -- if passed can be used for teacher forcing
        :return:
            [output size; batch size; vocab size] -- sequence with logits for each position
            [output size; batch size; encoder seq length] -- sequence with attention weights for each position
        """
        batch_size = segment_sizes.shape[0]

        # encoder output -- [batch size; max context len; units]
        # attention mask -- [batch size; max context len]
        batched_encoder_output, attention_mask = cut_into_segments(
            encoder_output, segment_sizes, self._negative_value
        )

        decoder_state = self._decoder_step.get_initial_state(
            batched_encoder_output, attention_mask
        )

        # [output size; batch size; vocab size]
        output = batched_encoder_output.new_zeros(
            (output_size, batch_size, self._out_size)
        )
        output[0, :, self._sos_token] = 1

        # [output size; batch size; encoder seq size]
        attentions = batched_encoder_output.new_zeros(
            (output_size, batch_size, attention_mask.shape[1])
        )

        # [batch size]
        current_input = batched_encoder_output.new_full(
            (batch_size,), self._sos_token, dtype=torch.long
        )
        for step in range(1, output_size):
            current_output, current_attention, decoder_state = self._decoder_step(
                current_input, batched_encoder_output, attention_mask, decoder_state
            )
            output[step] = current_output
            attentions[step] = current_attention
            if (
                self.training
                and target_sequence is not None
                and torch.rand(1) <= self._teacher_forcing
            ):
                current_input = target_sequence[step]
            else:
                current_input = output[step].argmax(dim=-1)

        return output, attentions


class TreeLSTM_Model(nn.Module):
    def __init__(
        self,
        db,
        config,
        num_vocabs,
        pad_idx,
        sos_idx,
        eos_idx,
    ):
        super(TreeLSTM_Model, self).__init__()
        self._config = config
        self.x_size = config.model.embedding_size
        self.num_vocabs = num_vocabs
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.embedding = nn.Embedding(
            num_vocabs, config.model.embedding_size, padding_idx=self.pad_idx
        )

        self.encoder = Encoder(config.model, db.vocab_db)

        decoder_step = LSTMDecoderStep(config.model, num_vocabs, self.pad_idx)
        self.decoder = Decoder(
            decoder_step, num_vocabs, self.sos_idx, config.train.teacher_forcing
        )

    def encode(self, trees, return_embed=False):
        """
        return_embed is true:
            Return the aggregated (average) representation for each graph
        else:
            return the encoded presentation in dgl batched graph
        """
        batched_trees = trees
        batched_trees.ndata["embed"] = self.embedding(batched_trees.ndata["id"])
        encoded_nodes = self.encoder(batched_trees)
        segment_sizes = trees.batch_num_nodes()
        if return_embed is True:
            segmented_encoded_nodes, attention_mask = cut_into_segments(
                encoded_nodes, segment_sizes
            )
            result = segmented_encoded_nodes.sum(dim=1) / (attention_mask == 0).sum(
                dim=1, keepdim=True
            )
            return result
        else:
            return encoded_nodes

    def decode(self, encoded_nodes, segment_sizes, output_length, labels=None):
        if labels is not None:
            output_logits, _ = self.decoder(
                encoded_nodes, segment_sizes, output_length, labels
            )
        else:
            output_logits, _ = self.decoder(encoded_nodes, segment_sizes, output_length)
        return output_logits

    def forward(self, trees, labels=None):
        encoded_nodes = self.encode(trees)
        segment_sizes = trees.batch_num_nodes()
        output_length = max(segment_sizes) + 2
        output_logits = self.decode(
            encoded_nodes, segment_sizes, output_length, labels=labels
        )
        return output_logits, encoded_nodes
