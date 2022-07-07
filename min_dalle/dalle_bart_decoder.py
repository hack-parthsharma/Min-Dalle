from typing import Tuple, List
import torch
from torch import LongTensor, nn, FloatTensor, BoolTensor
torch.set_grad_enabled(False)

from .dalle_bart_encoder import GLU, AttentionBase


class DecoderCrossAttention(AttentionBase):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_mask: BoolTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        attention_state = torch.where(
            token_mask[None, :, None], 
            torch.cat([keys, values]), 
            attention_state
        )
        batch_count = decoder_state.shape[0]
        keys = attention_state[:batch_count]
        values = attention_state[batch_count:]
        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        image_token_count: int,
        head_count: int, 
        embed_count: int,
        glu_embed_count: int
    ):
        super().__init__()
        self.image_token_count = image_token_count
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttention(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)

        self.token_indices = torch.arange(self.image_token_count)
        if torch.cuda.is_available():
            self.token_indices = self.token_indices.cuda()

    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        self_attn_mask = self.token_indices < token_index + 1
        self_attn_mask = self_attn_mask[None][[0] * decoder_state.shape[0]]
        token_mask = self.token_indices == token_index
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state,
            attention_state,
            self_attn_mask,
            token_mask
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state,
            encoder_state,
            attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_count: int,
        image_token_count: int,
        sample_token_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        start_token: int
    ):
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.sample_token_count = sample_token_count
        self.condition_factor = 10.0
        self.image_token_count = image_token_count
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        self.embed_positions = nn.Embedding(image_token_count, embed_count)
        self.layers: List[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                image_token_count,
                attention_head_count,
                embed_count,
                glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)
        self.zero_prob = torch.zeros([1])
        self.token_indices = torch.arange(self.sample_token_count)
        self.start_token = torch.tensor([start_token]).to(torch.long)
        if torch.cuda.is_available():
            self.zero_prob = self.zero_prob.cuda()
            self.token_indices = self.token_indices.cuda()
            self.start_token = self.start_token.cuda()


    def decode_step(
        self,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor
    ) -> Tuple[LongTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index_batched = token_index[[0] * image_count * 2]
        prev_tokens = prev_tokens[list(range(image_count)) * 2]
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        attention_states_new = []
        for i in range(self.layer_count):
            decoder_state, attention_state_layer = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
            attention_states_new.append(attention_state_layer)
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        a = self.condition_factor
        logits: FloatTensor = (
            logits[:image_count, -1] * (1 - a) + 
            logits[image_count:, -1] * a
        )

        top_logits, _ = logits.topk(50, dim=-1)
        probs = torch.where(
            logits < top_logits[:, [-1]],
            self.zero_prob,
            torch.exp(logits - top_logits[:, [0]])
        )
        return probs, torch.stack(attention_states_new)


    def forward(
        self,
        image_count: int,
        text_tokens: LongTensor,
        encoder_state: FloatTensor
    ) -> LongTensor:
        expanded_indices = [0] * image_count + [1] * image_count
        text_tokens = text_tokens[expanded_indices]
        encoder_state = encoder_state[expanded_indices]
        attention_mask = text_tokens.not_equal(1)

        attention_state_shape = (
            self.layer_count,
            image_count * 4,
            self.image_token_count,
            self.embed_count
        )
        attention_state = torch.zeros(attention_state_shape)
        if torch.cuda.is_available(): attention_state = attention_state.cuda()
        
        image_tokens = self.start_token[[0] * image_count]
        image_tokens_sequence: List[LongTensor] = []
        for i in range(self.sample_token_count):
            probs, attention_state = self.decode_step(
                attention_mask = attention_mask,
                encoder_state = encoder_state,
                attention_state = attention_state,
                prev_tokens = image_tokens,
                token_index = self.token_indices[[i]]
            )

            image_tokens = torch.multinomial(probs, 1)[:, 0]
            image_tokens_sequence += [image_tokens]
        
        return torch.stack(image_tokens_sequence).T