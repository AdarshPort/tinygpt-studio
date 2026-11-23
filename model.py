import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(keys.size(-1))

        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:sequence_length, :sequence_length] == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = attention_weights @ values
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embed, block_size, dropout):
        super().__init__()

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                head_size=head_size,
                n_embed=n_embed,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(n_head)
        ])

        self.output_projection = nn.Linear(n_head * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)

        output = self.output_projection(concatenated)
        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()

        head_size = n_embed // n_head

        self.attention = MultiHeadAttention(
            n_head=n_head,
            head_size=head_size,
            n_embed=n_embed,
            block_size=block_size,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(n_embed, dropout)

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTMini(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embed=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embed=n_embed,
                n_head=n_head,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(n_layer)
        ])

        self.final_layer_norm = nn.LayerNorm(n_embed)
        self.language_model_head = nn.Linear(n_embed, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape

        token_embeddings = self.token_embeddings(token_ids)

        positions = torch.arange(sequence_length, device=token_ids.device)
        position_embeddings = self.position_embeddings(positions)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)

        logits = self.language_model_head(x)

        loss = None

        if targets is not None:
            batch_size, sequence_length, vocab_dimension = logits.shape

            logits = logits.reshape(
                batch_size * sequence_length,
                vocab_dimension,
            )

            targets = targets.reshape(batch_size * sequence_length)

            loss = F.cross_entropy(logits, targets)

        return logits, loss


    @torch.no_grad()
    def generate(self, start_token_ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        token_ids = start_token_ids

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]

            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids