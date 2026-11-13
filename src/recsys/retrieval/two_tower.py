"""Two-tower retrieval model trained with in-batch negatives."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from recsys.retrieval.faiss_index import EmbeddingRetriever


class Tower(nn.Module):
    """An id embedding followed by a small MLP, producing a unit-length vector."""

    def __init__(self, n_ids: int, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_ids, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.embedding(ids)), dim=-1)


class TwoTowerRetriever:
    """Score a (user, item) pair as the cosine similarity of the two tower outputs.

    Training uses in-batch negatives: for a batch of positive pairs, every other item in the
    batch is a negative for each user, and a softmax over the batch is trained to pick the
    true item. Items that appear twice in a batch are masked so the same item is never both positive and negative for a row.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int, output_dim: int,
                 epochs: int, batch_size: int, learning_rate: float, temperature: float,
                 seed: int = 0):
        self.n_users, self.n_items = n_users, n_items
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.temperature, self.seed = learning_rate, temperature, seed
        torch.manual_seed(seed)  # seed before building the towers so initial weights are reproducible
        self.user_tower = Tower(n_users, embedding_dim, output_dim)
        self.item_tower = Tower(n_items, embedding_dim, output_dim)
        self.retriever: EmbeddingRetriever | None = None
        self.loss_history: list[float] = []

    def fit(self, train: pd.DataFrame) -> "TwoTowerRetriever":
        self.loss_history = []
        users = torch.as_tensor(train["user_id"].to_numpy(), dtype=torch.long)
        items = torch.as_tensor(train["item_id"].to_numpy(), dtype=torch.long)
        parameters = [*self.user_tower.parameters(), *self.item_tower.parameters()]
        optimiser = torch.optim.Adam(parameters, lr=self.learning_rate)

        for _ in range(self.epochs):
            order = torch.randperm(len(users))
            epoch_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                loss = self._batch_loss(users[batch], items[batch])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            self.loss_history.append(epoch_loss / len(order))

        self.retriever = self._build_retriever()
        return self

    def _batch_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        logits = self.user_tower(users) @ self.item_tower(items).T / self.temperature
        same_item = items[:, None] == items[None, :]
        is_positive = torch.eye(len(items), dtype=torch.bool)
        logits = logits.masked_fill(same_item & ~is_positive, float("-inf"))
        return F.cross_entropy(logits, torch.arange(len(items)))

    @torch.no_grad()
    def _build_retriever(self) -> EmbeddingRetriever:
        self.user_tower.eval()
        self.item_tower.eval()
        user_vectors = self.user_tower(torch.arange(self.n_users)).numpy()
        item_vectors = self.item_tower(torch.arange(self.n_items)).numpy()
        return EmbeddingRetriever(user_vectors, item_vectors)

    def recommend(self, users: np.ndarray, k: int, seen):
        return self.retriever.recommend(users, k, seen)
