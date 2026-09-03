"""Active-learning loss prediction for pseudo-query filtering.

The loss predictor is trained to approximate the ranking loss that a
retriever would suffer on a ``(query, positive, negative)`` triple. During
inference, the predicted loss is used as the active-learning quality score
of the pseudo query.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from igft.filtering.common import load_corpus_jsonl, load_pseudo_queries, write_json


class BertEncoder(nn.Module):
    """BERT mean-pooling encoder used to embed query / positive / negative texts."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**encoded).last_hidden_state  # (B, L, H)
        mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
        embeddings = torch.sum(output * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return embeddings

    def forward(self, query, pos_doc, neg_doc):
        query_emb = self.encode(query)
        pos_emb = self.encode(pos_doc)
        neg_emb = self.encode(neg_doc)
        pos_score = F.cosine_similarity(query_emb, pos_emb)
        neg_score = F.cosine_similarity(query_emb, neg_emb)
        return pos_score, neg_score, query_emb, pos_emb, neg_emb


class LossNet(nn.Module):
    """Small MLP that maps ``(q, pos, neg)`` embeddings to a predicted loss."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc3 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(3 * hidden_dim, 1).to(self.device)

    def forward(self, query_emb, pos_emb, neg_emb):
        query_emb = self.dropout(self.relu(self.fc1(query_emb)))
        pos_emb = self.dropout(self.relu(self.fc2(pos_emb)))
        neg_emb = self.dropout(self.relu(self.fc3(neg_emb)))
        features = torch.cat((query_emb, pos_emb, neg_emb), dim=1)
        return self.relu(self.linear(features))


class LossDataset(Dataset):
    def __init__(self, triples: List[dict]) -> None:
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        triple = self.triples[idx]
        return {
            "query": triple["query"],
            "pos_doc": triple["positive_corpus"],
            "neg_doc": triple["negative_corpus"],
            "cid": triple["cid"],
        }


def build_triples(pseudo_query_file: str, corpus_file: str) -> List[dict]:
    """Pair every pseudo query with a random negative document."""
    corpus = load_corpus_jsonl(corpus_file)
    pseudo_queries = load_pseudo_queries(pseudo_query_file)
    corpus_ids = list(corpus)

    triples = []
    for item in pseudo_queries:
        gold_id = item["cid"]
        if len(corpus_ids) < 2:
            raise ValueError("At least two corpus documents are required to build triples.")
        negative_id = random.choice(corpus_ids)
        while negative_id == gold_id:
            negative_id = random.choice(corpus_ids)
        triples.append(
            {
                "query": item["pseudo_query"],
                "positive_corpus": item["corpus"],
                "negative_corpus": corpus[negative_id],
                "cid": gold_id,
            }
        )
    return triples


def train_main(
    pseudo_query_file: str,
    corpus_file: str,
    lossnet_path: str = "lossnet",
    base_retriever_path: str = "bert-base-uncased",
    batch_size: int = 4,
    epochs: int = 10,
) -> None:
    """Train the loss predictor on pseudo queries with random negatives."""
    encoder = BertEncoder(base_retriever_path)
    lossnet = LossNet().to(encoder.device)
    optimizer = torch.optim.AdamW(lossnet.parameters(), lr=1e-5)
    mse_loss = nn.MSELoss()

    dataset = LossDataset(build_triples(pseudo_query_file, corpus_file))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lossnet.train()
    for _ in range(epochs):
        for batch in tqdm(dataloader, desc="Training loss predictor"):
            pos_score, neg_score, query_emb, pos_emb, neg_emb = encoder(
                batch["query"], batch["pos_doc"], batch["neg_doc"]
            )
            target = torch.ones_like(pos_score).to(encoder.device)
            per_sample_loss = torch.clamp(-target * (pos_score - neg_score), min=0.0)

            predicted_loss = lossnet(query_emb, pos_emb, neg_emb)
            loss = mse_loss(predicted_loss, per_sample_loss.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint_path = f"{lossnet_path}.pt"
    torch.save(lossnet.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}.")


def test_main(
    pseudo_query_file: str,
    corpus_file: str,
    output_file: str,
    lossnet_path: str = "lossnet",
    base_retriever_path: str = "bert-base-uncased",
    batch_size: int = 1,
) -> None:
    """Predict per-query losses and write them as filtering scores."""
    encoder = BertEncoder(base_retriever_path)
    lossnet = LossNet().to(encoder.device)
    lossnet.load_state_dict(torch.load(f"{lossnet_path}.pt", map_location=encoder.device))
    lossnet.eval()

    dataset = LossDataset(build_triples(pseudo_query_file, corpus_file))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting losses"):
            _, _, query_emb, pos_emb, neg_emb = encoder(
                batch["query"], batch["pos_doc"], batch["neg_doc"]
            )
            predicted = lossnet(query_emb, pos_emb, neg_emb).cpu().numpy().tolist()
            for i in range(len(batch["query"])):
                results.append(
                    {
                        "pseudo_query": batch["query"][i],
                        "corpus": batch["pos_doc"][i],
                        "negative_corpus": batch["neg_doc"][i],
                        "cid": batch["cid"][i],
                        "score": predicted[i][0],
                    }
                )

    write_json(results, output_file)
