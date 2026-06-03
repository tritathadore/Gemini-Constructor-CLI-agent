from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn


class HybridTransportCodec(nn.Module):
    """
    Hybrid transport + codec pipeline.

    This module:
    1. transforms input embeddings,
    2. computes a pairwise geodesic-style cost matrix,
    3. builds a Gibbs kernel,
    4. derives a Sinkhorn transport plan,
    5. scores chunks for codec routing / packing.

    Notes:
    - The "Gibbs kernel" here is a mathematical transport kernel, not the Linux kernel.
    - Codec scoring is heuristic and meant as an integration scaffold for future repo-specific logic.
    """

    def __init__(self, embedding_dim: int, epsilon: float = 0.05, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim

        self.bridge_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )

    def compute_cost_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute a pairwise angular/geodesic-style cost matrix from input positions.
        """
        norms = torch.norm(positions, p=2, dim=1, keepdim=True)
        normalized_positions = positions / (norms + 1e-12)
        dot_product = torch.clamp(
            torch.mm(normalized_positions, normalized_positions.t()),
            -0.99999,
            0.99999,
        )
        return torch.acos(dot_product)

    def compute_gibbs_kernel(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropic transport kernel K = exp(-C / epsilon).
        """
        return torch.exp(-cost_matrix / self.epsilon)

    def run_sinkhorn_scaling(
        self,
        gibbs_kernel: torch.Tensor,
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute an approximately doubly-stochastic transport plan with Sinkhorn scaling.
        """
        n = gibbs_kernel.size(0)
        dtype = gibbs_kernel.dtype
        device = gibbs_kernel.device

        mu = torch.ones((n, 1), dtype=dtype, device=device) / n
        nu = torch.ones((n, 1), dtype=dtype, device=device) / n
        u = mu.clone()
        v = nu.clone()

        for _ in range(max_iter):
            u_prev = u.clone()
            u = mu / (torch.mm(gibbs_kernel, v) + 1e-12)
            v = nu / (torch.mm(gibbs_kernel.t(), u) + 1e-12)
            if torch.max(torch.abs(u - u_prev)) < tolerance:
                break

        return u * gibbs_kernel * v.t()

    def extract_ollivier_ricci_curvature(
        self,
        cost_matrix: torch.Tensor,
        transport_plan: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a curvature-like score from the transport plan and cost matrix.
        """
        wasserstein_distance = torch.sum(transport_plan * cost_matrix, dim=1)
        mean_distance = torch.mean(cost_matrix, dim=1)
        return 1.0 - (wasserstein_distance / (mean_distance + 1e-12))

    def score_chunks_for_codec(
        self,
        transport_plan: torch.Tensor,
        chunk_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score chunks for codec routing.

        Higher scores loosely indicate stronger affinity relative to size and can be used
        to prefer reference-style or deduplicated storage strategies.
        """
        affinity = torch.sum(transport_plan, dim=1)
        normalized_sizes = chunk_sizes / (torch.max(chunk_sizes) + 1e-12)
        return affinity / (normalized_sizes + 1e-12)

    def encode_chunks(
        self,
        chunk_bytes: List[bytes],
        codec_scores: torch.Tensor,
        reference_threshold: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Encode chunks using a simple heuristic:
        - high-score chunks use "reference" mode with a small payload stub
        - lower-score chunks use "literal" mode with the original bytes
        """
        encoded = []
        for i, raw in enumerate(chunk_bytes):
            score = float(codec_scores[i].item())
            mode = "reference" if score > reference_threshold else "literal"
            payload = raw[: min(len(raw), 128)] if mode == "reference" else raw
            encoded.append(
                {
                    "chunk_index": i,
                    "mode": mode,
                    "payload": payload,
                    "score": score,
                }
            )
        return encoded

    def forward(
        self,
        input_tensors: torch.Tensor,
        chunk_bytes: List[bytes],
        chunk_sizes: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Execute the full hybrid transport + codec scoring pipeline.
        """
        transformed = self.bridge_network(input_tensors)
        cost_matrix = self.compute_cost_matrix(transformed)
        gibbs_kernel = self.compute_gibbs_kernel(cost_matrix)
        transport_plan = self.run_sinkhorn_scaling(gibbs_kernel)
        curvature_map = self.extract_ollivier_ricci_curvature(cost_matrix, transport_plan)
        codec_scores = self.score_chunks_for_codec(transport_plan, chunk_sizes)
        encoded_chunks = self.encode_chunks(chunk_bytes, codec_scores)

        return {
            "transformed_states": transformed,
            "cost_matrix": cost_matrix,
            "gibbs_kernel": gibbs_kernel,
            "transport_plan": transport_plan,
            "curvature_map": curvature_map,
            "codec_scores": codec_scores,
            "encoded_chunks": encoded_chunks,
        }


def chunk_bytes_uniform(payload: bytes, chunk_size: int) -> List[bytes]:
    """
    Split bytes into fixed-size chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]


def tensor_chunk_sizes(chunks: List[bytes], device: torch.device | None = None) -> torch.Tensor:
    """
    Build a tensor of chunk sizes for codec scoring.
    """
    return torch.tensor([len(chunk) for chunk in chunks], dtype=torch.float32, device=device)


def demo_payload_to_embeddings(chunks: List[bytes], embedding_dim: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Convert byte chunks into simple deterministic numeric embeddings.

    This is a placeholder embedding strategy intended to make the module runnable
    without requiring an external tokenizer or model.
    """
    rows = []
    for chunk in chunks:
        vec = torch.zeros(embedding_dim, dtype=torch.float32)
        for i, value in enumerate(chunk[:embedding_dim]):
            vec[i] = float(value) / 255.0
        rows.append(vec)

    if not rows:
        return torch.zeros((0, embedding_dim), dtype=torch.float32, device=device)

    embeddings = torch.stack(rows)
    if device is not None:
        embeddings = embeddings.to(device)
    return embeddings
