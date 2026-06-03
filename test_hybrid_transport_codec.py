import torch

from hybrid_transport_codec import (
    HybridTransportCodec,
    chunk_bytes_uniform,
    tensor_chunk_sizes,
    demo_payload_to_embeddings,
)


def test_chunk_bytes_uniform_splits_payload():
    payload = b"abcdefghij"
    chunks = chunk_bytes_uniform(payload, 4)
    assert chunks == [b"abcd", b"efgh", b"ij"]


def test_tensor_chunk_sizes_matches_chunk_lengths():
    chunks = [b"ab", b"cdef", b"ghi"]
    sizes = tensor_chunk_sizes(chunks)
    assert torch.equal(sizes, torch.tensor([2.0, 4.0, 3.0]))


def test_demo_payload_to_embeddings_shape():
    chunks = [b"abc", b"def"]
    embeddings = demo_payload_to_embeddings(chunks, embedding_dim=8)
    assert embeddings.shape == (2, 8)
    assert embeddings.dtype == torch.float32


def test_hybrid_transport_codec_forward_outputs_expected_keys():
    payload = b"transport codec payload" * 10
    chunks = chunk_bytes_uniform(payload, 16)
    chunk_sizes = tensor_chunk_sizes(chunks)
    embeddings = demo_payload_to_embeddings(chunks, embedding_dim=16)

    model = HybridTransportCodec(embedding_dim=16, epsilon=0.05)
    result = model(embeddings, chunks, chunk_sizes)

    expected_keys = {
        "transformed_states",
        "cost_matrix",
        "gibbs_kernel",
        "transport_plan",
        "curvature_map",
        "codec_scores",
        "encoded_chunks",
    }
    assert expected_keys.issubset(result.keys())

    n = len(chunks)
    assert result["cost_matrix"].shape == (n, n)
    assert result["gibbs_kernel"].shape == (n, n)
    assert result["transport_plan"].shape == (n, n)
    assert result["curvature_map"].shape == (n,)
    assert result["codec_scores"].shape == (n,)
    assert len(result["encoded_chunks"]) == n


def test_encode_chunks_modes_are_valid():
    payload = b"A" * 128
    chunks = chunk_bytes_uniform(payload, 32)
    chunk_sizes = tensor_chunk_sizes(chunks)
    embeddings = demo_payload_to_embeddings(chunks, embedding_dim=8)

    model = HybridTransportCodec(embedding_dim=8)
    result = model(embeddings, chunks, chunk_sizes)

    modes = {entry["mode"] for entry in result["encoded_chunks"]}
    assert modes.issubset({"literal", "reference"})
