# Gemini-Constructor-CLI-agent
Gemini CLI

## Hybrid transport codec usage

The repository includes `hybrid_transport_codec.py`, a PyTorch-based scaffold that combines:
- embedding transformation,
- geodesic-style pairwise cost computation,
- Gibbs kernel generation,
- Sinkhorn transport scaling,
- curvature extraction,
- heuristic codec scoring for chunk routing.

### What the module does

`HybridTransportCodec` is designed to connect the notebook's transport pipeline to a codec-style chunk scoring workflow.

Key outputs from `forward(...)`:
- `transformed_states`
- `cost_matrix`
- `gibbs_kernel`
- `transport_plan`
- `curvature_map`
- `codec_scores`
- `encoded_chunks`

### Minimal example

```python
from hybrid_transport_codec import (
    HybridTransportCodec,
    chunk_bytes_uniform,
    tensor_chunk_sizes,
    demo_payload_to_embeddings,
)

payload = b"example payload for transport-aware codec scoring" * 8
chunks = chunk_bytes_uniform(payload, chunk_size=32)
chunk_sizes = tensor_chunk_sizes(chunks)
embeddings = demo_payload_to_embeddings(chunks, embedding_dim=16)

model = HybridTransportCodec(embedding_dim=16, epsilon=0.05)
result = model(embeddings, chunks, chunk_sizes)

print(result["gibbs_kernel"].shape)
print(result["codec_scores"])
print(result["encoded_chunks"][0])
```

### Input expectations

- `input_tensors`: a `torch.Tensor` of shape `(num_chunks, embedding_dim)`
- `chunk_bytes`: a `List[bytes]` with one entry per chunk
- `chunk_sizes`: a tensor of per-chunk byte lengths

The number of rows in `input_tensors` should match the number of chunks and chunk sizes.

### Important note on terminology

The `gibbs_kernel` in this module is a **mathematical transport kernel**:

`K = exp(-C / epsilon)`

It is not related to the Linux kernel. If you later integrate operating-system-level optimizations such as `mmap`, `io_uring`, or NUMA-aware placement, those belong in a separate system layer.

### Running tests

If `pytest` is installed, run:

```bash
pytest -q
```
