# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N` | cache size in MiB, split among the GPUs like the layers (default: 0, disabled) |
| `--moe-cache-cpu-share F` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0) |
| `--moe-cache-max-batch N` | largest ubatch in tokens that uses the cache and its CPU share, larger ones upload the experts they use (default: 32) |

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm none`, `-lm mlock` and `-lm dio`, unless `--no-host`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

A ubatch of more than `--moe-cache-max-batch` tokens uploads every expert it uses, so its cost hardly depends on its size: a few hundred tokens already use most experts of a layer. A smaller ubatch reads the cache and, with the CPU share, computes most misses on the CPU. Raising the limit helps short prompts such as the new tokens of an agent turn. It hurts above the size where uploading everything is faster than the CPU, and those batches also evict the experts used in generation. On an RTX 2070 Max-Q (PCIe 3.0 x8) with Qwen3.8-35B-A3B, `-ub 2048`, a 2900 MiB cache and share 0.95, `--moe-cache-max-batch 256` cut the prefill of 48 to 200 new tokens by 29 to 42% and left longer prompts and generation unchanged; 512 made 512-token ubatches 15% slower.

The CPU share is only used for ubatches that no device would offload: the scheduler places the CPU part, so the cache gives no CPU part to a ubatch that the GPU backend would run from host weights. CUDA offloads from `GGML_OP_OFFLOAD_MIN_BATCH` tokens (default 32), so with a limit above 32 set `GGML_OP_OFFLOAD_MIN_BATCH` to the limit plus one, or the ubatches of 32 tokens up to the limit upload all their misses. Larger ubatches are offloaded as before. In the setup above with the limit at 256, `GGML_OP_OFFLOAD_MIN_BATCH=257` took the misses computed on the CPU in ubatches of more than 8 tokens from 9% back to 89%, cut the upload from 69 to 7.7 GB and made prompts of 48 to 200 new tokens 3 to 13% faster than the SLRU policy did; generation was 10% faster.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant.
- **`--markdown`:** prints Markdown tables to paste in a PR.
