# Fork divergences from upstream

This fork carries a few deliberate departures from upstream behaviour (`ggml-org/llama.cpp`). Re-check each one at every upstream sync: upstream may have fixed the reason for it, or a later upstream change may have to be adapted again. New features of the fork (PrismML types, MoE cache max batch, kv-mean-center, ...) are documented in their own pages and are not listed here.

| Area | Upstream | This fork | Why | Since |
|---|---|---|---|---|
| Tool-call arguments with invalid UTF-8 (`common/chat-peg-parser.cpp`) | #31861 repairs invalid bytes in tool-call arguments to U+FFFD | the PEG mapper keeps the raw argument text, so the parse of a corrupted tool call fails | a corrupted argument must not become an executable call (`pw\xeb\x82d` would run as `pw�d`); reasoning and content are still repaired (#244), `test_chat_malformed_utf8` checks both | sync of 2026-11-08 |
| Recurrent rollback bound (`llama_memory_recurrent::seq_rm`) | rollbacks are limited to the snapshots of the last ubatch (`rs_max`, #31368) | the same, except in replay mode (`LLAMA_GDN_REPLAY`), which keeps its own bound of `min(n - 1, n_rs_seq)` records per ubatch | replay keeps the update records of a ubatch longer than `n_rs_seq + 1` tokens, so it can still roll it back | sync of 2026-11-08 |
| Qwen3.5 MTP draft chain (`llama_context::mtp_chain_support`) | no draft chain | the chain is off when the unified-cache kv_rows attention (#29510) is on, and MTP drafts one step per graph | the chain slices a dense `[n_kv, n_tokens]` mask; kv_rows masks cell lists. `LLAMA_KV_ROWS_DISABLE=1` brings the chain back | sync of 2026-11-08 |
| CUDA sparse flash attention (`fattn.cu`, `fattn-mma-f16.cuh`) | sparse variants for every K/V type the MMA kernel takes | sparse variants only for F16 K/V | the fork's quantized K/V MMA path (`type_KV`) has no sparse variant | before 2026-11 |
| WebGPU mat-vec rows (`GGML_WEBGPU_MUL_MAT_VEC_MAX_ROWS`) | 4 | 8 | fork #301 routes up to 8 columns through the mat-vec kernel | before 2026-11 |
| Type ids (`ggml.h`, `llama.h`, `gguf-py`) | `GGML_TYPE_COUNT` follows the last upstream type | `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143`, `GGML_TYPE_COUNT = 144`; ftypes 141 to 143 | ids of the published PrismML GGUFs; a new upstream type must stay below 142 | before 2026-11 |
