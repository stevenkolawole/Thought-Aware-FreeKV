# Thought-Aware KV

**Please see [old README](old_README.md) for installation and testing.**

## What we are doing

FreeKV's speculative retrieval assumes adjacent decode-step query vectors are similar
(cosine sim > 0.84 typically). This holds for long-input tasks, but on AIME24 (complex
math reasoning), **43-52% of decode steps trigger correction** (query similarity drops
below τ). Nearly half of all steps, meaning speculation can only go so far!!

ThinKV identifies *why* this happens: reasoning models generate **thought transitions (T segments)**
where the model backtracks or redirects, since this are the precise moments where query
direction changes abruptly.

**Our goal is** to use thought-type classification to *predict* correction events before they happen, rather than detecting them after the fact.

We need to reproduce FreeKV on AIME24 (old readme covers that) and also instrument by adding two logging hooks to log cosine sim at eahc step and log bytes transfereed per `recall` (see [Github Issue](https://github.com/stevenkolawole/Thought-Aware-FreeKV/issues) for details). Then we want to make the minimal system changes that achieve our goals.

### Use FreeKV's existing cosine similarity signal as the thought-type proxy

Initially we wanted to use ThinKV's exact thought-type classifier with KDE calibration etc.. But Flashinfer's uses fused kernels, and they don't return attention weights. So the classifier can't get attention sparsity statistics this way. If we want to persist with this, then we'll have to tinker a whole lot with FreeKV's internals.

Alternatively, we can simply use a keyword labeler: run a regex scan over token texts; ThinKV explicitly notes that T-segments contain tokens like "Wait", "Hmm", "Actually", "But wait", etc; so we basically keyword match for these tokens. But this is crude, and here is another suggestion:

FreeKV already computes `cosine_sim(q_t, q_t-1)` at every decode step for correction. But that signal is a thought-type detector (I know this since we used cosine-similarity to measure semantic agreement in my EMNLP '25 work); they are just not using it that way. Since T-segnments are, by ThinKV's def, moments where the model's query direction changes; a drop in cosine similarity is that change when we measure it directly.

So instead of a separate classifier, we can maitain a running average of cosine similarity and threshold it. This costs **zero**, since the current system already computes it.


## Code Architecture Layers

**Layer 1:** Pytorch Interface (HF/Pytorch) -- `adapter/modeling.py + infer_state.py + pred.py`

**Layer 2:** Python Cache & Kernel Wrappers -- `kv_cache.py + kernels.py + utils.py`

**C++/CUDA impl.:** `flashinfer_ops.h + *.cu + thread_pool.cpp`


## File-by-File Breakdown

I can walk us through specific files (if needed) during our meeting...

## Key Algorithms

1. Sink + Window + Retrieved Strategy

	Every decode step, the KV budget is divided into:

	- Sink pages (e.g., first 512 tokens): always kept since they are critical for attention patterns
	- Window pages (e.g., last 512 tokens): sliding window for locality
	- Retrieved pages: dynamically selected by scoring

2. Digest Cache

	Instead of scoring full KV tensors, FreeKV stores max/min bounds per page (2× compression). `estimate_scores()` approximates relevance as `query x (max + min) / 2`. This lets the system score thousands of pages on-GPU in milliseconds.

3. Speculative Retrieval + Correction

	- Speculative: overlap the CPU->GPU data transfer for step N+1 with the attention computation of step N (using async CUDA streams + thread pool)
	- Correction: before using speculatively fetched pages, compute cosine similarity between current and previous query. If the query changed significantly (below threshold), throw out the speculative fetch and do a synchronous re-fetch -- trading latency for correctness during abrupt query shifts

4. Double-Buffered Recall

	The `cuda_cpy` backend uses two pinned-memory buffers alternately, so one buffer is being filled from CPU while the other is being read by the GPU kernel; this hides transfer latency.

## Summary of their Data Flow

**During Prefill,**

	`QKV projection -> RoPE -> append to GPU cache -> backup to CPU -> build digest cache -> attention -> evict excess if budget is exceeded`

**During Decode (speculative)**

	`Wait for previous recall events -> [optionally] correct if query has changed -> run attention on recalled pages -> launch next step's recall async in thread pull`