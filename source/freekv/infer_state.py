from typing import List, Union, Dict, Tuple, Optional
from collections import defaultdict
import os

import torch
from torch import Tensor

from .kv_cache import KvPool, KvCache
from . import kernels
from . import utils
from .utils import ThoughtType
import atexit

Digest = Tuple[Tensor, Tensor]


class InferState:
    def __init__(
        self,
        n_layers,
        n_qo_heads,
        n_kv_heads,
        head_dim,
        page_size,
        dtype: torch.dtype,
        device: torch.device,
        page_budgets: Union[int, List[int]] = None,
        n_max_pages=None,
        n_unlimited_layers=None,
        n_max_bytes=None,
        page_topks: Union[int, List[int]] = None,
        n_max_cpu_pages=None,
        n_max_cpu_bytes=None,
        n_sink_pages=2,
        n_win_pages=2,
        use_sparse_attn=False,
        group_size=None,
        n_groups=None,
        cpu_layout="NHD",
        spec_ret=False,
        thread_pool_size=2,
        n_recall_stream=2,
        recall_impl=None,
        corr=None,
        corr_impl=None,
        corr_max_batch=16,
        log_dir=None,
        thought_ema_alpha=0.1,
        thought_tau_r=0.84,
        thought_tau_t=0.6,
        **kwargs,
    ) -> None:
        self.n_layers = n_layers
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        if n_max_pages is None:
            assert n_max_bytes is not None
            n_max_pages = n_max_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        if n_max_cpu_pages is None:
            assert n_max_cpu_bytes is not None
            n_max_cpu_pages = n_max_cpu_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        if n_unlimited_layers is None:
            n_unlimited_layers = 0
        if not isinstance(page_budgets, (list, tuple)):
            page_budgets = [None] * n_unlimited_layers + [page_budgets] * (
                n_layers - n_unlimited_layers
            )
        full_attn_layers_env = os.getenv("FULLKV_LAYERS")
        if full_attn_layers_env:
            full_attn_layers = [
                int(x.strip()) for x in full_attn_layers_env.split(",") if x.strip()
            ]
            for layer_idx in full_attn_layers:
                assert 0 <= layer_idx < n_layers, (
                    f"FULL_ATTN_LAYERS has out-of-range layer index {layer_idx}, "
                    f"valid range is [0, {n_layers - 1}]"
                )
                page_budgets[layer_idx] = None
        if page_topks is None:
            page_topks = [b and b // 2 for b in page_budgets]
        elif not isinstance(page_topks, (list, tuple)):
            page_topks = [None] * n_unlimited_layers + [page_topks] * (
                n_layers - n_unlimited_layers
            )

        self.page_size = page_size
        self.n_max_pages = n_max_pages
        self.layer2budget = page_budgets
        self.budget2layers = defaultdict(list)
        for i, b in enumerate(page_budgets):
            self.budget2layers[b].append(i)
        self.layer2topk = page_topks
        self.n_sink_pages = n_sink_pages
        self.n_win_pages = n_win_pages
        assert n_win_pages >= 2
        self.use_sparse_attn = use_sparse_attn

        for i in range(n_layers):
            b = self.layer2budget[i]
            k = self.layer2topk[i]
            if b is not None:
                assert k is not None and k < b
                k = k - n_sink_pages - (n_win_pages - 1)
                assert k > 0
                self.layer2topk[i] = k

        self.layout = "NHD"
        self._i32 = dict(dtype=torch.int32, device=self.device)
        self._u8 = dict(dtype=torch.uint8, device=self.device)
        self._fp = dict(dtype=self.dtype, device=self.device)
        self._bool = dict(dtype=torch.bool, device=self.device)
        self._cpu_bool = dict(dtype=torch.bool, device=torch.device("cpu"))

        self._pool = KvPool(n_max_pages, page_size, n_kv_heads, head_dim, dtype, device)
        self.kv_caches: List[KvCache] = [None] * self.n_layers
        self.dg_caches: List[KvCache] = [None] * self.n_layers
        self._cpu_pool = KvPool(
            n_max_cpu_pages, page_size, n_kv_heads, head_dim, dtype, 
            torch.device("cpu"), layout=cpu_layout
        )
        self.cpu_kv_caches: List[KvCache] = [None] * self.n_layers
        self.topk_dout: Tensor = None
        self.topk_iout: Tensor = None
        self.topk_newi: Tensor = None
        self.topk_rids: Tensor = None
        self.topk_buff: Tensor = None

        self.kv_last_page_len = 0
        self.kv_last_page_lens: Tensor = None
        self.kv_indptrs_tab: Dict[int, Tensor] = {b: None for b in self.budget2layers}
        self.kv_decode_indptrs_tab: Dict[int, Tensor] = {
            b: None for b in self.budget2layers
        }

        self.dg_last_page_len = 0
        self.dg_last_page_lens: Tensor = None
        self.dg_indptrs: Tensor = None

        # maybe larger for larger batch
        wbufs = [torch.empty(64 * 1024 * 1024, **self._u8) for _ in self.budget2layers]
        self.prefill_handler = kernels.BatchPrefillWithPagedKVCacheWrapper(
            wbufs[0], self.layout
        )
        self.prefill_handler_tab = {
            b: kernels.BatchPrefillWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }
        self.decode_handler_tab = {
            b: kernels.BatchDecodeWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }

        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.decode_backup_stream = torch.cuda.Stream(self.device)

        # kv heads within the same group have common selection indices
        assert group_size is None or n_groups is None
        if group_size is None and n_groups is None:
            n_groups = 1
            group_size = n_kv_heads
        elif group_size is None:
            assert n_kv_heads % n_groups == 0
            group_size = n_kv_heads // n_groups
        else:
            assert n_kv_heads % group_size == 0
            n_groups = n_kv_heads // group_size
        self.group_size = group_size
        self.n_groups = n_groups
        
        assert recall_impl in ("arkvale", "torch_cpy", "cuda_cpy"), f"Unknown recall_impl: {recall_impl}"
        self.recall_impl = recall_impl
        self.recall_buf1 = torch.empty((0), **self._fp)
        self.recall_buf2 = torch.empty((0), **self._fp)
        if page_budgets[-1] is not None and cpu_layout == "HND":
            self.recall_buf1 = torch.empty((page_budgets[-1], group_size, 2, page_size, head_dim), 
                                           **self._fp)
            self.recall_buf2 = torch.empty((page_budgets[-1], group_size, 2, page_size, head_dim), 
                                           **self._fp)
        
        assert n_recall_stream % 2 == 0
        self.n_recall_stream = n_recall_stream
        self.recall_streams = [torch.cuda.Stream(self.device) 
                                for _ in range(n_recall_stream)]

        self.spec_ret = spec_ret
        self.corr_impl = corr_impl or os.getenv("CORR_IMPL", "managed_cuda")
        assert self.corr_impl in ("torch", "managed_cuda")
        self.corr_max_batch = corr_max_batch
        assert self.corr_max_batch > 0
        self.to_corr_gpu = torch.empty((self.corr_max_batch, self.n_kv_heads), **self._bool)
        self.need_corr_gpu = torch.empty((1,), **self._bool)
        self.to_corr_cpu_pinned = torch.empty(
            (self.corr_max_batch, self.n_kv_heads), pin_memory=True, **self._cpu_bool
        )
        self.need_corr_cpu_pinned = torch.empty((1,), pin_memory=True, **self._cpu_bool)
        self.to_corr_managed = None
        if self.corr_impl == "managed_cuda":
            self.to_corr_managed = kernels.alloc_managed_bool(
                self.corr_max_batch, self.n_kv_heads
            )
        if spec_ret:
            assert self.recall_impl == "cuda_cpy" and cpu_layout == "HND", f"{self.recall_impl=} {cpu_layout=}"
            self.compute_stream = torch.cuda.Stream(self.device) 
            self.spec_ret_recall_status = [None] * n_layers
            self.spec_ret_recall_events = [
                (torch.cuda.Event(), torch.cuda.Event()) for _ in range(n_layers)
            ]
            for e1, e2 in self.spec_ret_recall_events:
                # init to be a valid event...
                e1.record()
                e2.record()
            try:
                kernels.init_recall_thread_pool(thread_pool_size)
                atexit.register(self._shutdown_cpp_pool)
            except Exception as e:
                raise RuntimeError("cannot initialize thread pool") from e
            self.corr = corr
            self.last_step_q = [None] * n_layers
        else:
            self.compute_stream = self.default_stream

        self.recall_stat_ms = []
        self.sel_stat_ms = []
        self.corr_checks = [0] * n_layers
        self.corr_triggers = [0] * n_layers

        # --- Issue #1: instrumentation ---
        self.log_dir = log_dir
        self.step_id = -1  # first decode step becomes 0
        self._corr_log_fh = None
        self._recall_log_fh = None
        self._timing_log_fh = None
        self._tbt_log_fh = None
        # _pending_timings: list of [step_id, layer, component, s_idx, e_idx, ended]
        self._pending_timings: list = []
        # Reusable CUDA-event pool for timing — allocated once on first use,
        # bounded so it never grows past a few hundred events regardless of
        # how many problems / steps we run.
        self._timer_pool = None
        self._timer_free: list = []
        self._timer_capacity = 0
        self._per_head_sim_buffer = None
        self._per_head_sim_tag = None
        # per-recalled-page byte size: 2 (K+V) * page_size * group_size * head_dim * itemsize
        self._recall_bytes_per_page = (
            2 * page_size * self.group_size * head_dim * dtype.itemsize
        )

        # --- Issue #2: EMA thought-type tracker ---
        self.thought_ema_alpha = thought_ema_alpha
        self.thought_tau_r = thought_tau_r
        self.thought_tau_t = thought_tau_t
        self.sim_ema = 1.0
        self.current_thought_type = ThoughtType.R.value

        # last_step_q is also used for logging when corr/spec_ret are off,
        # so allocate it unconditionally when a log_dir is set.
        if self.log_dir is not None and not hasattr(self, "last_step_q"):
            self.last_step_q = [None] * n_layers

    def _ensure_timer_pool(self, capacity: int = 512):
        """Lazy pre-allocate a fixed pool of CUDA events for timing.
        Reusing a bounded pool avoids the millions of ad-hoc event allocations
        that otherwise occur on long-running problems and cause CUDA driver
        errors after thousands of steps."""
        if getattr(self, "_timer_pool", None) is not None:
            return
        self._timer_pool = [
            torch.cuda.Event(enable_timing=True) for _ in range(capacity)
        ]
        self._timer_free = list(range(capacity))
        self._timer_capacity = capacity

    def _acquire_event(self):
        """Pop a free event index from the pool. Allocates ad-hoc and grows
        the pool only if we ever exhaust it (degraded fallback)."""
        pool = self._timer_pool
        free = self._timer_free
        if free:
            idx = free.pop()
            return idx, pool[idx]
        # Pool exhausted — extend rather than fail. Should be rare with
        # proper sizing (~6 components × 32 layers × 2 events = 384 needed).
        idx = len(pool)
        pool.append(torch.cuda.Event(enable_timing=True))
        return idx, pool[idx]

    def _release_event(self, idx: int):
        if 0 <= idx < len(self._timer_pool):
            self._timer_free.append(idx)

    def time_block_start(self, layer_idx: int, component: str):
        """Start a CUDA-event-based timer. Returns a handle to pass to
        time_block_end(). No-op (returns None) when timing isn't enabled."""
        if self._timing_log_fh is None:
            return None
        self._ensure_timer_pool()
        s_idx, s_evt = self._acquire_event()
        e_idx, e_evt = self._acquire_event()
        s_evt.record()
        entry = [self.step_id, layer_idx, component, s_idx, e_idx, False]
        self._pending_timings.append(entry)
        return entry

    def time_block_end(self, handle):
        if handle is None:
            return
        e_idx = handle[4]
        self._timer_pool[e_idx].record()  # end event
        handle[5] = True

    def flush_step_timing(self):
        """Drain completed timings to the CSV and release events back to the
        pool. Pending (not-yet-ended) entries are kept for the next call."""
        if self._timing_log_fh is None or not self._pending_timings:
            return
        keep = []
        pool = self._timer_pool
        for entry in self._pending_timings:
            step_id, layer_id, component, s_idx, e_idx, ended = entry
            if not ended:
                keep.append(entry)
                continue
            s_evt = pool[s_idx]
            e_evt = pool[e_idx]
            try:
                ready = e_evt.query()
            except Exception:
                ready = True
            if not ready:
                keep.append(entry)
                continue
            try:
                us = s_evt.elapsed_time(e_evt) * 1000.0  # ms -> us
            except Exception:
                us = float("nan")
            self._timing_log_fh.write(
                f"{step_id},{layer_id},{component},{us:.2f}\n"
            )
            # Release both events back to the pool for reuse.
            self._release_event(s_idx)
            self._release_event(e_idx)
        self._pending_timings = keep

    def log_tbt(self, step_id: int, total_ms: float):
        if self._tbt_log_fh is None:
            return
        self._tbt_log_fh.write(f"{step_id},{total_ms:.4f}\n")

    def open_logs(self, tag: str = "run", max_steps: int | None = None):
        """Open CSV log files + per-head sim buffer. Call at start of each problem.

        If max_steps is given, also allocates a dense per-head-sim buffer of
        shape [max_steps, n_layers, n_q_heads] (filled with NaN). Each call
        to log_per_head_sim writes one layer's 1-D slice. On close_logs, the
        trimmed buffer is saved as `sims_<tag>.npz`.
        """
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self.close_logs()
        corr_path = os.path.join(self.log_dir, f"corr_{tag}.csv")
        recall_path = os.path.join(self.log_dir, f"recall_{tag}.csv")
        timing_path = os.path.join(self.log_dir, f"timing_{tag}.csv")
        tbt_path = os.path.join(self.log_dir, f"tbt_{tag}.csv")
        self._corr_log_fh = open(corr_path, "w", buffering=1)
        self._recall_log_fh = open(recall_path, "w", buffering=1)
        self._timing_log_fh = open(timing_path, "w", buffering=1)
        self._tbt_log_fh = open(tbt_path, "w", buffering=1)
        self._corr_log_fh.write("step_id,layer_id,cos_sim,need_corr,thought_type,sim_ema\n")
        # bytes_actual reflects need_recall_corr masking (zero for non-drifted heads).
        self._recall_log_fh.write(
            "step_id,layer_id,n_pages,bytes,n_pages_actual,bytes_actual\n"
        )
        self._timing_log_fh.write("step_id,layer_id,component,us\n")
        self._tbt_log_fh.write("step_id,total_ms\n")
        self._pending_timings = []

        if max_steps is not None and max_steps > 0:
            import numpy as _np
            self._per_head_sim_buffer = _np.full(
                (max_steps, self.n_layers, self.n_qo_heads),
                _np.nan, dtype=_np.float32,
            )
            self._per_head_sim_tag = tag
        else:
            self._per_head_sim_buffer = None
            self._per_head_sim_tag = None

    def close_logs(self):
        # Drain any remaining timing rows (sync if needed).
        if getattr(self, "_pending_timings", None):
            pool = getattr(self, "_timer_pool", None) or []
            for entry in self._pending_timings:
                step_id, layer_id, component, s_idx, e_idx, ended = entry
                if not ended:
                    continue
                if not pool:
                    continue
                s_evt = pool[s_idx]
                e_evt = pool[e_idx]
                try:
                    e_evt.synchronize()
                    us = s_evt.elapsed_time(e_evt) * 1000.0
                except Exception:
                    us = float("nan")
                if self._timing_log_fh is not None:
                    self._timing_log_fh.write(
                        f"{step_id},{layer_id},{component},{us:.2f}\n"
                    )
                self._release_event(s_idx)
                self._release_event(e_idx)
            self._pending_timings = []

        for attr in (
            "_corr_log_fh",
            "_recall_log_fh",
            "_timing_log_fh",
            "_tbt_log_fh",
        ):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, attr, None)

        buf = getattr(self, "_per_head_sim_buffer", None)
        tag = getattr(self, "_per_head_sim_tag", None)
        if buf is not None and tag is not None and self.log_dir is not None:
            import numpy as _np
            # Trim trailing steps we never reached (all-NaN rows).
            valid = ~_np.all(_np.isnan(buf), axis=(1, 2))
            if valid.any():
                last = int(_np.where(valid)[0].max()) + 1
                buf = buf[:last]
            else:
                buf = buf[:0]
            path = os.path.join(self.log_dir, f"sims_{tag}.npz")
            _np.savez_compressed(path, sim=buf)
            self._per_head_sim_buffer = None
            self._per_head_sim_tag = None

    def log_per_head_sim(self, layer_idx: int, sim_per_head):
        """sim_per_head: 1-D numpy array of length n_q_heads (for bsz=1)."""
        buf = getattr(self, "_per_head_sim_buffer", None)
        if buf is None:
            return
        s = self.step_id
        if 0 <= s < buf.shape[0]:
            buf[s, layer_idx] = sim_per_head

    def log_corr(self, layer_idx: int, cos_sim: float, need_corr: bool):
        if self._corr_log_fh is None:
            return
        self._corr_log_fh.write(
            f"{self.step_id},{layer_idx},{cos_sim:.6f},{int(bool(need_corr))},"
            f"{self.current_thought_type},{self.sim_ema:.6f}\n"
        )

    def log_recall(self, layer_idx: int, n_pages: int,
                   n_pages_actual: int | None = None):
        if self._recall_log_fh is None:
            return
        bytes_decided = int(n_pages) * self._recall_bytes_per_page
        if n_pages_actual is None:
            n_pages_actual = int(n_pages)
        bytes_actual = int(n_pages_actual) * self._recall_bytes_per_page
        self._recall_log_fh.write(
            f"{self.step_id},{layer_idx},{int(n_pages)},{bytes_decided},"
            f"{int(n_pages_actual)},{bytes_actual}\n"
        )

    def update_thought_type(self, cos_sim: float) -> int:
        """Update EMA of cosine sim and classify the current decode step.

        R (reasoning) = stable query direction, sim_ema >= tau_r
        T (transition) = abrupt drop, sim_ema < tau_t
        E (execution) = in between
        """
        alpha = self.thought_ema_alpha
        self.sim_ema = (1.0 - alpha) * self.sim_ema + alpha * float(cos_sim)
        if self.sim_ema >= self.thought_tau_r:
            self.current_thought_type = ThoughtType.R.value
        elif self.sim_ema < self.thought_tau_t:
            self.current_thought_type = ThoughtType.T.value
        else:
            self.current_thought_type = ThoughtType.E.value
        return self.current_thought_type

    def _shutdown_cpp_pool(self):
        try:
            kernels.shutdown_recall_thread_pool()
        except Exception as e:
            raise RuntimeError("cannot shutdown thread pool") from e


    def get_corr_trigger_stats(self):
        total_checks = sum(self.corr_checks)
        total_triggers = sum(self.corr_triggers)
        total_rate = (total_triggers / total_checks) if total_checks > 0 else 0.0
        return {
            "total_checks": total_checks,
            "total_triggers": total_triggers,
            "total_rate": total_rate,
            "layer_checks": self.corr_checks,
            "layer_triggers": self.corr_triggers,
        }

    @property
    def seq_len(self):
        return self.kv_caches[0].seq_len

    @property
    def n_pages(self):
        return self.kv_caches[0].n_pages

    @property
    def batch_size(self):
        return self.kv_caches[0].batch_size

    def _prepare_prefill(self, bsz, q_len):
        self._pool.clear()
        self.kv_caches = [
            KvCache(
                self._pool,
                bsz,
                self.layer2budget[i],
                self.n_sink_pages,
                self.n_win_pages,
                n_groups=self.n_groups,
            )
            for i in range(self.n_layers)
        ]
        self.dg_caches = [
            self.layer2budget[i] and KvCache(self._pool, bsz)
            for i in range(self.n_layers)
        ]
        self._cpu_pool.clear()
        self.cpu_kv_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]

        max_topk = max([k for k in self.layer2topk if k] + [0])
        max_cap = max([b for b in self.layer2budget if b] + [0])
        bsz1 = bsz * self.n_groups
        self.topk_dout = torch.empty([self.n_layers, bsz1 * max_cap], **self._fp)
        self.topk_iout = torch.empty(
            [self.n_layers, bsz1 * (max_topk + self.n_sink_pages + self.n_win_pages)], **self._i32
        )
        self.topk_newi = torch.empty([self.n_layers, bsz1 * max_cap], **self._fp)
        self.topk_rids = torch.empty([self.n_layers, bsz1 * (max_topk + 1)], **self._i32)

        self.topk_buff = torch.empty([self.n_layers, bsz1 * (1 << 10)], **self._u8)
        # we do not pre-allocate real kv-pages before prefill
        [kvc.prefill_alloc_n_tokens(q_len) for kvc in self.cpu_kv_caches if kvc]
        n_kv_pages = (q_len + self.page_size - 1) // self.page_size
        self.kv_last_page_len = (q_len - 1) % self.page_size + 1
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        for b in self.kv_indptrs_tab:
            self.kv_indptrs_tab[b] = kv_indptr = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )

        n_filled_kv_pages = n_kv_pages - 1
        assert n_filled_kv_pages > 0, (
            f"prefill requires q_len > page_size for digest cache construction: "
            f"q_len={q_len}, page_size={self.page_size}"
        )
        [
            dgc.prefill_alloc_n_tokens(n_filled_kv_pages, self.alloc_page)
            for dgc in self.dg_caches
            if dgc
        ]
        n_dg_pages = (n_filled_kv_pages + self.page_size - 1) // self.page_size
        self.dg_last_page_len = (n_filled_kv_pages - 1) % self.page_size + 1
        self.dg_last_page_lens = torch.tensor(
            [self.dg_last_page_len] * bsz, **self._i32
        )
        self.dg_indptrs = torch.arange(0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32)

        self.rope_indptr = torch.arange(0, (bsz+1)*q_len, q_len, **self._i32)
        self.rope_offsets = torch.zeros(bsz, **self._i32)

        qo_indptr = torch.arange(0, bsz * q_len + 1, q_len, **self._i32)
        self.prefill_handler.begin_forward(
            qo_indptr,
            kv_indptr,
            self.kv_last_page_lens,
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
            self.page_size,
            self.dtype,
        )

    def _finish_prefill(self, bsz, q_len):
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(self.kv_caches[l].n_real_pages for l in ls)
            self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )
            if b is not None and self.use_sparse_attn:
                topk = utils.all_eq(self.layer2topk[l] for l in ls)
                n_kv_pages = min(
                    self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                )
                self.kv_decode_indptrs_tab[b] = torch.arange(
                    0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                )
        self.prefill_handler.end_forward()

    def _prepare_decode(self, bsz):
        if self.kv_last_page_len + 1 >= self.page_size:
            self.default_stream.wait_stream(self.decode_backup_stream)
        pre = [kvc.n_real_pages for kvc in self.kv_caches]
        n_new_kv_pages = utils.all_eq(
            kvc.decode_alloc_1_token(self.alloc_page) for kvc in self.kv_caches
        )
        self.kv_last_page_len = utils.all_eq(
            kvc.last_page_len for kvc in self.kv_caches
        )
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        [kvc.decode_alloc_1_token() for kvc in self.cpu_kv_caches if kvc]

        if n_new_kv_pages > 0:
            assert n_new_kv_pages == 1

            cur = [kvc.n_real_pages for kvc in self.kv_caches]
            for b, ls in self.budget2layers.items():
                n_new_kv_real_pages = utils.all_eq(cur[l] - pre[l] for l in ls)
                if n_new_kv_real_pages > 0:
                    n_kv_pages = self.kv_caches[ls[0]].n_real_pages
                    self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = (
                        torch.arange(0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32)
                    )
                    if b is not None and self.use_sparse_attn:
                        topk = utils.all_eq(self.layer2topk[l] for l in ls)
                        n_kv_pages = min(
                            self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                        )
                        self.kv_decode_indptrs_tab[b] = torch.arange(
                            0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                        )

            dg_caches = [dgc for dgc in self.dg_caches if dgc]
            if len(dg_caches) > 0:
                n_new_dg_pages = utils.all_eq(
                    dgc.decode_alloc_1_token(self.alloc_page) for dgc in dg_caches
                )
                self.dg_last_page_len = utils.all_eq(
                    dgc.last_page_len for dgc in dg_caches
                )
                self.dg_last_page_lens = torch.tensor(
                    [self.dg_last_page_len] * bsz, **self._i32
                )
                if n_new_dg_pages > 0:
                    assert n_new_dg_pages == 1
                    n_dg_pages = utils.all_eq(dgc.n_real_pages for dgc in dg_caches)
                    self.dg_indptrs = torch.arange(
                        0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32
                    )

                with torch.cuda.stream(self.decode_backup_stream):
                    [self.decode_backup_1_page(l) for l in range(self.n_layers)]
                [self.decode_save_1_digest(l) for l in range(self.n_layers)]

        self.rope_indptr = torch.arange(0, bsz+1, 1, **self._i32)
        self.rope_offsets = torch.full((bsz,), self.seq_len, **self._i32)

        for b, h in self.decode_handler_tab.items():
            h.begin_forward(
                self.kv_decode_indptrs_tab[b],
                self.kv_last_page_lens,
                self.n_qo_heads,
                self.n_kv_heads,
                self.head_dim,
                self.page_size,
                data_type=self.dtype,
            )

    def _finish_decode(self, bsz):
        for handler in self.decode_handler_tab.values():
            handler.end_forward()

    def begin_forward(self, bsz, q_len):
        if q_len > 1:
            # new prompt → reset step counter and thought tracker
            self.step_id = -1
            self.sim_ema = 1.0
            self.current_thought_type = ThoughtType.R.value
            self._prepare_prefill(bsz, q_len)
        else:
            self.step_id += 1
            self._prepare_decode(bsz)

    def end_forward(self, bsz, q_len):
        if q_len > 1:
            self._finish_prefill(bsz, q_len)
        else:
            self._finish_decode(bsz)

    def append_paged_kv_cache(self, layer_idx: int, keys: Tensor, vals: Tensor):
        kvc = self.kv_caches[layer_idx]
        kernels.append_paged_kv_cache(
            keys,
            vals,
            kvc.buffer,
            kvc.c2p,
            self.kv_indptrs_tab[self.layer2budget[layer_idx]],
            self.kv_last_page_lens,
            self.layout,
        )

    def save_digests(self, layer_idx: int, digest: Digest):
        dgc = self.dg_caches[layer_idx]
        kernels.append_paged_kv_cache(
            *digest,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            self.layout,
        )

    def estimate_scores(
        self, layer_idx: int, query_states: Tensor, n_groups: int = None
    ):
        if n_groups is None:
            n_groups = self.n_groups
        dgc = self.dg_caches[layer_idx]
        return kernels.estimate_scores(
            query_states,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            dgc.seq_len,
            self.layout,
            n_groups,
        )

    def select_topk(self, layer_idx: int, scores: Tensor):
        bsz = self.batch_size * self.n_groups
        budget = self.layer2budget[layer_idx]
        topk = self.layer2topk[layer_idx]
        kvc = self.kv_caches[layer_idx]
        ns = self.n_sink_pages
        nw = self.n_win_pages
        eids_range = topk + ns + nw
        dout = self.topk_dout[layer_idx, : bsz * topk].view(bsz, topk)
        eids = self.topk_iout[layer_idx, : bsz * eids_range].view(bsz, eids_range)
        newi = self.topk_newi[layer_idx, : bsz * budget].view(bsz, budget)
        rids = self.topk_rids[layer_idx, : bsz * (topk + 1)].view(bsz, topk + 1)
        buff = self.topk_buff[layer_idx]
        scores = scores.reshape(bsz, -1)
        cc2gp = kvc.cc2gp.reshape(bsz, -1)
        gc2cc = kvc.gc2cc.reshape(bsz, -1)
        kernels.select_topk(
            scores, dout, eids, newi, cc2gp, gc2cc, rids, buff, topk, ns, nw
        )
        return eids, rids

    def recall(self, layer_idx: int, 
               eids: Tensor, rids: Tensor, 
               blocking=True,
               recall_evt1=None, 
               recall_evt2=None,
               need_recall_corr=None,):
        nw = self.n_win_pages
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        bsz = kvc.batch_size
        gs = self.group_size
        ng = self.n_groups
        eids = eids.reshape(bsz, ng, -1)
        rids = rids.reshape(bsz, ng, -1)
        impl = self.recall_impl

        if self._recall_log_fh is not None:
            page_counts = rids[:, :, 0]              # [bsz, n_groups]
            total_pages = int(page_counts.sum().item())
            # Apply need_recall_corr mask (per [bsz, n_kv_heads]) so we report
            # the actual transfer, not the unmasked top-k diff. group_size==1
            # in our standard config so kv-heads index 1:1 with groups.
            n_pages_actual = total_pages
            if (
                need_recall_corr is not None
                and hasattr(need_recall_corr, "numel")
                and need_recall_corr.numel() > 0
            ):
                try:
                    mask = need_recall_corr.to(page_counts.device).to(page_counts.dtype)
                    n_pages_actual = int((page_counts * mask).sum().item())
                except Exception:
                    n_pages_actual = total_pages
            self.log_recall(layer_idx, total_pages, n_pages_actual=n_pages_actual)

        torch.cuda.nvtx.range_push("recall")
        if impl == "arkvale":
            for i in range(kvc.batch_size):
                for j in range(self.n_groups):
                    nr = rids[i, j, 0]
                    if nr == 0:
                        continue
                    heads = slice(j * gs, (j + 1) * gs)
                    if cpu_kvc.layout == "NHD":
                        for ei, ri in zip(eids[i, j, -(nr + nw) : -nw], rids[i, j, 1 : nr + 1]):
                            kvc.pool[ei][..., heads, :].copy_(
                                cpu_kvc[i, ri][..., heads, :], non_blocking=True
                            )
                    else:
                        for ei, ri in zip(eids[i, j, -(nr + nw) : -nw], rids[i, j, 1 : nr + 1]):
                            kvc.pool[ei][..., heads, :].copy_(
                                cpu_kvc[i, ri][heads, ...].view(2, self.page_size, gs, -1), 
                                non_blocking=True
                            )
            torch.cuda.synchronize(self.device)
        else:
            if impl == "cuda_cpy" and cpu_kvc.layout == "HND":
                # NOTE: need_recall only valid here
                rs_group = layer_idx % (self.n_recall_stream // 2)
                recall_stream1 = self.recall_streams[rs_group*2]
                recall_stream2 = self.recall_streams[rs_group*2 + 1]
                if blocking:
                    kernels.recall_cuda_cpy_cpuhnd_2buf(
                        rids, eids, cpu_kvc.c2p, cpu_kvc.buffer, kvc.buffer, 
                        ng, gs, nw, self.recall_buf1, self.recall_buf2,
                        recall_stream1.cuda_stream, recall_stream2.cuda_stream,
                        need_recall_corr
                    )
                else:
                    # non-blocking double buffer
                    kernels.recall_cuda_cpy_cpuhnd_2buf_pool(
                        rids, eids, cpu_kvc.c2p, cpu_kvc.buffer, kvc.buffer, 
                        ng, gs, nw, self.recall_buf1, self.recall_buf2,
                        recall_stream1.cuda_stream,
                        recall_stream2.cuda_stream,
                        recall_evt1.cuda_event,
                        recall_evt2.cuda_event,
                        need_recall_corr
                    )
            else:
                kernels.recall(rids, eids, cpu_kvc.c2p, cpu_kvc.buffer, kvc.buffer, 
                                ng, gs, nw, impl, cpu_kvc.layout, self.recall_buf1)
        torch.cuda.nvtx.range_pop()

    def estimate_select_recall(self, layer_idx: int, q: Tensor):
        torch.cuda.nvtx.range_push("score&topk")
        scores = self.estimate_scores(layer_idx, q)
        eids, rids = self.select_topk(layer_idx, scores)
        torch.cuda.nvtx.range_pop()

        self.recall(layer_idx, eids, rids, blocking=True, need_recall_corr=torch.Tensor())

    def estimate_select(self, layer_idx: int, q: Tensor):
        torch.cuda.nvtx.range_push("score&topk")
        scores = self.estimate_scores(layer_idx, q)
        eids, rids = self.select_topk(layer_idx, scores)
        torch.cuda.nvtx.range_pop()
        return eids, rids

    def estimate_select_recall_pool(self, layer_idx: int, q: Tensor, 
                                    recall_evt1: torch.cuda.Event, 
                                    recall_evt2: torch.cuda.Event):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        dgc = self.dg_caches[layer_idx]
        bsz = self.batch_size * self.n_groups
        budget = self.layer2budget[layer_idx]
        topk = self.layer2topk[layer_idx]
        ng = self.n_groups
        gs = self.group_size
        ns = self.n_sink_pages
        nw = self.n_win_pages
        eids_range = topk + ns + nw
        dout = self.topk_dout[layer_idx, : bsz * topk].view(bsz, topk)
        eids = self.topk_iout[layer_idx, : bsz * eids_range].view(bsz, eids_range)
        newi = self.topk_newi[layer_idx, : bsz * budget].view(bsz, budget)
        rids = self.topk_rids[layer_idx, : bsz * (topk + 1)].view(bsz, topk + 1)
        buff = self.topk_buff[layer_idx]
        cc2gp = kvc.cc2gp.reshape(bsz, -1)
        gc2cc = kvc.gc2cc.reshape(bsz, -1)

        rs_group = layer_idx % (self.n_recall_stream // 2)
        recall_stream1 = self.recall_streams[rs_group*2]
        recall_stream2 = self.recall_streams[rs_group*2 + 1]
        kernels.estimate_select_recall_pool(
            # for estimate
            q, dgc.buffer, dgc.c2p, dgc.seq_len,
            self.dg_indptrs, self.dg_last_page_lens,
            # for topk
            dout, eids, rids, newi, buff,
            cc2gp, gc2cc, topk,
            # for recall
            cpu_kvc.c2p, cpu_kvc.buffer,
            kvc.buffer,
            ng, gs, ns, nw, 
            self.recall_buf1, self.recall_buf2,
            recall_stream1.cuda_stream, recall_stream2.cuda_stream,
            recall_evt1.cuda_event, recall_evt2.cuda_event
        )

    def prefill_backup_pages(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for ci, gi in zip(cpu_kvc.c2p.reshape(-1), kvc.c2p.reshape(-1)):
            if cpu_kvc.layout == "HND":
                # [2, page_size, n_kv_heads, head_dim] => [n_kv_heads, 2, page_size, head_dim]
                cpu_kvc.buffer[ci].copy_(kvc.buffer[gi].permute(2, 0, 1, 3).contiguous(),
                                         non_blocking=True)
            else:
                cpu_kvc.buffer[ci].copy_(kvc.buffer[gi], non_blocking=True)

    def decode_backup_1_page(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for i in range(kvc.batch_size):
            if cpu_kvc.layout == "HND":
                cpu_kvc[i, -2].copy_(kvc[i, kvc.evict_idx].permute(2, 0, 1, 3).contiguous(), 
                                     non_blocking=True)
            else:
                cpu_kvc[i, -2].copy_(kvc[i, kvc.evict_idx], non_blocking=True)

    def _summarize_keys(self, filled_keys: Tensor) -> Digest:
        maxs = filled_keys.max(dim=2).values
        mins = filled_keys.min(dim=2).values
        centers = (maxs + mins) / 2
        dists = (
            (
                centers.reshape(*filled_keys.shape[:2], 1, -1, self.head_dim)
                - filled_keys
            )
            .abs()
            .mean(dim=2)
        )
        mins = centers - dists
        maxs = centers + dists
        return maxs, mins

    def prefill_save_digests(self, layer_idx: int, keys: Tensor):
        if self.layer2budget[layer_idx] is None:
            return
        bsz, q_len, *_ = keys.shape
        n_filled_pages = (q_len + self.page_size - 1) // self.page_size - 1
        filled_keys = keys[:, : n_filled_pages * self.page_size, ...].reshape(
            bsz, n_filled_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def decode_save_1_digest(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        if kvc.budget is None:
            return
        filled_keys = torch.cat(
            [kvc[i, kvc.evict_idx][:1][None, ...] for i in range(self.batch_size)],
            dim=0,
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def prefill_evict_extra_pages(self, layer_idx: int, query_states: Tensor):
        kvc = self.kv_caches[layer_idx]
        bsz = kvc.batch_size
        ng = self.n_groups
        if kvc.budget is None:
            return
        if kvc.n_real_pages > kvc.budget:
            ns = max(2, kvc.n_sink_pages)
            nw = kvc.n_win_pages
            assert ns + nw <= kvc.budget
            topk = kvc.budget - ns - nw
            kvc.c2p = torch.empty([bsz, kvc.budget], **kvc._i32)
            kvc.gc2cc = torch.empty([bsz, kvc.budget], **kvc._i32)
            ev_gpi = kvc.cc2gp.clone()
            kvc.cc2gp.fill_(-1)
            dout = self.topk_dout[layer_idx, : bsz * topk].view(bsz, topk)
            buff = self.topk_buff[layer_idx]
            scores = self.estimate_scores(layer_idx, query_states, n_groups=1)
            scores = scores.reshape(bsz, -1)
            kernels.prefill_select_topk(
                scores, dout, kvc.c2p, kvc.cc2gp, ev_gpi, kvc.gc2cc, buff, topk, ns, nw
            )
            self.prefill_evicted_pages[layer_idx] = ev_gpi
        kvc.cc2gp = (
            kvc.cc2gp.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.cc2gp.shape[-1])
            .contiguous()
        )
        kvc.gc2cc = (
            kvc.gc2cc.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.gc2cc.shape[-1])
            .contiguous()
        )

    def alloc_page(self):
        if len(self._pool._free_ids) <= 0:
            for i in range(self.n_layers):
                kvc = self.kv_caches[i]
                evt = self.prefill_backup_events[i]
                ev_gpi = self.prefill_evicted_pages[i]
                if ev_gpi is None:
                    continue
                assert evt is not None
                self.default_stream.wait_event(evt)
                [
                    kvc.pool.free_page(pid)
                    for pid in ev_gpi.reshape(-1).tolist()
                    if pid >= 0
                ]
                self.prefill_backup_events[i] = None
                self.prefill_evicted_pages[i] = None
                break
        return self._pool.alloc_page()

    def prefill_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.prefill_handler.forward(q, kvc.buffer, page_ids)

    def decode_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.decode_handler_tab[kvc.budget].forward(q, kvc.buffer, page_ids)
