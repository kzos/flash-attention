# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import math
from typing import Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32
from cutlass.cute import FastDivmodDivisor

from cutlass.utils import ClcDynamicPersistentTileScheduler
from flash_attn.cute.tile_scheduler import (
    SchedulerState,
    compute_sm100_fmha_grid as compute_grid,
    compute_sm100_fmha_grid_clc as compute_grid_clc,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
)
from flash_attn.cute.mask import (
    Sm100FusedMask as FusedMask,
)
from flash_attn.cute.tile_scheduler import SM100_TMEM_CAPACITY_COLUMNS
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors, _TUNING_CONFIG
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.utils import ex2_emulation_2, as_bshkrd_tensor, AuxData
import flash_attn.cute.utils as fa_utils
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.pack_gqa import pack_gqa_layout


class BlackwellHd512FusedMultiHeadAttentionForward:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        is_static_persistent: bool = True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        has_tile_count_semaphore: bool = False,
        seqlen_k_per_split: Optional[int] = None,
        max_seqlen_q: int = 4,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 512 and head_dim_v == 512, (
            "SM103 dedicated decode kernel only supports (head_dim, head_dim_v) = (512, 512)"
        )
        assert score_mod is None, "SM103 forward with head_dim=512 does not support score_mod"
        assert mask_mod is None, "SM103 forward with head_dim=512 does not support mask_mod"
        assert not has_aux_tensors, "SM103 forward with head_dim=512 does not support aux tensors"
        self.use_tma_KV = not paged_kv_non_tma
        self.pack_gqa = pack_gqa
        assert not is_split_kv, "SM103 forward with head_dim=512 does not support SplitKV"
        assert q_subtile_factor == 1, (
            "SM103 forward with head_dim=512 does not support q_subtile_factor"
        )
        assert kv_subtile_factor == 1, (
            "SM103 forward with head_dim=512 does not support kv_subtile_factor"
        )
        assert m_block_size == 64 and n_block_size == 128, (
            "SM103 hd512 dedicated kernel requires tile_m=64 and tile_n=128"
        )
        # q_stage / persistence / scheduler knobs are accepted for interface parity,
        # but this dedicated kernel uses fixed internal settings.

        qk_acc_dtype = cutlass.Float32
        pv_acc_dtype = cutlass.Float32
        # The hd256 ancestor uses 128 query rows per CTA (256 per 2CTA
        # cluster).  That geometry cannot hold two dV=256 FP32 fragments in
        # the 512-column TMEM allocation.  Match the working SM100 MLA
        # dV=512 geometry instead: 64 rows per CTA, 128 per cluster.
        mma_tiler = (64, 128, head_dim)
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.qhead_per_kvhead = qhead_per_kvhead
        self.mma_tiler = mma_tiler
        assert mma_tiler[0] == 64 and mma_tiler[1] == 128, (
            "Only the SM103 hd512 CTA tile 64x128 is supported"
        )
        assert mma_tiler[2] == 512, "Only 512 is supported for the SM103 decode tile"
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128),
        )
        # Four QK reductions cover dQK=512.  Two independent dV=256 PV
        # fragments cover dV=512.
        self.pv_mma_tiler = (
            self.qk_mma_tiler[0],
            head_dim_v // 2,
            self.qk_mma_tiler[1],
        )
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // 2,
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)
        self.tmem_warp_shape_mn = (4, 1)
        # Dedicated hd256 kernel uses fixed scheduling policy.
        self.is_persistent = False
        self.is_causal = is_causal
        self.is_local = is_local
        self.use_semantic_trip_range = is_causal or is_local
        self.use_clc_scheduler = False
        self.max_seqlen_q = max_seqlen_q

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.relay_warp_id = 10
        self.cpasync_load_warp_id = 11
        # Page128 uses warp 9 for Q/K/V descriptor TMA. Page64 uses warp 9 for
        # Q plus CTA-local K/V descriptor TMA, warp 10 as the local-to-cluster
        # relay, and leaves warp 11 inactive.
        self.empty_warp_id = (10, 11) if self.use_tma_KV else (11,)
        self.sched_warp_id = self.empty_warp_id[0] if use_clc_scheduler else None
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        # Keep the fixed 12-warp launch geometry. On page64 warp 9 is the
        # descriptor-TMA producer and warp 10 is the TMA-to-UMMA relay; on
        # page128 warps 10/11 remain inactive.
        self.threads_per_cta = self.threads_per_warp * 12

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.softmax_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=len(self.softmax_warp_ids) * self.threads_per_warp,
        )

        # TMEM layout for the 2CTA cluster:
        #   S0:  [  0,  64), S1: [ 64, 128)
        #   O0:  [128, 256), O1: [256, 384)
        # A `(cluster_m=128, n=128)` score fragment consumes 64 columns per
        # CTA and a `(cluster_m=128, dV=256)` output fragment consumes 128.
        self.tmem_s_offset = 0
        self.tmem_s_cols_per_stage = self.qk_mma_tiler[1] // self.cluster_shape_mn[0]
        self.tmem_s_stage_count = 2
        self.tmem_o_offset = self.tmem_s_cols_per_stage * self.tmem_s_stage_count
        self.tmem_o_cols_per_split = self.pv_mma_tiler[1] // self.cluster_shape_mn[0]
        self.tmem_o_split_stride = self.tmem_o_cols_per_split
        self.tmem_total_cols = (
            self.tmem_o_offset + self.iterations_pv * self.tmem_o_cols_per_split
        )
        assert self.tmem_total_cols <= self.tmem_alloc_cols, (
            f"SM103 hd512 TMEM layout requires {self.tmem_total_cols} columns, "
            f"capacity is {self.tmem_alloc_cols}"
        )
        _tune_key = (True, is_causal, 512, True)
        _tune = _TUNING_CONFIG.get(_tune_key, {})
        self.num_regs_softmax = _tune.get("num_regs_softmax", 256)
        self.num_regs_correction = _tune.get("num_regs_correction", 160)
        self.num_regs_other = 32
        self.num_regs_cpasync = 32 if self.use_tma_KV else 80
        self.num_cpasync_load_threads = (
            self.threads_per_warp if self.use_tma_KV else 2 * self.threads_per_warp
        )
        self.ex2_emu_freq = _tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = _tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = _tune.get("ex2_emu_start_frg", 0)

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        # Two KV stages keep the d512 specialization within SM103's dynamic
        # shared-memory limit while avoiding immediate reuse of the sole K/V
        # slot in the two-CTA producer-consumer sequence.
        self.kv_stage = 2
        self.qk_acc_stage = 2
        assert self.qk_acc_stage == self.tmem_s_stage_count
        self.mma_corr_stage = 1
        if cutlass.const_expr(self.use_clc_scheduler):
            self.num_clc_stage = 1
            self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        stream: cuda.CUstream = None,
    ):
        # Keep parity with FlashAttentionForwardSm100.__call__ interface.
        # (TODO@wangsiyu) Implement these features.
        assert learnable_sink is None, (
            "SM100 forward with head_dim=256 does not support learnable_sink"
        )
        assert blocksparse_tensors is None, (
            "SM100 forward with head_dim=256 does not support block sparsity"
        )
        assert aux_data.tensors is None, (
            "SM100 forward with head_dim=256 does not support aux_tensors"
        )
        assert aux_data.scalars is None, (
            "SM100 forward with head_dim=256 does not support aux_scalars"
        )
        assert not self.is_local, (
            "SM100 forward with head_dim=256 does not support local attention yet"
        )
        assert window_size_left is None and window_size_right is None, (
            "SM100 forward with head_dim=256 does not support runtime window_size overrides"
        )
        assert descale_tensors is None, (
            "SM100 forward with head_dim=256 does not support descale_tensors"
        )

        q_tensor, k_tensor, v_tensor, o_tensor = mQ, mK, mV, mO
        lse_tensor = mLSE
        cum_seqlen_q = mCuSeqlensQ
        cum_seqlen_k = mCuSeqlensK

        q_rank = len(mQ.shape)
        k_rank = len(mK.shape)
        if cutlass.const_expr(cum_seqlen_q is not None):
            # Varlen path accepts either legacy 5D tensors or standard 3D tensors.
            if cutlass.const_expr(q_rank == 5):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2] * mQ.shape[3]
                d = mQ.shape[4]
            elif cutlass.const_expr(q_rank == 3):
                s_q = mQ.shape[0]
                h_q = mQ.shape[1]
                d = mQ.shape[2]
            else:
                raise RuntimeError(f"hd256 forward varlen expects q rank 3 or 5, got rank {q_rank}")
        else:
            # Non-varlen path accepts either legacy 5D tensors or standard 4D tensors.
            if cutlass.const_expr(q_rank == 5):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2] * mQ.shape[3]
                d = mQ.shape[4]
            elif cutlass.const_expr(q_rank == 4):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2]
                d = mQ.shape[3]
            else:
                raise RuntimeError(
                    f"hd256 forward non-varlen expects q rank 4 or 5, got rank {q_rank}"
                )

        if cutlass.const_expr(cum_seqlen_k is not None):
            if cutlass.const_expr(k_rank == 5):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            elif cutlass.const_expr(k_rank == 3):
                s_k = mK.shape[0]
                h_k = mK.shape[1]
            else:
                raise RuntimeError(f"hd256 forward varlen expects k rank 3 or 5, got rank {k_rank}")
        else:
            if cutlass.const_expr(k_rank == 5):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            elif cutlass.const_expr(k_rank == 4):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            else:
                raise RuntimeError(
                    f"hd256 forward non-varlen expects k rank 4 or 5, got rank {k_rank}"
                )
        if cutlass.const_expr(cum_seqlen_q is not None):
            b = mCuSeqlensQ.shape[0] - 1
        elif cutlass.const_expr(cum_seqlen_k is not None):
            b = mCuSeqlensK.shape[0] - 1
        else:
            b = mQ.shape[0]

        scale_softmax = softmax_scale
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        scale_output = 1.0
        s_lse = s_q
        h_r = h_q // h_k
        s_q64 = Int64(s_q)
        s_k64 = Int64(s_k)
        s_lse64 = Int64(s_lse)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)
        s_q_total = (
            q_tensor.shape[1]
            if cum_seqlen_q is not None and q_rank == 5
            else (q_tensor.shape[0] if cum_seqlen_q is not None else s_q64)
        )
        s_k_total = (
            k_tensor.shape[1]
            if cum_seqlen_k is not None and k_rank == 5
            else (k_tensor.shape[0] if cum_seqlen_k is not None else s_k64)
        )
        b_lse = b64 if cum_seqlen_q is None else 1
        stride_b_lse = h_r64 * h_k64 * s_lse64 if cum_seqlen_q is None else 0

        varlen_q = cum_seqlen_q is not None
        varlen_k = cum_seqlen_k is not None
        q_norm = as_bshkrd_tensor(q_tensor, h_k, h_r, varlen_q)
        o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, varlen_q)

        # Forward layout: (s, d, ((h_r, h_k), b)). Stride picks from canonical
        # positions 1=S, 4=D, 3=H_r, 2=H_k, 0=B.
        q = cute.make_tensor(
            q_norm.iterator,
            cute.make_layout(
                (s_q_total, d, h_q, b),
                stride=(
                    q_norm.stride[1],
                    q_norm.stride[4],
                    q_norm.stride[3],
                    q_norm.stride[0],
                ),
            ),
        )
        if cutlass.const_expr(mPageTable is not None):
            # Paged: input k/v are rank-4 (num_pages, page_size, h_k, d); the kernel
            # consumes K as (page_size, d, h_k, num_pages) and V as
            # (d, page_size, h_k, num_pages).
            # cute.select reorders modes while preserving input strides
            page_size = k_tensor.shape[1]
            max_seqlen_k_paged = Int32(mPageTable.shape[1] * page_size)
            k = cute.make_tensor(k_tensor.iterator, cute.select(k_tensor.layout, mode=[1, 3, 2, 0]))
            v = cute.make_tensor(v_tensor.iterator, cute.select(v_tensor.layout, mode=[3, 1, 2, 0]))
            page_table = cute.make_tensor(
                mPageTable.iterator,
                cute.make_layout(
                    (b, mPageTable.shape[1]),
                    stride=(mPageTable.stride[0], mPageTable.stride[1]),
                ),
            )
        else:
            # K/V have no h_r dim; pass h_r=1 to the normalizer and override the
            # h_r stride to 0 below to broadcast across the query-grouped heads.
            k_norm = as_bshkrd_tensor(k_tensor, h_k, 1, varlen_k)
            v_norm = as_bshkrd_tensor(v_tensor, h_k, 1, varlen_k)
            # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
            k = cute.make_tensor(
                k_norm.iterator,
                cute.make_layout(
                    (s_k_total, d, ((h_r, h_k), b)),
                    stride=(
                        k_norm.stride[1],
                        k_norm.stride[4],
                        ((0, k_norm.stride[2]), k_norm.stride[0]),
                    ),
                ),
            )
            # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
            v = cute.make_tensor(
                v_norm.iterator,
                cute.make_layout(
                    (d, s_k_total, ((h_r, h_k), b)),
                    stride=(
                        v_norm.stride[4],
                        v_norm.stride[1],
                        ((0, v_norm.stride[2]), v_norm.stride[0]),
                    ),
                ),
            )
            page_table = None
            max_seqlen_k_paged = None
        # (s, d, ((h_r, h_k), b))
        o = cute.make_tensor(
            o_norm.iterator,
            cute.make_layout(
                (s_q_total, d, h_q, b),
                stride=(
                    o_norm.stride[1],
                    o_norm.stride[4],
                    o_norm.stride[3],
                    o_norm.stride[0],
                ),
            ),
        )
        if cutlass.const_expr(lse_tensor is not None):
            # (s, h, b)
            lse_layout = cute.make_layout(
                (s_lse64, h_q, b_lse),
                stride=(1, s_lse64, stride_b_lse),
            )
            lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
        else:
            lse = None

        if cutlass.const_expr(self.pack_gqa):
            q = pack_gqa_layout(q, self.qhead_per_kvhead, h_k, head_idx=2)
            o = pack_gqa_layout(o, self.qhead_per_kvhead, h_k, head_idx=2)
            if cutlass.const_expr(lse is not None):
                lse = pack_gqa_layout(lse, self.qhead_per_kvhead, h_k, head_idx=1)

        # The dedicated scheduler carries (head, batch) as one hierarchical
        # mode. Re-nest the generic PackGQA view without changing its strides.
        if cutlass.const_expr(self.pack_gqa):
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[1], (q.shape[2], q.shape[3])),
                    stride=(q.stride[0], q.stride[1], (q.stride[2], q.stride[3])),
                ),
            )
            o = cute.make_tensor(
                o.iterator,
                cute.make_layout(
                    (o.shape[0], o.shape[1], (o.shape[2], o.shape[3])),
                    stride=(o.stride[0], o.stride[1], (o.stride[2], o.stride[3])),
                ),
            )
            if cutlass.const_expr(lse is not None):
                lse = cute.make_tensor(
                    lse.iterator,
                    cute.make_layout(
                        (lse.shape[0], (lse.shape[1], lse.shape[2])),
                        stride=(lse.stride[0], (lse.stride[1], lse.stride[2])),
                    ),
                )
        else:
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[1], ((h_r, h_k), b)),
                    stride=(q.stride[0], q.stride[1], ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0])),
                ),
            )
            o = cute.make_tensor(
                o.iterator,
                cute.make_layout(
                    (o.shape[0], o.shape[1], ((h_r, h_k), b)),
                    stride=(o.stride[0], o.stride[1], ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0])),
                ),
            )
            if cutlass.const_expr(lse is not None):
                lse = cute.make_tensor(
                    lse.iterator,
                    cute.make_layout(
                        (lse.shape[0], ((h_r, h_k), b_lse)),
                        stride=(lse.stride[0], ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
                    ),
                )

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        # Packed Q stores total_q in the tensor's leading extent, while the
        # scheduler already carries batch as a separate dimension.  The public
        # route admits at most four Q tokens per sequence, so one M tile per
        # batch is both sufficient and avoids multiplying the grid by total_q.
        grid_s_q = self.max_seqlen_q if cum_seqlen_q is not None else s_q
        if cutlass.const_expr(self.pack_gqa and cum_seqlen_q is not None):
            grid_s_q *= self.qhead_per_kvhead
        if cutlass.const_expr(self.use_clc_scheduler):
            self.tile_sched_params, grid = compute_grid_clc(
                (grid_s_q, o.shape[1], o.shape[2]) if cum_seqlen_q is not None else o.shape,
                self.cta_tiler,
                (*self.cluster_shape_mn, 1),
            )
        else:
            self.tile_sched_params, grid = compute_grid(
                (grid_s_q, o.shape[1], o.shape[2]) if cum_seqlen_q is not None else o.shape,
                self.cta_tiler,
                self.is_persistent,
            )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # CTA-M64 has two softmax threads per row. Stage P in shared memory,
        # following the working SM100 MLA ownership model, so both thread
        # partitions can populate the same logical row before PV consumes it.
        p_source = tcgen05.OperandSource.SMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Page128 uses the existing 2CTA MMA-aware descriptor. Page64 uses
        # CTA-local generic descriptors: K transfers one 64xd128 page tile,
        # while V transfers two 64-token halves into one d128x128 local stage.
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = None, k
        tma_atom_v, tma_tensor_v = None, v
        if cutlass.const_expr(self.use_tma_KV):
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                v,
                v_smem_layout,
                self.pv_mma_tiler,
                pv_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        else:
            tma_load_op_cta1 = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
                tcgen05.CtaGroup.ONE
            )
            k_tma_smem_layout = cute.composition(
                k_smem_layout,
                cute.make_ordered_layout((64, self.qk_mma_tiler[2]), order=(0, 1)),
            )
            # Reproduce the proven transposed-V shared view for one 64-token
            # page, then expose it in source order (d128, token64).
            v_tma_smem_layout_td = cute.composition(
                v_smem_layout,
                cute.make_ordered_layout((64, self.qk_mma_tiler[2]), order=(1, 0)),
            )
            v_tma_smem_layout = cute.select(v_tma_smem_layout_td, mode=[1, 0])
            tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op_cta1,
                k,
                k_tma_smem_layout,
                (64, self.qk_mma_tiler[2]),
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op_cta1,
                v,
                v_tma_smem_layout,
                (self.qk_mma_tiler[2], 64),
            )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kv_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_v_bytes = v_copy_size * cute.size(pv_tiled_mma.thr_id.shape)
        assert self.tma_copy_v_bytes == 2 * self.tma_copy_kv_bytes
        self.tma_copy_v_extra_bytes = self.tma_copy_v_bytes - self.tma_copy_kv_bytes
        # CTA-local page64 transactions use the per-CTA sizes, without the
        # 2CTA multiplier used by PipelineTmaUmma above.
        self.tma_copy_local_k_bytes = k_copy_size
        self.tma_copy_local_v_extra_bytes = v_copy_size - k_copy_size
        assert self.tma_copy_local_k_bytes == 64 * 128 * self.k_dtype.width // 8
        assert self.tma_copy_local_v_extra_bytes == self.tma_copy_local_k_bytes

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[
                Int64, self.q_stage * 2
            ]  # load_q_{producer,consumer}
            load_kv_mbar_ptr: cute.struct.MemRange[
                Int64, self.kv_stage * 2
            ]  # load_kv_{producer,consumer}
            # Local CTA descriptor-TMA completion -> relay warp. Page128 keeps a
            # zero-sized field so its shared-memory footprint is unchanged.
            load_kv_cpasync_mbar_ptr: cute.struct.MemRange[
                Int64, 0 if self.use_tma_KV else self.kv_stage * 2
            ]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            # Softmax -> Correction signaling barriers (shared rescale ready)
            s_corr_mbar_ptr: cute.struct.MemRange[
                Int64, self.qk_acc_stage * 2
            ]  # s_corr_{producer,consumer}
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            # MMA -> Correction ownership barriers for O_partial tokens (online rescale/finalize)
            mma_corr_mbar_ptr: cute.struct.MemRange[
                Int64, self.mma_corr_stage * 2
            ]  # mma_corr_{producer,consumer}
            # A CTA-wide "TMEM lifetime" barrier used to safely deallocate TMEM after all users finish.
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            cum_seqlen_q,
            cum_seqlen_k,
            mSeqUsedQ,
            mSeqUsedK,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            page_table,
            max_seqlen_k_paged,
            window_size_left,
            window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_smem_layout_staged,
            v_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: Optional[cute.CopyAtom],
        mK_kdl: cute.Tensor,
        tma_atom_v: Optional[cute.CopyAtom],
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        mPageTable: Optional[cute.Tensor],
        max_seqlen_k: Optional[Int32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams | FmhaClcDynamicTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline_custom.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        if cutlass.const_expr(self.use_tma_KV):
            load_kv_pipeline = pipeline_custom.PipelineTmaUmma.create(
                num_stages=self.kv_stage,
                producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                tx_count=self.tma_copy_kv_bytes,
                barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            load_kv_cpasync_pipeline = None
        else:
            load_kv_pipeline = pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                # One elected relay lane from each CTA arrives at the
                # cluster-visible UMMA barrier.
                producer_group=make_thread_cooperative_group(self.cluster_shape_mn[0]),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            load_kv_cpasync_pipeline = pipeline_custom.PipelineTmaAsync.create(
                num_stages=self.kv_stage,
                producer_group=make_thread_cooperative_group(1),
                # PipelineTmaAsync elects one signalling lane per unit CTA for
                # the empty-barrier release, so the arrival count must be one.
                consumer_group=make_thread_cooperative_group(1),
                tx_count=self.tma_copy_local_k_bytes,
                barrier_storage=storage.load_kv_cpasync_mbar_ptr.data_ptr(),
                defer_sync=True,
            )
        load_kv_producer, load_kv_consumer = load_kv_pipeline.make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
        # Initialize CLC state if using dynamic scheduler
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(self.cluster_shape_mnk)
            num_clc_consumer_threads = self.threads_per_warp * (
                1  # sched_warp (CTA 0 only)
                + cluster_size
                * (
                    len(self.softmax_warp_ids)
                    + len(self.correction_warp_ids)
                    + 1  # mma_warp
                    + 1  # load_warp
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc = SchedulerState.create_clc(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_sched_params.clc_hw_params(),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                    num_stages=self.num_clc_stage,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=self.num_clc_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_clc_stage
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_clc_stage
                ),
            )
        else:
            clc = None
            clc_response_ptr = None

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        # K and V now use separate memory since we removed the transform stage
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sP = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=p_smem_layout_staged.outer,
            swizzle=p_smem_layout_staged.inner,
            byte_alignment=128,
        )

        # CTA-M64 gives two softmax/correction threads per row.  Keep the two
        # partial row reductions in shared memory and combine them before
        # normalization, following the SM100 MLA CTA-M64 implementation.
        sRowMax = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout((self.cta_tiler[0], self.cluster_shape_mn[0])),
            byte_alignment=128,
        )
        sScale = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout((self.cta_tiler[0], self.qk_acc_stage)),
            byte_alignment=128,
        )
        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout((self.cta_tiler[0], self.cluster_shape_mn[0])),
            byte_alignment=128,
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrP = pv_thr_mma.make_fragment_A(sP)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.tmem_o_split_stride,
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        if cutlass.const_expr(self.use_clc_scheduler):
            tile_sched = FmhaClcDynamicTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
                clc,
            )
        else:
            blk_idx = cute.arch.block_idx()
            tile_sched = FmhaStaticTileScheduler(
                tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
            )
        work_tile = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        is_load_warp = warp_idx == self.load_warp_id
        if is_load_warp:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_cpasync)
            local_tma_producer_state = None
            if cutlass.const_expr(not self.use_tma_KV):
                local_tma_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.kv_stage
                )
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx  # (q_tile_idx, 0, (head_idx, batch_idx))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                batch_coord = curr_block_coord[2][1]
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                block_offset = (
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(self.pack_gqa):
                        seqlen_q *= self.qhead_per_kvhead
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if cutlass.const_expr(mSeqUsedQ is not None):
                    seqlen_q = mSeqUsedQ[batch_coord]
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if cutlass.const_expr(mSeqUsedK is not None):
                    seqlen_k = mSeqUsedK[batch_coord]
                if not continue_cond:
                    if cutlass.const_expr(self.pack_gqa):
                        mQ_qdl_ = (
                            cute.domain_offset(
                                ((Int32(0), cuseqlen_q), Int32(0), (Int32(0), Int32(0))),
                                mQ_qdl,
                            )
                            if cutlass.const_expr(cum_seqlen_q is not None)
                            else mQ_qdl
                        )
                    else:
                        mQ_qdl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl
                        )
                    # Local tile partition global tensors
                    q_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    )
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    kv_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    )
                    if cutlass.const_expr(mPageTable is None):
                        assert self.use_tma_KV
                        assert tma_atom_k is not None and tma_atom_v is not None
                        # Dense path: domain_offset K/V by batch block, select batch via mma_block_coord[2].
                        mK_kdl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[1, 2, 3]), mK_kdl
                        )
                        mV_dkl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[2, 1, 3]), mV_dkl
                        )
                        gK_kdl = cute.flat_divide(
                            mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                        )
                        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_k,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK_kdl, 0, 3),
                        )
                        gV_dkl = cute.flat_divide(
                            mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                        )
                        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_v,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tSgV_dkl, 0, 3),
                        )
                        # ((atom_v, rest_v), RestN, RestK)
                        tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                        tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                        paged_kv_manager = None
                    else:
                        # Paged path: page128 uses TMA, while page64 uses the
                        # generic d-offset-aware cp.async page manager.
                        head_kv_coord = (
                            curr_block_coord[2][0]
                            if cutlass.const_expr(self.pack_gqa)
                            else curr_block_coord[2][0] // self.qhead_per_kvhead
                        )
                        if cutlass.const_expr(self.use_tma_KV):
                            assert tma_atom_k is not None and tma_atom_v is not None
                            # Keep num_pages for page-index-based TMA.
                            mK_kdl_ = mK_kdl[None, None, head_kv_coord, None]
                            mV_dkl_ = mV_dkl[None, None, head_kv_coord, None]
                            gK_kdl = cute.flat_divide(
                                mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                            )
                            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_k,
                                block_in_cluster_coord_vmnk[1],
                                kv_cta_layout,
                                cute.group_modes(sK, 0, 3),
                                cute.group_modes(tSgK_kdl, 0, 3),
                            )
                            gV_dkl = cute.flat_divide(
                                mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                            )
                            tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                            tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_v,
                                block_in_cluster_coord_vmnk[1],
                                kv_cta_layout,
                                cute.group_modes(sV, 0, 3),
                                cute.group_modes(tSgV_dkl, 0, 3),
                            )
                            tKgK = tKgK_kdl
                            tVgV = tVgV_dkl
                            paged_kv_manager = None
                        else:
                            # CTA-local descriptor tensors retain their page
                            # mode. Dynamic page-table values become TMA source
                            # coordinates in the load helpers below.
                            assert tma_atom_k is not None and tma_atom_v is not None
                            mK_page64 = mK_kdl[None, None, head_kv_coord, None]
                            mV_page64 = mV_dkl[None, None, head_kv_coord, None]
                            paged_kv_manager = None
                            tKsK, tKgK = None, None
                            tVsV, tVgV = None, None
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        self.get_trip_start_count(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )
                    )
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps
                    # Q
                    if warp_idx == self.load_warp_id:
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            q_handle = load_q_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_q,
                                tQgQ[None, iter],
                                tQsQ[None, q_handle.index],
                                tma_bar_ptr=q_handle.barrier,
                            )

                    # K0
                    kv_coord = seqlen_kv_loop_start
                    k_page_idx = (
                        mPageTable[batch_coord, kv_coord]
                        if cutlass.const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        k_handle = load_kv_producer.acquire_and_advance()
                        if cutlass.const_expr(self.use_tma_KV):
                            assert tma_atom_k is not None and tKgK is not None and tKsK is not None
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, kv_coord, iter]
                                if cutlass.const_expr(mPageTable is None)
                                else tKgK[None, 0, iter, k_page_idx],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                        else:
                            assert local_tma_producer_state is not None
                            load_kv_cpasync_pipeline.producer_acquire(
                                local_tma_producer_state,
                                try_acquire_token=cutlass.Boolean(1),
                            )
                            self.tma_page64_load_K(
                                tma_atom_k,
                                mK_page64,
                                mPageTable,
                                batch_coord,
                                load_kv_cpasync_pipeline,
                                sK,
                                cta_rank_in_cluster,
                                kv_coord,
                                local_tma_producer_state.index,
                                iter,
                                load_kv_cpasync_pipeline.producer_get_barrier(
                                    local_tma_producer_state
                                ),
                            )
                            local_tma_producer_state.advance()
                    kv_coord += 1
                    # v_page_idx_prev carries K[i-1]'s page index for use as V[i-1]'s page
                    # (K and V for the same KV block share the same physical page).
                    # Also serves as the Vend page index when seqlen_kv_loop_steps == 1.
                    v_page_idx_prev = (
                        k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                    )
                    # Prefetch K1 page after K0 TMA dispatch to hide L2 latency.
                    if cutlass.const_expr(mPageTable is not None and self.use_tma_KV):
                        if seqlen_kv_loop_steps > 1:
                            k_page_idx = mPageTable[batch_coord, kv_coord]

                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki: k_page_idx was prefetched at end of previous iteration
                        # (or in the prologue for i==1); L2 latency already hidden.
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_kv_producer.acquire_and_advance()
                            if cutlass.const_expr(self.use_tma_KV):
                                assert tma_atom_k is not None and tKgK is not None and tKsK is not None
                                cute.copy(
                                    tma_atom_k,
                                    tKgK[None, kv_coord, iter]
                                    if cutlass.const_expr(mPageTable is None)
                                    else tKgK[None, 0, iter, k_page_idx],
                                    tKsK[None, k_handle.index],
                                    tma_bar_ptr=k_handle.barrier,
                                )
                            else:
                                assert local_tma_producer_state is not None
                                load_kv_cpasync_pipeline.producer_acquire(
                                    local_tma_producer_state,
                                    try_acquire_token=cutlass.Boolean(1),
                                )
                                self.tma_page64_load_K(
                                    tma_atom_k,
                                    mK_page64,
                                    mPageTable,
                                    batch_coord,
                                    load_kv_cpasync_pipeline,
                                    sK,
                                    cta_rank_in_cluster,
                                    kv_coord,
                                    local_tma_producer_state.index,
                                    iter,
                                    load_kv_cpasync_pipeline.producer_get_barrier(
                                        local_tma_producer_state
                                    ),
                                )
                                local_tma_producer_state.advance()
                        # Vi-1: reuse v_page_idx_prev (= K[i-1]'s page), no extra GMEM read.
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            if cutlass.const_expr(self.use_tma_KV):
                                v_handle = load_kv_producer.acquire_and_advance(
                                    extra_tx_count=self.tma_copy_v_extra_bytes
                                )
                                assert tma_atom_v is not None and tVgV is not None and tVsV is not None
                                cute.copy(
                                    tma_atom_v,
                                    tVgV[None, iter, kv_coord - 1]
                                    if cutlass.const_expr(mPageTable is None)
                                    else tVgV[None, iter, 0, v_page_idx_prev],
                                    tVsV[None, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                )
                            else:
                                v_handle = load_kv_producer.acquire_and_advance()
                                assert local_tma_producer_state is not None
                                load_kv_cpasync_pipeline.producer_acquire(
                                    local_tma_producer_state,
                                    try_acquire_token=cutlass.Boolean(1),
                                    extra_tx_count=self.tma_copy_local_v_extra_bytes,
                                )
                                self.tma_page64_load_V(
                                    tma_atom_v,
                                    mV_page64,
                                    mPageTable,
                                    batch_coord,
                                    load_kv_cpasync_pipeline,
                                    sV,
                                    cta_rank_in_cluster,
                                    kv_coord - 1,
                                    local_tma_producer_state.index,
                                    iter,
                                    load_kv_cpasync_pipeline.producer_get_barrier(
                                        local_tma_producer_state
                                    ),
                                )
                                local_tma_producer_state.advance()
                        v_page_idx_prev = (
                            k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                        )
                        kv_coord += 1
                        # Prefetch next K page while V TMA is in flight.
                        if cutlass.const_expr(mPageTable is not None and self.use_tma_KV):
                            if kv_coord < seqlen_kv_loop_end:
                                k_page_idx = mPageTable[batch_coord, kv_coord]
                    # Vend: reuse v_page_idx_prev (= K[end-1]'s page), no extra GMEM read.
                    for iter in cutlass.range(self.iterations_pv, unroll=1):
                        if cutlass.const_expr(self.use_tma_KV):
                            v_handle = load_kv_producer.acquire_and_advance(
                                extra_tx_count=self.tma_copy_v_extra_bytes
                            )
                            assert tma_atom_v is not None and tVgV is not None and tVsV is not None
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, iter, seqlen_kv_loop_end - 1]
                                if cutlass.const_expr(mPageTable is None)
                                else tVgV[None, iter, 0, v_page_idx_prev],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                        else:
                            v_handle = load_kv_producer.acquire_and_advance()
                            assert local_tma_producer_state is not None
                            load_kv_cpasync_pipeline.producer_acquire(
                                local_tma_producer_state,
                                try_acquire_token=cutlass.Boolean(1),
                                extra_tx_count=self.tma_copy_local_v_extra_bytes,
                            )
                            self.tma_page64_load_V(
                                tma_atom_v,
                                mV_page64,
                                mPageTable,
                                batch_coord,
                                load_kv_cpasync_pipeline,
                                sV,
                                cta_rank_in_cluster,
                                seqlen_kv_loop_end - 1,
                                local_tma_producer_state.index,
                                iter,
                                load_kv_cpasync_pipeline.producer_get_barrier(
                                    local_tma_producer_state
                                ),
                            )
                            local_tma_producer_state.advance()

                work_tile = tile_sched.advance_to_next_work()
                # End of persistent scheduler loop
            if cutlass.const_expr(self.use_tma_KV):
                load_kv_producer.tail()
            else:
                assert local_tma_producer_state is not None
                load_kv_cpasync_pipeline.producer_tail(local_tma_producer_state)
            if warp_idx == self.load_warp_id:
                load_q_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  PAGE64 CP.ASYNC -> 2CTA UMMA RELAY
        # ///////////////////////////////////////////////////////////////////////////////
        # The local cp.async completion barrier is CTA-local.  One elected lane
        # from this warp in each CTA relays every completed token to the
        # cluster-visible PipelineAsyncUmma barrier consumed by the leader MMA
        # warp.  Direct cp.async arrivals at that UMMA barrier are racy because
        # a leader-CTA wait does not prove follower-CTA completion.
        if cutlass.const_expr(not self.use_tma_KV):
            if warp_idx == self.relay_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                producer_state_kv = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.kv_stage
                )
                consumer_state_cpasync = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.kv_stage
                )

                while work_tile.is_valid_tile:
                    curr_block_coord = work_tile.tile_idx
                    mma_block_coord = (
                        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                        curr_block_coord[1],
                        curr_block_coord[2],
                    )
                    batch_coord = curr_block_coord[2][1]
                    continue_cond = False
                    seqlen_q = cute.size(mQ_qdl.shape[0])
                    seqlen_k = max_seqlen_k
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        seqlen_q = cum_seqlen_q[batch_coord + 1] - cum_seqlen_q[batch_coord]
                        if cutlass.const_expr(self.pack_gqa):
                            seqlen_q *= self.qhead_per_kvhead
                        if cutlass.const_expr(cum_seqlen_k is not None):
                            seqlen_k = (
                                cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                            )
                        continue_cond = (
                            not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                                self.qk_mma_tiler[0], mma_block_coord[0], seqlen_q
                            )
                        )
                    if cutlass.const_expr(mSeqUsedQ is not None):
                        seqlen_q = mSeqUsedQ[batch_coord]
                        continue_cond = (
                            not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                                self.qk_mma_tiler[0], mma_block_coord[0], seqlen_q
                            )
                        )
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    if not continue_cond:
                        _, seqlen_kv_loop_steps = (
                            self.get_trip_start_count(
                                mma_block_coord,
                                self.qk_mma_tiler,
                                seqlen_q,
                                seqlen_k,
                                self.is_causal,
                                self.is_local,
                                window_size_left,
                                window_size_right,
                            )
                        )
                        for _ in cutlass.range(seqlen_kv_loop_steps, unroll=1):
                            for _k in cutlass.range_constexpr(self.iterations_qk):
                                consumer_state_cpasync, producer_state_kv = (
                                    self.relay_kv_token(
                                        load_kv_cpasync_pipeline,
                                        load_kv_pipeline,
                                        consumer_state_cpasync,
                                        producer_state_kv,
                                    )
                                )
                            for _v in cutlass.range_constexpr(self.iterations_pv):
                                consumer_state_cpasync, producer_state_kv = (
                                    self.relay_kv_token(
                                        load_kv_cpasync_pipeline,
                                        load_kv_pipeline,
                                        consumer_state_cpasync,
                                        producer_state_kv,
                                    )
                                )
                    work_tile = tile_sched.advance_to_next_work()

                load_kv_pipeline.producer_tail(producer_state_kv)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % 2 == 0

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                batch_coord = curr_block_coord[2][1]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(self.pack_gqa):
                        seqlen_q *= self.qhead_per_kvhead
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if cutlass.const_expr(mSeqUsedQ is not None):
                    seqlen_q = mSeqUsedQ[batch_coord]
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        self.get_trip_start_count(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )
                    )
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps

                    load_q_releaser = load_q_consumer.clone()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if seqlen_kv_loop_steps > 1:
                        # QK0
                        if is_leader_cta:
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            s_handle.commit()
                        for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                            # QKi
                            if is_leader_cta:
                                s_handle = mma_s_producer.acquire_and_advance()
                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_qk, unroll=1):
                                    tSrQ_slice = tSrQ[None, None, None, iter]
                                    k_handle = load_kv_consumer.wait_and_advance()
                                    tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    k_handle.release()
                                s_handle.commit()

                                # PVi-1
                                p_handle = p_mma_consumer.wait_and_advance()
                                o_handle = mma_corr_producer.acquire_and_advance()
                                pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                                for iter in cutlass.range(self.iterations_pv, unroll=1):
                                    v_handle = load_kv_consumer.wait_and_advance()
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                    tOtO_slice = tOtO_staged[None, None, None, iter]
                                    tOrP_slice = tOrP[None, None, None, p_handle.index]
                                    tOrV_slice = tOrV[None, None, None, v_handle.index]
                                    num_kphases = cute.size(tOrV_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            pv_tiled_mma,
                                            tOtO_slice,
                                            tOrP_slice[kphase_coord],
                                            tOrV_slice[kphase_coord],
                                            tOtO_slice,
                                        )
                                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    v_handle.release()
                                o_handle.commit()
                                p_handle.release()
                        if is_leader_cta:
                            # QKend
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # PVend-1
                            p_handle = p_mma_consumer.wait_and_advance()
                            o_handle = mma_corr_producer.acquire_and_advance()
                            pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_pv, unroll=1):
                                v_handle = load_kv_consumer.wait_and_advance()
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                tOtO_slice = tOtO_staged[None, None, None, iter]
                                tOrP_slice = tOrP[None, None, None, p_handle.index]
                                tOrV_slice = tOrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tOrV_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrP_slice[kphase_coord],
                                        tOrV_slice[kphase_coord],
                                        tOtO_slice,
                                    )
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            o_handle.commit()
                            p_handle.release()
                    else:
                        if is_leader_cta:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                    if is_leader_cta:
                        # PVend
                        p_handle = p_mma_consumer.wait_and_advance()
                        o_handle = mma_corr_producer.acquire_and_advance()
                        pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_consumer.wait_and_advance()
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            tOtO_slice = tOtO_staged[None, None, None, iter]
                            tOrP_slice = tOrP[None, None, None, p_handle.index]
                            tOrV_slice = tOrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tOrV_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrP_slice[kphase_coord],
                                    tOrV_slice[kphase_coord],
                                    tOtO_slice,
                                )
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                        o_handle.commit()
                        p_handle.release()
                work_tile = tile_sched.advance_to_next_work()
            # End of persistent scheduler loop
            mma_s_producer.tail()
            mma_corr_producer.tail()

        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(self.pack_gqa):
                        seqlen_q *= self.qhead_per_kvhead
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if cutlass.const_expr(mSeqUsedQ is not None):
                    seqlen_q = mSeqUsedQ[batch_coord]
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    row_max = -Float32.inf
                    row_max_prev = -Float32.inf
                    row_sum = 0.0

                    start_count, trip_count = self.get_trip_start_count(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                    end_count = start_count + trip_count
                    # require at least one softmax iteration for zero trip_count case;
                    # rely on masking this iteration for correctness
                    if end_count <= start_count:
                        start_count = 0
                        end_count = 1
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask, n_block_min_before_local_mask = (
                            FusedMask.get_trip_mask_bounds_via_block_info(
                                mma_block_coord,
                                self.qk_mma_tiler,
                                seqlen_q,
                                seqlen_k,
                                self.is_causal,
                                self.is_local,
                                window_size_left,
                                window_size_right,
                            )
                        )
                    cS_base = cute.make_identity_tensor(
                        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                    )
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)

                    for step in cutlass.range(start_count, end_count, 1, unroll=1):
                        cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)
                        if cutlass.const_expr(self.use_semantic_trip_range):
                            need_apply_mask = (
                                step >= n_block_min_causal_local_mask
                                or step < n_block_min_before_local_mask
                                or step == end_count - 1
                            )
                        else:
                            # Residual path only needs seqlen masking on the last K tile.
                            need_apply_mask = step == end_count - 1
                        # Si -> Pi
                        (
                            row_max,
                            row_sum,
                            mma_s_consumer,
                            p_mma_producer,
                            s_corr_producer,
                        ) = self.softmax_step(
                            (need_apply_mask, window_size_left, window_size_right),
                            (
                                row_max_prev,
                                row_sum,
                                seqlen_q,
                                seqlen_k,
                                scale_softmax_log2,
                            ),
                            (tStS, tScS_iter, sP, sRowMax, sScale),
                            (mma_s_consumer, p_mma_producer, s_corr_producer),
                        )
                        row_max_prev = row_max
                    sum_producer = self.store_sum_max(
                        row_max,
                        mLSE,
                        row_sum,
                        sSum,
                        sum_producer,
                        curr_block_coord,
                        seqlen_q,
                        cum_seqlen_q,
                        cuseqlen_q,
                        scale_softmax,
                    )
                work_tile = tile_sched.advance_to_next_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                continue_cond = False
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(self.pack_gqa):
                        seqlen_q *= self.qhead_per_kvhead
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if cutlass.const_expr(mSeqUsedQ is not None):
                    seqlen_q = mSeqUsedQ[batch_coord]
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    mO_qdl_eff = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        if cutlass.const_expr(self.pack_gqa):
                            mO_qdl_eff = cute.domain_offset(
                                ((Int32(0), cuseqlen_q), Int32(0), (Int32(0), Int32(0))),
                                mO_qdl,
                            )
                        else:
                            block_offset_o = (
                                cuseqlen_q,
                                Int32(0),
                                Int32(0),
                                ((Int32(0), Int32(0)), Int32(0)),
                            )
                            mO_qdl_eff = cute.domain_offset(
                                cute.select(block_offset_o, mode=[0, 2, 3]), mO_qdl
                            )

                    # (bM, bN, loopM, loopN, loopL)
                    gO_qdl = cute.flat_divide(
                        mO_qdl_eff, cute.select(self.pv_block_tiler, mode=[0, 1])
                    )
                    cO_qdl = cute.flat_divide(
                        cute.make_identity_tensor(mO_qdl_eff.shape),
                        cute.select(self.pv_block_tiler, mode=[0, 1]),
                    )

                    _, seqlen_kv_loop_steps = self.get_trip_start_count(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                    gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    tScS = qk_thr_mma.partition_C(cS)

                    # Empty step as the first step is no need for correction
                    stats_handle = s_corr_consumer.wait_and_advance()
                    stats_handle.release()
                    for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Oi-1 -> Oi
                        mma_corr_consumer, s_corr_consumer = self.correction_rescale(
                            scale_softmax_log2,
                            (s_corr_consumer, sScale),
                            (mma_corr_consumer, tOtO_staged, cO_staged),
                            self.epi_tile,
                        )
                    # O_partial -> O_final
                    mma_corr_consumer, sum_consumer = self.correction_epilog(
                        (seqlen_q, scale_output),
                        (sum_consumer, sSum),
                        (mma_corr_consumer, gO_staged, cO_staged, tOtO_staged),
                        self.epi_tile,
                    )
                work_tile = tile_sched.advance_to_next_work()
            # NOTE: tmem.free() moved to kernel end to enable cluster-wide sync

        # ///////////////////////////////////////////////////////////////////////////////
        #  Scheduler Warp (only for CLC dynamic scheduler)
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            is_first_cta_in_cluster = cta_rank_in_cluster == 0

            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                while work_tile.is_valid_tile:
                    tile_sched.prefetch_next_work()
                    work_tile = tile_sched.advance_to_next_work()
                tile_sched.producer_tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty warps reg dealloc
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            if warp_idx > self.load_warp_id:
                if not (warp_idx == self.sched_warp_id and is_first_cta_in_cluster):
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
        else:
            if warp_idx > self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cooperative TMEM Deallocation (2CTA)
        # ///////////////////////////////////////////////////////////////////////////////
        # All warps (including scheduler) have finished by this point.
        # Cluster-wide sync ensures both CTAs reach here before dealloc.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def relay_kv_token(
        self,
        pipeline_cpasync,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        consumer_state: pipeline.PipelineState,
        producer_state: pipeline.PipelineState,
    ):
        """Relay one CTA-local cp.async completion to the 2CTA UMMA barrier."""
        pipeline_cpasync.consumer_wait(consumer_state)
        with cute.arch.elect_one():
            pipeline_mma.producer_commit(producer_state)
        # The next local TMA producer is also gated by the cluster pipeline's
        # empty stage, so releasing the CTA-local stage here cannot overwrite
        # K/V before UMMA has finished consuming it.
        cute.arch.sync_warp()
        pipeline_cpasync.consumer_release(consumer_state)
        consumer_state.advance()
        producer_state.advance()
        return consumer_state, producer_state

    @cute.jit
    def tma_page64_load_K(
        self,
        tma_atom_k: cute.CopyAtom,
        mK_page64: cute.Tensor,
        mPageTable: cute.Tensor,
        batch_coord: Int32,
        pipeline_local,
        sK: cute.Tensor,
        cta_rank_in_cluster: Int32,
        n_block: Int32,
        stage: Int32,
        d_block: Int32,
        tma_bar_ptr,
    ):
        """Issue one CTA-local 64-token by d128 K descriptor transaction."""
        page_slot = n_block * self.cluster_shape_mn[0] + cta_rank_in_cluster
        page_idx = Int32(mK_page64.shape[2])
        if page_slot < mPageTable.shape[1]:
            page_idx = mPageTable[batch_coord, page_slot]

        sK_stage = sK[None, None, None, stage]
        sK_nd = cute.composition(
            sK_stage,
            cute.make_ordered_layout((64, self.qk_mma_tiler[2]), order=(0, 1)),
        )
        gK = cute.local_tile(
            mK_page64,
            (64, self.qk_mma_tiler[2]),
            (0, None, None),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_nd, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        cute.copy(
            tma_atom_k,
            tKgK[None, d_block, page_idx],
            tKsK,
            tma_bar_ptr=tma_bar_ptr,
        )

    @cute.jit
    def tma_page64_load_V(
        self,
        tma_atom_v: cute.CopyAtom,
        mV_page64: cute.Tensor,
        mPageTable: cute.Tensor,
        batch_coord: Int32,
        pipeline_local,
        sV: cute.Tensor,
        cta_rank_in_cluster: Int32,
        n_block: Int32,
        stage: Int32,
        pv_split: Int32,
        tma_bar_ptr,
    ):
        """Issue two CTA-local V page transactions against one 32-KiB barrier."""
        sV_stage = sV[None, None, None, stage]
        sV_td = cute.composition(
            sV_stage,
            cute.make_ordered_layout(
                (self.qk_mma_tiler[1], self.qk_mma_tiler[2]), order=(1, 0)
            ),
        )
        sV_dt = cute.make_tensor(sV_td.iterator, cute.select(sV_td.layout, mode=[1, 0]))
        gV = cute.local_tile(
            mV_page64,
            (self.qk_mma_tiler[2], 64),
            (None, 0, None),
        )
        d_block = pv_split * self.cluster_shape_mn[0] + cta_rank_in_cluster
        for page_half in cutlass.range_constexpr(self.cluster_shape_mn[0]):
            page_slot = n_block * self.cluster_shape_mn[0] + page_half
            page_idx = Int32(mV_page64.shape[2])
            if page_slot < mPageTable.shape[1]:
                page_idx = mPageTable[batch_coord, page_slot]
            sV_half = cute.local_tile(
                sV_dt,
                (self.qk_mma_tiler[2], 64),
                (0, page_half),
            )
            tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_half, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            cute.copy(
                tma_atom_v,
                tVgV[None, d_block, page_idx],
                tVsV,
                tma_bar_ptr=tma_bar_ptr,
            )

    @cute.jit
    def cpasync_paged_load_KV(
        self,
        paged_kv_manager: PagedKVManager,
        pipeline_cpasync: pipeline.PipelineAsync,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        cta_rank_in_cluster: Int32,
        n_block: Int32,
        stage: Int32,
        d_offset: int = 0,
    ):
        """Load one CTA-owned d128 K or V slice from paged GMEM.

        The caller first acquires the matching cluster-visible UMMA stage and
        loads the page table outside this @cute.jit boundary.  This routine
        issues ordinary 16-byte cp.async copies and associates their completion
        with the CTA-local barrier consumed by the relay warp.
        """
        tPrXPtr = paged_kv_manager.compute_X_ptr(K_or_V, d_offset)
        head_dim = (
            paged_kv_manager.head_dim_v_padded
            if cutlass.const_expr(K_or_V == "V")
            else paged_kv_manager.head_dim_padded
        )
        cta_tile_n = (
            self.qk_mma_tiler[1]
            if cutlass.const_expr(transpose)
            else self.qk_mma_tiler[1] // self.cluster_shape_mn[0]
        )
        order = (1, 0) if cutlass.const_expr(transpose) else (0, 1)

        sX_stage = sX[None, None, None, stage]
        sX_nd_layout = cute.make_ordered_layout((cta_tile_n, head_dim), order=order)
        sX_nd = cute.composition(sX_stage, sX_nd_layout)

        cX = cute.make_identity_tensor((cta_tile_n, head_dim))
        tXsX = paged_kv_manager.gmem_thr_copy_KV.partition_D(sX_nd)
        tXcX = paged_kv_manager.gmem_thr_copy_KV.partition_S(cX)
        tXc0X = paged_kv_manager.gmem_thr_copy_KV.get_slice(0).partition_S(cX)

        base_offset = n_block * self.qk_mma_tiler[1]
        if cutlass.const_expr(not transpose):
            base_offset += cta_tile_n * cta_rank_in_cluster
        seqlenk_row_limit = (
            paged_kv_manager.seqlen_k - base_offset - tXcX[0][0]
            if n_block >= 0
            else 0
        )
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = tXc0X[0, m, 0][0] < seqlenk_row_limit
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)

            # With two load warps, PagedKVManager caches two 64-row waves for
            # the 128-token logical block.  K is split into 64 rows per CTA,
            # so CTA1 selects pointer slot 1 rather than permuting source lanes
            # (the MLA +lane-offset scheme is specific to its 128-thread
            # loader).  Transposed V consumes both pointer slots in both CTAs
            # and therefore has no rank offset.
            ptr_slot = m // paged_kv_manager.gmem_threads_per_row
            if cutlass.const_expr(not transpose):
                ptr_slot += cta_rank_in_cluster * (
                    cta_tile_n // paged_kv_manager.num_threads
                )
            x_ptr_i64 = fa_utils.shuffle_sync(
                tPrXPtr[ptr_slot],
                m % paged_kv_manager.gmem_threads_per_row,
                width=paged_kv_manager.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                paged_kv_manager.mK_paged.element_type,
                x_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (paged_kv_manager.async_copy_elems,))

            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // paged_kv_manager.async_copy_elems
                mX_cur_copy_ki = mX_cur_copy[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(mX_cur_copy_ki.iterator, tXsX_k.layout)
                cute.copy(
                    paged_kv_manager.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load,
                )

        cute.arch.cp_async_commit_group()
        pipeline_cpasync.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def get_trip_start_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool],
        is_local: cutlass.Constexpr[bool],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
    ) -> Tuple[Int32, Int32]:
        if cutlass.const_expr(self.pack_gqa):
            # Packed rows interleave query heads, so a sequence-M block bound
            # cannot describe their causal range. Decode all KV blocks and let
            # the elementwise mask below map packed rows back to query tokens.
            return Int32(0), cute.ceil_div(seqlen_k, tile_shape[1])
        return FusedMask.get_trip_start_count_via_block_info(
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            is_causal,
            is_local,
            window_size_left,
            window_size_right,
        )

    @cute.jit
    def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS, sP, sRowMax, sScale = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % len(
            self.softmax_warp_ids
        )
        row_idx = thread_idx % self.cta_tiler[0]
        row_partial = warp_idx // self.cluster_shape_mn[0]
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)

        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if need_apply_mask:
            if cutlass.const_expr(self.pack_gqa):
                FusedMask.apply_mask_via_causal_local(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q // self.qhead_per_kvhead,
                    seqlen_k,
                    self.use_semantic_trip_range,
                    self.is_causal,
                    self.is_local,
                    window_size_left,
                    window_size_right,
                    index_transform=lambda index_q, index_k: (
                        index_q // self.qhead_per_kvhead,
                        index_k,
                    ),
                )
            else:
                FusedMask.apply_mask_via_causal_local(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    self.use_semantic_trip_range,
                    self.is_causal,
                    self.is_local,
                    window_size_left,
                    window_size_right,
                )
        old_row_max = row_max
        row_max_local = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        sRowMax[row_idx, row_partial] = row_max_local
        self.softmax_barrier.arrive_and_wait()
        row_max = max(sRowMax[row_idx, 0], sRowMax[row_idx, 1])
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
        acc_scale_ = scale_softmax_log2 * (old_row_max - row_max_safe)
        output_rescale = cute.math.exp2(acc_scale_, fastmath=True)
        acc_scale = output_rescale * 0.5
        if row_partial == 0:
            sScale[row_idx, stats_handle.index] = output_rescale
        cute.arch.fence_view_async_shared()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        # Acquire P write slot early — overlaps any pipeline stall with exp2 compute
        p_handle = p_mma_producer.acquire_and_advance()
        # Fragment-based FMA + exp2 + bf16 conversion
        # Trades SFU for FMA via polynomial emulation on a fraction of elements
        ex2_frg_tile = 32
        ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
        tTMEM_LOADrS_ex2 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(ex2_frg_tile))
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        tTMEM_STORErP_ex2 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(ex2_frg_tile))
        for j in cutlass.range_constexpr(ex2_frg_cnt):
            for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
                tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
                if cutlass.const_expr(self.ex2_emu_freq == 0):
                    tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
                    tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                        tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res
                        or j >= ex2_frg_cnt - 1
                        or j < self.ex2_emu_start_frg
                    ):
                        tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k, j], fastmath=True
                        )
                        tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                        )
                    else:
                        tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = ex2_emulation_2(
                            tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]
                        )
            tTMEM_STORErP_ex2[None, j].store(tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype))
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=128,
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_tiled_load)
        smem_store_thr = smem_store_tiled.get_slice(thread_idx)
        sP_mnp_layout = cute.make_ordered_layout(
            self.cta_tiler[:2] + (self.qk_acc_stage,), order=(0, 1, 2)
        )
        sP_mnp = cute.composition(sP, sP_mnp_layout)
        sP_smem_view = smem_store_thr.partition_D(sP_mnp)
        rP_smem_view = smem_store_thr.retile(tTMEM_STORErP)
        cute.copy(
            smem_store_thr,
            rP_smem_view,
            sP_smem_view[None, None, None, p_handle.index],
        )
        cute.arch.fence_view_async_shared()

        p_handle.commit()
        # Both row partitions must finish reading sRowMax before either can
        # overwrite it on the next KV tile.
        self.softmax_barrier.arrive_and_wait()
        # TODO: calc row sum with TensorSSA
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, sScale) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        scale = sScale[thread_idx % self.cta_tiler[0], stats_handle.index]
        cute.arch.fence_view_async_shared()
        stats_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            cO = cO_staged[None, None, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
            thr_store = tmem_store_atom.get_slice(thread_idx)
            tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_load.partition_D(cO_epi)
            tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
            tTMrO = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                    cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                ),
                self.pv_acc_dtype,
            )
            tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
            cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
            iter_num = cute.size(tTMEM_LOADtO, mode=[1])
            for i in cutlass.range(1, iter_num, unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2]),
                        (scale, scale),
                    )
                tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)

            for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2] = (
                    cute.arch.mul_packed_f32x2(
                        (
                            tTMrO[j, (iter_num - 1) % 2],
                            tTMrO[j + 1, (iter_num - 1) % 2],
                        ),
                        (scale, scale),
                    )
                )
            cute.copy(
                tmem_store_atom,
                tTMrO[None, (iter_num - 1) % 2],
                tTMEM_STOREtO[None, iter_num - 1, 0],
            )
        cute.arch.fence_view_async_tmem_store()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (seqlen_q, scale_output) = value_args
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_idx = thread_idx % self.cta_tiler[0]
        row_sum = sSum[row_idx, 0] + sSum[row_idx, 1]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale, scale),
                    )
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                output_bound = (
                    (self.qhead_per_kvhead, seqlen_q // self.qhead_per_kvhead)
                    if cutlass.const_expr(self.pack_gqa)
                    else seqlen_q
                )
                if cute.elem_less(tTMEM_LOADcO_i[0][0], output_bound):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
        o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def store_sum_max(
        self,
        row_max,
        mLSE,
        row_sum,
        sSum,
        sum_producer,
        current_block_coord,
        seqlen_q,
        cum_seqlen_q,
        cuseqlen_q,
        scale_softmax,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % len(
            self.softmax_warp_ids
        )
        row_idx = thread_idx % self.cta_tiler[0]
        row_partial = warp_idx // self.cluster_shape_mn[0]
        sum_handle = sum_producer.acquire_and_advance()
        sSum[row_idx, row_partial] = row_sum
        cute.arch.fence_view_async_shared()
        self.softmax_barrier.arrive_and_wait()
        if cutlass.const_expr(mLSE is not None):
            combined_row_sum = sSum[row_idx, 0] + sSum[row_idx, 1]
            combined_row_sum_is_zero_or_nan = (
                combined_row_sum == 0.0 or combined_row_sum != combined_row_sum
            )
            q_idx = current_block_coord[0] * self.cta_tiler[0] + row_idx
            hb_idx = (
                (current_block_coord[2][0], Int32(0))
                if cutlass.const_expr(cum_seqlen_q is not None)
                else current_block_coord[2]
            )
            lse_value = (
                scale_softmax * row_max + cute.math.log(combined_row_sum, fastmath=True)
                if not combined_row_sum_is_zero_or_nan
                else -Float32.inf
            )
            if thread_idx < self.cta_tiler[0] and cute.elem_less(q_idx, seqlen_q):
                global_q_idx = (
                    q_idx + cuseqlen_q if cutlass.const_expr(cum_seqlen_q is not None) else q_idx
                )
                mLSE[global_q_idx, hb_idx] = lse_value
        sum_handle.commit()
        return sum_producer
