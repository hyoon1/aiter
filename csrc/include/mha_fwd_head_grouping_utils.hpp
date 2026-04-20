#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mha_fwd.h"

#if ENABLE_CK
#include "fmha_fwd_head_grouping.hpp"

#include <iostream>
#include <string>

namespace aiter {

inline size_t get_element_size_from_dtype_string(const std::string& dtype)
{
    if(dtype == "fp16" || dtype == "bf16")
        return 2;
    if(dtype == "fp8" || dtype == "bf8" || dtype == "fp8bf16" || dtype == "fp8fp32")
        return 1;
    return 0;
}

template <typename FmhaFwdTraits, typename FmhaFwdArgs, typename FmhaFwdFn>
inline float maybe_dispatch_head_grouped_fwd(const ck_tile::stream_config& stream_config,
                                             const FmhaFwdTraits& traits,
                                             const FmhaFwdArgs& args,
                                             ck_tile::index_t num_heads,
                                             ck_tile::index_t num_heads_k,
                                             ck_tile::index_t batch_size,
                                             ck_tile::index_t seqlen_k,
                                             ck_tile::index_t head_size_q,
                                             ck_tile::index_t head_size_v,
                                             size_t elem_bytes_k,
                                             size_t elem_bytes_v,
                                             const std::string& q_dtype,
                                             FmhaFwdFn&& fmha_fwd_fn)
{
    namespace head_grouping = fmha_fwd_head_grouping;

    if(head_grouping::disabled_by_env())
    {
        if(head_grouping::log_enabled())
        {
            std::cout << "[LLC Head Grouping] disabled by env" << std::endl;
        }
        return -1.0f;
    }

    const auto group_size_opt = head_grouping::get_head_group_size(num_heads,
                                                                   num_heads_k,
                                                                   batch_size,
                                                                   seqlen_k,
                                                                   head_size_q,
                                                                   head_size_v,
                                                                   elem_bytes_k,
                                                                   elem_bytes_v);
    if(!group_size_opt.has_value() || group_size_opt.value() >= num_heads)
    {
        if(head_grouping::log_enabled())
        {
            std::cout << "[LLC Head Grouping] skipped (group_size not set or >= nhead)"
                      << std::endl;
        }
        return -1.0f;
    }

    if(head_grouping::log_enabled())
    {
        const std::string arch = ck_tile::get_device_name();
        const size_t llc_bytes = head_grouping::get_llc_cache_bytes(arch);
        const ck_tile::index_t gqa_ratio =
            (num_heads_k > 0 ? (num_heads / num_heads_k) : 1);
        const ck_tile::index_t group_sz = group_size_opt.value();
        const ck_tile::index_t n_groups = ck_tile::integer_divide_ceil(num_heads, group_sz);
        std::cout << "[LLC Head Grouping] enabled"
                  << " arch=" << (arch.empty() ? "unknown" : arch)
                  << " llc_mb=" << (llc_bytes / (1024ull * 1024ull))
                  << " nhead_q=" << num_heads << " nhead_k=" << num_heads_k
                  << " gqa_ratio=" << gqa_ratio << " group_size=" << group_sz
                  << " groups=" << n_groups << std::endl;
    }

    const bool use_blockscale_qscale = traits.qscale_type == quant_scale_enum::blockscale;
    auto dispatch_grouped_fwd        = [&](auto type_config_tag) {
        using TypeConfig = decltype(type_config_tag);
        return head_grouping::run_fwd_head_grouped<typename TypeConfig::QDataType,
                                                   typename TypeConfig::KDataType,
                                                   typename TypeConfig::VDataType,
                                                   typename TypeConfig::ODataType,
                                                   float,
                                                   typename TypeConfig::LSEDataType,
                                                   typename TypeConfig::RandValOutputDataType>(
            stream_config,
            traits,
            args,
            num_heads,
            num_heads_k,
            group_size_opt.value(),
            use_blockscale_qscale,
            [&](const auto& grouped_traits, auto& grouped_args, const auto& grouped_sc) {
                return fmha_fwd_fn(grouped_traits, grouped_args, grouped_sc);
            });
    };

    if(q_dtype == "fp16")
    {
        return dispatch_grouped_fwd(FmhaFwdTypeConfig<FmhaFwdFp16>{});
    }
    if(q_dtype == "bf16")
    {
        return dispatch_grouped_fwd(FmhaFwdTypeConfig<FmhaFwdBf16>{});
    }
    return -1.0f;
}

} // namespace aiter
#endif
