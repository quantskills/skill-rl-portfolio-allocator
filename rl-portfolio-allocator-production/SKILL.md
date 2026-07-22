---
name: rl-portfolio-allocator-production
description: Read-only queries against the pre-computed RL portfolio allocations. Does not train or recompute.
license: GPL-3.0-only
tags: [quant, rl, portfolio, production, ashare]
---

# RL 组合权重优化器(只读生产查询)

只读模式:从 `data/allocations.parquet` 读取由 `../rl-portfolio-allocator/scripts/allocate.py` 落盘的每日持仓。**不训练、不重算、不联网。**

## 用法
```bash
python scripts/query.py --latest
python scripts/query.py --range 2024-01-01 2024-06-30
```

## 字段
`trade_date, symbol, weight, side(long|short|cash), factor_weights(JSON), composite_score, strategy_id='RLPA', data_version='real-v1', update_time`
