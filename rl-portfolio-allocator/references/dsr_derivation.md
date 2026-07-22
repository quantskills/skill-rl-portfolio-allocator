# Differential Sharpe Ratio (DSR)

Reference: Moody & Saffell (1998)

## Formulas

```
A_t = A_{t-1} + η·(R_t − A_{t-1})              # 收益一阶矩 (EMA)
B_t = B_{t-1} + η·(R_t² − B_{t-1})             # 收益二阶矩 (EMA)
DSR_t = (B_{t-1}·ΔA − ½·A_{t-1}·ΔB) / (B_{t-1} − A_{t-1}²)^1.5
        其中 ΔA = R_t − A_{t-1}, ΔB = R_t² − B_{t-1}
```

Where:
- `R_t` = net daily return after all costs (commission, stamp tax, impact, borrowing)
- `η` = DSR EMA decay rate (hyperparameter)
- `A_t` = first moment (mean) EMA
- `B_t` = second moment EMA
- `DSR_t` = differential Sharpe reward for step t
