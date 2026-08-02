# QuantSkills Entry

**Name**: skill-rl-portfolio-allocator

**Description**: PPO-based dynamic factor-weight allocator on CSI300 with causal market state, bounded net-return reward, weekly actions, embedded costs, walk-forward gates, and approval-controlled publication.

## Runtime Entries

- **Codex**: Read this file, then follow the appropriate inner SKILL.md file.
- **Claude Code**: `CLAUDE.md`
- **Cursor**: `.cursor/rules/skill-rl-portfolio-allocator.mdc`
- **Hermes**: `HERMES.md`
- **OpenClaw**: `OPENCLAW.md`

## Research contract

研究结果必须经过 causal `market_state`、独立 scaler、周频 action、成本敏感性和多 fold/seed walk-forward。单次 backtest 或 smoke 运行不能背书生产；gate 失败时必须 fail closed，不得发布生产 checkpoint 或 allocations。

标准入口是 `./run_pipeline.sh --research-smoke` 或 `./run_pipeline.sh --research-full`。只有显式提供通过 full walk-forward 生成的 approval，才允许 `./run_pipeline.sh --publish --approval PATH`。

## Dual Mode

- **Research and training**: Follow `rl-portfolio-allocator/SKILL.md`
- **Read-only production allocation queries**: Follow `rl-portfolio-allocator-production/SKILL.md`

## Boundaries

本仓库为研究与工程材料。**不构成投资建议、不承诺收益、不代表 QuantSkills / Panda data / Codex / Claude Code / Cursor / Hermes / OpenClaw 的官方背书。** 不得记录或提交 Panda data 凭据。
