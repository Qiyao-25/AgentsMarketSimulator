---
noteId: "38496420585e11f1b0bfe5d6c0c9ce88"
tags: []

---

# 官方基线方案

本包将 `题目.md` 中的两项任务整理为可运行的基线实现：

- `investment_agent.py`：BDI 风格 `InvestmentAgent`，包含行情、新闻、社交信息接入，个性参数，信念更新，处置效应偏差和决策思考。
- `exchange.py`：价格优先、时间优先的限价订单簿，以及用于识别洗售交易、虚假挂单、拉高出货的监管 Agent。
- `metrics.py`：PGR/PLR/DE、信念与行动的 Spearman 相关、换手率 Wasserstein 距离、F1 和响应延迟评分。
- `demo.py`：端到端冒烟运行示例。

运行方式：

```bash
python -m competition_solution.demo
python -m unittest discover -s competition_solution/tests
```
