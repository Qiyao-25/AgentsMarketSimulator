"""初赛两项任务的正式提交实现。

本实现为确定性、自包含方案：默认不进行网络调用，不依赖外部服务，
只使用官方提交接口。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Mapping, Optional

from competition_solution.exchange import ExchangeAgent
from submission_interface.api import (
    AgentDecision,
    AlertRecord,
    CompetitionSubmission,
    MarketObservation,
    MatchResult,
    OrderRequest,
    TradeRecord,
)


POSITIVE_WORDS = {
    "accumulate",
    "analyst",
    "beat",
    "breakout",
    "bull",
    "bullish",
    "buy",
    "growth",
    "launch",
    "profit",
    "revenue",
    "strong",
    "surprise",
    "upgrade",
    "upside",
}

NEGATIVE_WORDS = {
    "avoid",
    "bear",
    "demand shock",
    "downgrade",
    "forced selling",
    "fraud",
    "lawsuit",
    "liquidity stress",
    "panic",
    "risk",
    "sell",
    "stay away",
}


class TeamSubmission(CompetitionSubmission):
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.reset(seed=0, config=config)

    def reset(self, seed: int = 0, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config.update(dict(config or {}))
        self.seed = seed
        self.exchange = ExchangeAgent()
        self.entity_flow: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=80))
        self.symbol_trades: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=120))
        self.alerted_keys: set[tuple[str, str, str]] = set()
        self.symbol_steps: Dict[str, int] = defaultdict(int)
        self.llm_client = None
        self.llm_calls = 0
        self.llm_cache: Dict[str, Dict[str, Any]] = {}
        self.use_llm = bool(self.config.get("use_llm", False))
        self.max_llm_calls = int(self.config.get("max_llm_calls", 24))
        if self.use_llm:
            self._init_llm_client()

    def decide(self, observation: MarketObservation) -> AgentDecision:
        if observation.symbol.startswith("STOCK_"):
            return self._decide_60d(observation)
        return self._decide_sim(observation)

    def _decide_sim(self, observation: MarketObservation) -> AgentDecision:
        closes = [float(item.close) for item in observation.klines if item.close > 0]
        price = closes[-1] if closes else max(float(observation.avg_cost), 1.0)
        equity = max(float(observation.cash) + observation.position * price, 1.0)
        momentum = _momentum(closes)
        text_score = _text_score(observation.news, observation.social_posts)
        pnl = 0.0
        if observation.position > 0 and observation.avg_cost > 0:
            pnl = price / float(observation.avg_cost) - 1.0

        action = "hold"
        target_turnover = 0.0
        reason = "the signals are mixed, so I avoid forcing a trade"

        if observation.position > 0 and (pnl > 0.24 or _contains(observation, ("take profit", "stretched"))):
            action = "sell"
            target_turnover = 0.11
            reason = "the position has a large unrealized gain and profit taking pressure is visible"
        elif text_score > 0.15 and momentum > 0.03:
            action = "buy"
            target_turnover = 0.08 if observation.position > 0 else 0.10
            reason = "bullish information and price momentum point in the same direction"
        elif observation.position > 0 and pnl < 0 and text_score < 0:
            action = "hold"
            reason = "bad news conflicts with loss aversion, so I hesitate to realize the loss"
        elif observation.position == 0 and text_score < -0.15 and momentum < -0.08:
            action = "buy"
            target_turnover = 0.07
            reason = "the panic looks crowded, so I take only a small contrarian probe"

        limit_price = price
        if action == "buy":
            limit_price = round(price * 1.004, 4)
        elif action == "sell":
            limit_price = round(price * 0.996, 4)

        quantity = 0
        if action == "buy":
            budget = min(float(observation.cash), equity * target_turnover)
            quantity = int(budget // max(limit_price, 1e-9))
        elif action == "sell":
            desired = int((equity * target_turnover) // max(limit_price, 1e-9))
            quantity = min(int(observation.position), max(1, desired))

        if quantity <= 0:
            action = "hold"
            quantity = 0
            limit_price = price

        action_value = {"sell": -1, "hold": 0, "buy": 1}[action]
        belief_score = _clip(action_value * (0.35 + 0.25 * abs(text_score) + 0.15 * abs(momentum)), -1.0, 1.0)
        sentiment_class = action_value
        thought = (
            f"{observation.symbol} momentum={momentum:.2f}, text_score={text_score:.2f}, "
            f"unrealized_pnl={pnl:.1%}; {reason}; I choose {action}."
        )

        return AgentDecision(
            agent_id=observation.agent_id,
            symbol=observation.symbol,
            action=action,
            quantity=int(quantity),
            limit_price=round(float(limit_price), 4),
            thought=thought,
            belief_score=round(float(belief_score), 4),
            sentiment_class=sentiment_class,
        )

    def _decide_60d(self, observation: MarketObservation) -> AgentDecision:
        closes = [float(item.close) for item in observation.klines if item.close > 0]
        price = closes[-1] if closes else max(float(observation.avg_cost), 1.0)
        equity = max(float(observation.cash) + observation.position * price, 1.0)
        step = self.symbol_steps[observation.symbol]
        self.symbol_steps[observation.symbol] += 1

        momentum_3 = _return(closes, 3)
        momentum_8 = _return(closes, 8)
        drawdown = _drawdown(closes[-15:])
        rebound = _rebound(closes[-10:])
        volatility = _volatility(closes[-12:])
        pnl = 0.0
        if observation.position > 0 and observation.avg_cost > 0:
            pnl = price / float(observation.avg_cost) - 1.0

        market_score = _clip(2.2 * momentum_3 + 1.4 * momentum_8 + 0.7 * rebound - 0.35 * volatility, -1.0, 1.0)
        llm_view = self._llm_market_view(observation, closes, market_score, pnl)
        llm_score = float(llm_view.get("belief_score", 0.0) or 0.0)
        if llm_view:
            market_score = _clip(0.85 * market_score + 0.15 * llm_score, -1.0, 1.0)

        action = "hold"
        target_turnover = 0.0
        reason = "signals are balanced after reviewing the 60-day anonymous market history"

        profitable = observation.position > 0 and pnl >= 0.035
        loss_making = observation.position > 0 and pnl <= -0.035
        scheduled_buy = step % 5 in {0, 1}
        scheduled_rebalance = step % 8 in {3, 5}

        if profitable and scheduled_rebalance:
            action = "sell"
            target_turnover = 0.065
            reason = "the holding is profitable and I realize part of the gain, showing disposition effect"
        elif observation.position > 0 and pnl >= 0.018 and step % 10 == 7:
            action = "sell"
            target_turnover = 0.045
            reason = "the position has a modest gain and I trim it instead of waiting for a perfect exit"
        elif loss_making:
            action = "hold"
            reason = "the position is losing money, so loss aversion makes me avoid realizing the loss"
        elif observation.position == 0 and (market_score >= -0.18 or scheduled_buy):
            action = "buy"
            target_turnover = 0.060 if market_score < 0 else 0.085
            reason = "I open exposure based on recent trend and portfolio rebalancing needs"
        elif observation.position > 0 and market_score > 0.22 and step % 5 == 2:
            action = "buy"
            target_turnover = 0.045
            reason = "positive short-term trend justifies adding moderately to an existing holding"
        elif observation.position > 0 and profitable and market_score < -0.20:
            action = "sell"
            target_turnover = 0.055
            reason = "the gain is at risk after momentum cools, so I lock in a smaller profit"

        limit_price = price
        quantity = 0
        if action == "buy":
            limit_price = round(price * 1.003, 4)
            budget = min(float(observation.cash), equity * target_turnover)
            quantity = int(budget // max(limit_price, 1e-9))
        elif action == "sell":
            limit_price = round(price * 0.997, 4)
            desired = int((equity * target_turnover) // max(limit_price, 1e-9))
            quantity = min(int(observation.position), max(1, desired))

        if quantity <= 0:
            action = "hold"
            quantity = 0
            limit_price = price

        action_value = {"sell": -1, "hold": 0, "buy": 1}[action]
        sentiment_class = action_value
        belief_score = _clip(
            action_value * (0.28 + 0.35 * abs(market_score) + 0.12 * min(abs(pnl) * 8.0, 1.0)),
            -1.0,
            1.0,
        )
        llm_note = str(llm_view.get("thought", "")).strip()
        if llm_note:
            llm_note = f" LLM view: {llm_note[:140]}"
        thought = (
            f"{observation.symbol} 60d step={step}, m3={momentum_3:.2%}, m8={momentum_8:.2%}, "
            f"drawdown={drawdown:.2%}, pnl={pnl:.2%}; {reason}; I choose {action}.{llm_note}"
        )

        return AgentDecision(
            agent_id=observation.agent_id,
            symbol=observation.symbol,
            action=action,
            quantity=int(quantity),
            limit_price=round(float(limit_price), 4),
            thought=thought,
            belief_score=round(float(belief_score), 4),
            sentiment_class=sentiment_class,
        )

    def _init_llm_client(self) -> None:
        if self.llm_client is not None:
            return
        try:
            from llm_helper import create_llm_client

            config_path = str(self.config.get("llm_config_path", "config.yaml"))
            self.llm_client = create_llm_client(config_path)
        except Exception:
            self.llm_client = None

    def _llm_market_view(
        self,
        observation: MarketObservation,
        closes: List[float],
        market_score: float,
        pnl: float,
    ) -> Dict[str, Any]:
        if not self.use_llm or self.llm_client is None or self.llm_calls >= self.max_llm_calls:
            return {}
        step = self.symbol_steps[observation.symbol]
        cache_key = f"{observation.symbol}:{step // 8}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        if step % 8 != 0:
            return {}

        system = (
            "You are a cautious retail-investor BDI agent for an anonymized 60-day market simulation. "
            "Return only JSON with keys thought, belief_score, sentiment_class. "
            "belief_score must be between -1 and 1; sentiment_class must be -1, 0, or 1."
        )
        recent = ", ".join(f"{value:.2f}" for value in closes[-10:])
        news = " ".join(str(item) for item in observation.news[:3])
        user = (
            f"symbol={observation.symbol}, day={observation.extra.get('day')}, "
            f"position={observation.position}, avg_cost={observation.avg_cost:.4f}, "
            f"cash={observation.cash:.2f}, recent_closes=[{recent}], "
            f"rule_market_score={market_score:.3f}, unrealized_pnl={pnl:.3f}, "
            f"news_excerpt={news[:900]}. "
            "Judge near-term belief and explain in one short sentence."
        )
        try:
            raw = self.llm_client(system, user)
            self.llm_calls += 1
            parsed = _parse_json(raw)
            view = {
                "thought": str(parsed.get("thought", "")).strip(),
                "belief_score": _clip(float(parsed.get("belief_score", 0.0) or 0.0), -1.0, 1.0),
                "sentiment_class": int(parsed.get("sentiment_class", 0) or 0),
            }
            if view["sentiment_class"] not in {-1, 0, 1}:
                view["sentiment_class"] = 0
            self.llm_cache[cache_key] = view
            return view
        except Exception:
            self.llm_client = None
            return {}

    def match_orders(
        self,
        orders: List[OrderRequest],
        last_prices: Mapping[str, float],
        tick: int,
    ) -> MatchResult:
        accepted: List[str] = []
        rejected: List[str] = []
        trades: List[TradeRecord] = []
        alerts: List[AlertRecord] = []
        close_prices = dict(last_prices)

        for order in orders:
            pre_alert = self._pre_trade_alert(order, close_prices, tick)
            if pre_alert is not None:
                alerts.append(pre_alert)

            result = self.exchange.submit_order(
                agent_id=order.agent_id,
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                quantity=order.quantity,
                timestamp=order.timestamp,
                entity_id=order.entity_id,
            )

            if result["accepted"]:
                accepted.append(order.order_id)
            else:
                rejected.append(order.order_id)

            for item in result["trades"]:
                trade = TradeRecord(**item)
                trades.append(trade)
                close_prices[trade.symbol] = trade.price
                self.symbol_trades[trade.symbol].append(
                    {
                        "timestamp": trade.timestamp,
                        "price": trade.price,
                        "quantity": trade.quantity,
                        "buyer_id": trade.buyer_id,
                        "seller_id": trade.seller_id,
                    }
                )
                pump_alert = self._post_trade_pump_alert(trade)
                if pump_alert is not None:
                    alerts.append(pump_alert)

            for item in result["alerts"]:
                alerts.append(AlertRecord(**item))

            entity = order.entity_id or order.agent_id
            self.entity_flow[str(entity)].append(
                {
                    "timestamp": order.timestamp,
                    "symbol": order.symbol,
                    "side": order.side,
                    "price": float(order.price),
                    "quantity": int(order.quantity),
                }
            )

        return MatchResult(
            trades=trades,
            accepted_order_ids=accepted,
            rejected_order_ids=rejected,
            alerts=alerts,
            close_prices=close_prices,
        )

    def _pre_trade_alert(
        self,
        order: OrderRequest,
        close_prices: Mapping[str, float],
        tick: int,
    ) -> Optional[AlertRecord]:
        last_price = float(close_prices.get(order.symbol, order.price) or order.price)
        if last_price <= 0:
            last_price = float(order.price)

        far_from_market = abs(float(order.price) / last_price - 1.0) >= 0.15
        large_order = int(order.quantity) >= 500
        if far_from_market and large_order:
            key = ("spoofing", order.symbol, order.order_id)
            if key not in self.alerted_keys:
                self.alerted_keys.add(key)
                return AlertRecord(
                    timestamp=int(order.timestamp or tick),
                    alert_type="spoofing",
                    severity=0.90,
                    order_id=order.order_id,
                    entity_id=str(order.entity_id or order.agent_id),
                    symbol=order.symbol,
                    action="warn_and_throttle_order",
                    thought=(
                        "The order is oversized and far away from the current reference price, "
                        "which can mislead visible depth without a clear execution intent."
                    ),
                )
        return None

    def _post_trade_pump_alert(self, trade: TradeRecord) -> Optional[AlertRecord]:
        recent = [
            item
            for item in self.symbol_trades[trade.symbol]
            if item["timestamp"] >= trade.timestamp - 10
        ]
        if len(recent) < 4:
            return None

        start_price = max(float(recent[0]["price"]), 1e-9)
        price_lift = float(recent[-1]["price"]) / start_price - 1.0
        buyers = {str(item["buyer_id"]) for item in recent}
        sellers = {str(item["seller_id"]) for item in recent}
        repeated_dump_seller = any(
            sum(1 for item in recent if str(item["seller_id"]) == seller) >= 4
            for seller in sellers
        )
        if price_lift >= 0.03 and len(buyers) >= 3 and repeated_dump_seller:
            key = ("pump_and_dump", trade.symbol, "|".join(sorted(sellers)))
            if key not in self.alerted_keys:
                self.alerted_keys.add(key)
                return AlertRecord(
                    timestamp=int(trade.timestamp),
                    alert_type="pump_and_dump",
                    severity=0.88,
                    order_id=f"{trade.buy_order_id},{trade.sell_order_id}",
                    entity_id="|".join(sorted(sellers)),
                    symbol=trade.symbol,
                    action="warn_and_review_coordinated_flow",
                    thought=(
                        "Several buyer accounts repeatedly lifted offers while one seller "
                        "distributed inventory into the rise, matching a pump-and-dump pattern."
                    ),
                )
        return None


def create_submission(config: Optional[Mapping[str, Any]] = None) -> CompetitionSubmission:
    return TeamSubmission(config)


def _momentum(closes: List[float]) -> float:
    if len(closes) < 2:
        return 0.0
    short = sum(closes[-3:]) / min(3, len(closes))
    long = sum(closes) / len(closes)
    return _clip((short / max(long, 1e-9) - 1.0) * 5.0, -1.0, 1.0)


def _return(closes: List[float], window: int) -> float:
    if len(closes) <= window or closes[-window - 1] <= 0:
        return 0.0
    return closes[-1] / closes[-window - 1] - 1.0


def _drawdown(closes: List[float]) -> float:
    if not closes:
        return 0.0
    peak = max(closes)
    return closes[-1] / max(peak, 1e-9) - 1.0


def _rebound(closes: List[float]) -> float:
    if not closes:
        return 0.0
    trough = min(closes)
    return closes[-1] / max(trough, 1e-9) - 1.0


def _volatility(closes: List[float]) -> float:
    returns = [
        closes[idx] / closes[idx - 1] - 1.0
        for idx in range(1, len(closes))
        if closes[idx - 1] > 0
    ]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return variance ** 0.5


def _text_score(news: List[str], social_posts: List[Mapping[str, Any]]) -> float:
    texts = list(news)
    texts.extend(str(post.get("text", "")) for post in social_posts)
    text = " ".join(texts).lower()
    pos = sum(text.count(word) for word in POSITIVE_WORDS)
    neg = sum(text.count(word) for word in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return _clip((pos - neg) / (pos + neg), -1.0, 1.0)


def _contains(observation: MarketObservation, phrases: tuple[str, ...]) -> bool:
    text = " ".join(observation.news).lower()
    text += " " + " ".join(str(post.get("text", "")) for post in observation.social_posts).lower()
    return any(phrase in text for phrase in phrases)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_json(raw: str) -> Dict[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}
