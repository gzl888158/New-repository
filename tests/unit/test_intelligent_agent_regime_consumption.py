from core.intelligent_agent import IntelligentTradingAgent


def test_volatility_uses_symbol_regime_dict_schema():
    agent = IntelligentTradingAgent.__new__(IntelligentTradingAgent)

    class RegimeEngine:
        def get_regime(self, *args, **kwargs):
            raise AssertionError("global regime API must not be used for a symbol lookup")

        def get_symbol_regime(self, symbol):
            assert symbol == "BTC-USDT-SWAP"
            return {
                "volatility": 0.04,
                "volatility_percentile": 0.8,
                "factor_scores": {},
            }

    agent._regime_engine = RegimeEngine()

    assert agent._get_volatility_for_symbol("BTC-USDT-SWAP") == 0.8


def test_volatility_falls_back_to_factor_score():
    agent = IntelligentTradingAgent.__new__(IntelligentTradingAgent)

    class RegimeEngine:
        def get_symbol_regime(self, symbol):
            return {"factor_scores": {"volatility": 0.65}}

    agent._regime_engine = RegimeEngine()

    assert agent._get_volatility_for_symbol("ETH-USDT-SWAP") == 0.65