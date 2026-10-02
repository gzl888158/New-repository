import numpy as np

from decision.rl_agent import AgentMode, RLAction, StateEncoding, TradingRLAgent


def _agent(tmp_path, **overrides):
    config = {
        "rl_agent": {
            "name": "test_agent",
            "enabled": True,
            "mode": AgentMode.ONLINE_FINETUNE.value,
            "persist_dir": str(tmp_path),
            "drift_min_samples": 10,
            "drift_window_size": 10,
            "drift_drop_threshold": 0.5,
        }
    }
    config["rl_agent"].update(overrides)
    return TradingRLAgent(config)


def test_live_decisions_and_online_training_are_opt_in(tmp_path):
    agent = _agent(tmp_path)

    assert agent._decision_enabled is False
    assert agent._online_training_enabled is False
    assert agent.get_parameter_adjustment("leverage", 3.0) == 3.0
    assert agent.get_stats()["decision_enabled"] is False
    assert agent.get_stats()["online_training_enabled"] is False


def test_state_encoding_includes_portfolio_context(tmp_path):
    agent = _agent(tmp_path)
    encoded = agent.encode_state(StateEncoding(
        factor_score=0.7,
        regime_confidence=0.8,
        utilization_rate=0.6,
        avg_correlation=0.4,
    ))

    assert encoded.shape == (12,)
    assert np.allclose(encoded[-4:], [0.7, 0.8, 0.6, 0.4])


def test_regime_engine_fuses_confidence_and_weighted_factor_scores(tmp_path):
    agent = _agent(tmp_path)

    class RegimeEngine:
        def get_symbol_regime(self, symbol):
            return {
                "regime": "trend_bullish",
                "confidence": 0.85,
                "factor_scores": {"trend": 0.8, "momentum": 0.4},
                "factor_weights": {"trend": 0.75, "momentum": 0.25},
            }

    agent.set_regime_engine(RegimeEngine())
    encoded = agent.encode_state(StateEncoding(symbol="BTC-USDT-SWAP"))

    assert np.isclose(encoded[-4], 0.7)
    assert np.isclose(encoded[-3], 0.85)


def test_parameter_adjustment_requires_state_and_uses_greedy_action(tmp_path):
    agent = _agent(tmp_path, decision_enabled=True)
    assert agent.get_parameter_adjustment("leverage", 5.0) == 5.0
    agent.select_action = lambda *args, **kwargs: (0, RLAction.INCREASE_LEVERAGE, 1.0)

    adjusted = agent.get_parameter_adjustment("leverage", 5.0, StateEncoding())

    assert adjusted > 5.0
    assert adjusted <= agent._param_bounds["leverage"][1]


def test_reward_drift_freezes_online_training_until_explicit_reset(tmp_path):
    agent = _agent(tmp_path, online_training_enabled=True)

    for reward in [1.0] * 5 + [-1.0] * 5:
        agent.end_episode(reward)

    assert agent._training_frozen is True
    assert "reward_drift" in agent._training_freeze_reason
    assert agent.train_step() is None

    agent.reset_training_drift_freeze()
    assert agent._training_frozen is False
    assert agent._training_freeze_reason == ""


def test_reward_drift_freeze_survives_restart(tmp_path):
    config = {
        "name": "test_agent",
        "enabled": True,
        "mode": AgentMode.ONLINE_FINETUNE.value,
        "persist_dir": str(tmp_path),
        "online_training_enabled": True,
        "drift_min_samples": 10,
        "drift_window_size": 10,
        "drift_drop_threshold": 0.5,
    }
    agent = _agent(tmp_path, **{key: value for key, value in config.items()
                                if key != "persist_dir"})
    for reward in [1.0] * 5 + [-1.0] * 5:
        agent.end_episode(reward)

    restarted = _agent(tmp_path, online_training_enabled=True)

    assert restarted._training_frozen is True
    assert "reward_drift" in restarted._training_freeze_reason


def test_old_checkpoint_shape_is_not_loaded(tmp_path):
    old_agent = _agent(tmp_path, n_states=8)
    old_agent._q_network["W_shared1"][0, 0] = 123.0
    old_agent._save()

    new_agent = _agent(tmp_path)

    assert new_agent._q_network["W_shared1"].shape[0] == 12
    assert new_agent._q_network["W_shared1"][0, 0] != 123.0