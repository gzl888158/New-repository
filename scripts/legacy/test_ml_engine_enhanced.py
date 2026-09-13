"""
机器学习决策引擎功能测试
"""
import sys
sys.path.insert(0, '.')
import random
random.seed(42)
import numpy as np

from decision.ml_decision_engine import MLDecisionEngine

print('=== ML Decision Engine Functional Test ===')

# 1. 实例化引擎
engine = MLDecisionEngine({})
print(f'1. Engine created: enabled={engine.is_enabled()}, model={engine._model_id}')

# 2. 检查组件状态
print(f'2. Feature pipeline: {len(engine.feature_pipeline.get_feature_names())} features')
print(f'   Model ensemble fitted: {engine.model_ensemble.is_fitted()}')
print(f'   Calibrator fitted: {engine.calibrator.is_fitted()}')

# 3. 模拟训练数据
n = 30
raw_data = []
labels = []
for i in range(n):
    raw_data.append({
        'price_return_1m': random.uniform(-0.01, 0.01),
        'price_return_5m': random.uniform(-0.02, 0.02),
        'price_return_15m': random.uniform(-0.03, 0.03),
        'price_volatility_20': random.uniform(0.01, 0.05),
        'volume_ratio': random.uniform(0.5, 2.0),
        'volume_trend_5m': random.uniform(-0.5, 0.5),
        'rsi_14': random.uniform(30, 70),
        'macd_diff': random.uniform(-0.5, 0.5),
        'macd_signal': random.uniform(-0.3, 0.3),
        'bb_width_pct': random.uniform(0.01, 0.05),
        'bb_position': random.uniform(0, 1),
        'atr_14': random.uniform(0.5, 5.0),
        'momentum_10': random.uniform(-0.02, 0.02),
        'momentum_30': random.uniform(-0.05, 0.05),
    })
    labels.append(random.choice([-1, 0, 1]))

# 4. 训练模型
print(f'3. Training with {n} samples...')
meta = engine.train(raw_data, labels, symbol='BTC-USDT-SWAP')
print(f'   Result: version={meta.version}, status={meta.status.value}, samples={meta.n_samples_trained}')

# 5. 预测
print('4. Making prediction...')
pred = engine.predict(raw_data[0], symbol='BTC-USDT-SWAP')
print(f'   Direction: {pred.direction}, buy={pred.buy_prob:.4f}, sell={pred.sell_prob:.4f}, hold={pred.hold_prob:.4f}')
print(f'   Confidence: {pred.confidence:.4f}, calibrated={pred.calibrated}')

# 6. 批量预测
print('5. Batch prediction...')
batch_preds = engine.predict_batch(raw_data[:5], symbol='TEST')
print(f'   Batch size: {len(batch_preds)}, directions: {[p.direction for p in batch_preds]}')

# 7. 在线学习反馈
print('6. Feeding results for online learning...')
for i in range(min(10, len(raw_data))):
    engine.feed_result(raw_data[i], labels[i], symbol='BTC-USDT-SWAP')
print(f'   Buffer size: {engine.online_trainer.get_buffer_size()}')
drift = engine.online_trainer.get_drift_status()
print(f'   Drift status: level={drift["drift_level"]}, mean_err={drift["mean_error"]:.4f}')

# 8. 检查重训练
should_retrain, reason = engine.online_trainer.should_retrain()
print(f'7. Should retrain: {should_retrain}, reason: {reason}')

# 9. 状态摘要
status = engine.get_status()
print(f'8. Status: total_predictions={status["total_predictions"]}, fitted={status["fitted"]}')

# 10. 版本管理
versions = engine.model_registry.get_version_history(engine._model_id)
print(f'9. Versions: {len(versions)} registered, active={engine.model_registry.get_active_model(engine._model_id)}')

# 11. 特征重要性
imp = engine.get_feature_importance()
top3 = sorted(imp.items(), key=lambda x: -x[1])[:3]
print(f'10. Top features: {[(k, round(v,4)) for k,v in top3]}')

# 12. 引擎统计
stats = engine.get_engine_stats()
print(f'11. Engine stats: predictions={stats["predictions"]}, features={stats["n_features"]}')

# 13. 预测历史
history = engine.get_prediction_history(10)
print(f'12. Prediction history: {len(history)} entries')

# 14. Q-learning (if enabled)
if engine._q_learning:
    print(f'13. Q-table states: {len(engine._q_table)}')
else:
    print('13. Q-learning disabled')

# 15. 健康监控
degraded, deg = engine.health_monitor.check_quality_degradation()
print(f'14. Quality degraded: {degraded}, degradation={deg}')

# 16. 模型持久化测试
engine.model_registry.persist(engine._model_id)
import os
persist_path = os.path.join(engine.model_registry._persist_dir, f'{engine._model_id}_registry.json')
print(f'15. Model persisted: {persist_path} (exists={os.path.exists(persist_path)})')

# 17. reset
engine.reset()
print(f'16. After reset: predictions={engine._prediction_counter}, history_size={len(engine._prediction_history)}')

print()
print('=== ALL TESTS PASSED ===')
