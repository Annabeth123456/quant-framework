"""
项目配置文件
"""

# 数据获取配置
DATA_CONFIG = {
    # 时间范围
    'start_date': '2018-01-01',
    'end_date': '2023-12-31',
    'train_end_date': '2022-12-31',  # 训练集结束日期
    
    # 股票池配置
    'markets': ['US', 'HK'],  # 美股和港股
    
    # 行业分类
    'industry_segments': {
        'upstream': ['材料', '设备'],  # 上游：材料和设备供应商
        'midstream': ['晶圆制造', '封装测试', 'IDM'],  # 中游：制造商
        'downstream': ['芯片设计', '应用', '终端产品']  # 下游：设计和应用
    },
    
    # 数据源配置
    'data_sources': {
        'us_stocks': 'yfinance',  # 美股数据源
        'hk_stocks': 'akshare',   # 港股数据源
        'financial_data': 'yfinance',  # 财务数据
        'market_data': 'yfinance'  # 市场数据
    }
}

# 因子模型配置
FACTOR_CONFIG = {
    # 技术因子
    'technical_factors': [
        'momentum_1m', 'momentum_3m', 'momentum_6m',  # 动量因子
        'volatility_1m', 'volatility_3m',  # 波动率因子
        'rsi_14', 'macd', 'bollinger_band',  # 技术指标
        'volume_ratio', 'turnover_rate'  # 交易量相关
    ],
    
    # 基本面因子
    'fundamental_factors': [
        'pe_ratio', 'pb_ratio', 'ps_ratio',  # 估值因子
        'roe', 'roa', 'gross_margin',  # 盈利能力
        'debt_to_equity', 'current_ratio',  # 财务健康度
        'revenue_growth', 'profit_growth',  # 增长性
        'r_and_d_intensity'  # 研发投入强度（特别适合科技股）
    ],
    
    # 行业特定因子
    'industry_specific_factors': [
        'market_share',  # 市场份额
        'product_cycle',  # 产品周期
        'tech_leadership',  # 技术领先度
        'customer_concentration'  # 客户集中度
    ],
    
    # 情感因子
    'sentiment_factors': [
        'news_sentiment',  # 新闻情感
        'analyst_ratings',  # 分析师评级
        'social_media_sentiment'  # 社交媒体情感
    ]
}

# 机器学习模型配置
ML_CONFIG = {
    # 特征工程
    'feature_engineering': {
        'normalization': 'standard_scaler',  # 标准化方法
        'feature_selection': 'recursive_feature_elimination',  # 特征选择方法
        'dimension_reduction': None  # 降维方法
    },
    
    # 模型选择
    'models': {
        'linear': {
            'type': 'linear_regression',
            'params': {}
        },
        'tree': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'boosting': {
            'type': 'xgboost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        },
        'deep_learning': {
            'type': 'lstm',
            'params': {
                'units': 50,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32
            }
        }
    },
    
    # 预测目标
    'prediction_targets': {
        'horizon': '5d',  # 预测5天后的收益率
        'target_type': 'return'  # 预测收益率
    },
    
    # 交叉验证
    'cross_validation': {
        'method': 'time_series_split',
        'n_splits': 5
    }
}

# 资产配置配置
ALLOCATION_CONFIG = {
    # 投资组合优化
    'optimization': {
        'method': 'mean_variance',  # 均值-方差优化
        'objective': 'sharpe_ratio',  # 优化目标：夏普比率
        'constraints': {
            'max_weight': 0.2,  # 单个资产最大权重
            'min_weight': 0.0,  # 单个资产最小权重
            'sector_constraints': {  # 行业权重约束
                'upstream': (0.2, 0.4),  # (最小权重, 最大权重)
                'midstream': (0.3, 0.5),
                'downstream': (0.2, 0.4)
            }
        }
    },
    
    # 风险管理
    'risk_management': {
        'risk_measure': 'conditional_value_at_risk',  # 风险度量：条件风险价值
        'confidence_level': 0.95,
        'max_drawdown_limit': 0.2,  # 最大回撤限制
        'volatility_target': 0.15  # 波动率目标
    },
    
    # 再平衡
    'rebalancing': {
        'frequency': 'monthly',  # 再平衡频率
        'threshold': 0.05  # 再平衡阈值
    }
}

# 强化学习配置
RL_CONFIG = {
    # 环境设置
    'environment': {
        'state_space': [
            'price_features',  # 价格特征
            'technical_indicators',  # 技术指标
            'portfolio_state',  # 投资组合状态
            'market_features'  # 市场特征
        ],
        'action_space': 'discrete',  # 离散动作空间
        'reward_function': 'sharpe_ratio',  # 奖励函数：夏普比率
        'episode_length': 252  # 一个回合的长度（一年交易日）
    },
    
    # 算法选择
    'algorithm': {
        'type': 'ppo',  # 近端策略优化算法
        'params': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        }
    },
    
    # 训练设置
    'training': {
        'total_timesteps': 1000000,
        'eval_frequency': 10000,
        'save_frequency': 50000
    }
}

# 回测配置
BACKTEST_CONFIG = {
    # 回测设置
    'settings': {
        'initial_capital': 1000000,  # 初始资金
        'commission': 0.001,  # 交易手续费
        'slippage': 0.001,  # 滑点
        'lot_size': 100,  # 最小交易单位
        'frequency': 'daily'  # 回测频率
    },
    
    # 性能指标
    'metrics': [
        'total_return',  # 总收益率
        'annual_return',  # 年化收益率
        'sharpe_ratio',  # 夏普比率
        'max_drawdown',  # 最大回撤
        'volatility',  # 波动率
        'win_rate',  # 胜率
        'profit_factor',  # 盈亏比
        'sortino_ratio',  # 索提诺比率
        'calmar_ratio',  # 卡玛比率
        'omega_ratio'  # 欧米伽比率
    ],
    
    # 基准
    'benchmark': {
        'us': '^IXIC',  # 纳斯达克指数
        'hk': '^HSI'  # 恒生指数
    }
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'style': 'seaborn',
    'figsize': (12, 8),
    'dpi': 100,
    'save_format': 'png'
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/trading_system.log'
} 