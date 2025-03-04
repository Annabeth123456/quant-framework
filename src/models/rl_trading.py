"""
强化学习交易决策模块

该模块实现了基于强化学习的交易策略，用于根据市场状态做出最优的交易决策。
"""

import os
import numpy as np
import pandas as pd
import logging
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
import sys
import pickle
import json
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置
from config.config import RL_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """交易环境类，实现了gym.Env接口，用于强化学习训练"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
               max_steps: int = None, transaction_cost: float = 0.001,
               reward_function: str = 'sharpe', window_size: int = 10):
        """
        初始化交易环境
        
        Args:
            data: 股票数据，包含OHLCV和特征
            initial_balance: 初始资金
            max_steps: 最大步数，默认为数据长度
            transaction_cost: 交易成本
            reward_function: 奖励函数类型，可选'sharpe'、'returns'、'sortino'
            window_size: 观察窗口大小
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_function = reward_function
        self.window_size = window_size
        
        # 数据预处理
        self.preprocess_data()
        
        # 设置最大步数
        self.max_steps = max_steps or len(self.data) - self.window_size
        
        # 定义动作空间：0=持有现金，1=买入，2=卖出
        self.action_space = spaces.Discrete(3)
        
        # 定义观察空间：[窗口内的特征, 当前持仓, 当前资金]
        # 计算特征数量
        feature_columns = [col for col in self.data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        n_features = len(feature_columns)
        
        obs_shape = (self.window_size * n_features + 2,)  # +2表示当前持仓和资金状态
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        # 保存特征列名
        self.feature_columns = feature_columns
        
        # 初始化环境状态
        self.reset()
    
    def preprocess_data(self):
        """预处理数据"""
        # 确保数据按时间排序
        self.data = self.data.sort_index()
        
        # 确保有必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.data.columns for col in required_columns):
            logger.error(f"数据缺少必要的列: {required_columns}")
            raise ValueError(f"数据缺少必要的列: {required_columns}")
        
        # 丢弃包含NaN的行
        self.data = self.data.dropna()
        
        # 归一化特征，以便于RL训练
        # 排除OHLCV和目标列
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        exclude_cols.extend([col for col in self.data.columns if col.startswith('future_')])
        
        # 对每个特征进行标准化
        for col in self.data.columns:
            if col not in exclude_cols:
                mean = self.data[col].mean()
                std = self.data[col].std()
                if std != 0:
                    self.data[col] = (self.data[col] - mean) / std
    
    def reset(self):
        """重置环境状态"""
        # 重置当前步数
        self.current_step = 0
        
        # 重置资金和持仓
        self.balance = self.initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.total_value = self.balance + self.asset_value
        
        # 重置交易历史
        self.trades = []
        self.portfolio_values = [self.total_value]
        self.returns = []
        
        # 返回初始观察
        return self._get_observation()
    
    def _get_observation(self):
        """获取当前观察"""
        # 获取当前窗口内的数据
        window_data = self.data.iloc[self.current_step:self.current_step + self.window_size]
        
        # 提取特征数据
        features = window_data[self.feature_columns].values.flatten()
        
        # 添加当前持仓和资金状态
        normalized_shares = self.shares_held / (self.initial_balance / self.data.iloc[self.current_step]['Close'])
        normalized_balance = self.balance / self.initial_balance
        
        # 组合观察
        observation = np.hstack([features, normalized_shares, normalized_balance])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """
        执行动作并更新环境状态
        
        Args:
            action: 动作，0=持有现金，1=买入，2=卖出
            
        Returns:
            Tuple: (observation, reward, done, info)
        """
        # 获取当前价格
        current_price = self.data.iloc[self.current_step + self.window_size - 1]['Close']
        
        # 执行交易动作
        self._execute_trade_action(action, current_price)
        
        # 更新当前步数
        self.current_step += 1
        
        # 检查是否达到最大步数
        done = self.current_step >= self.max_steps
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 更新历史数据
        self.portfolio_values.append(self.total_value)
        if len(self.portfolio_values) > 1:
            daily_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
            self.returns.append(daily_return)
        
        # 返回信息
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'asset_value': self.asset_value,
            'total_value': self.total_value,
            'return': self.returns[-1] if self.returns else 0
        }
        
        return observation, reward, done, info
    
    def _execute_trade_action(self, action, current_price):
        """
        执行交易动作
        
        Args:
            action: 动作，0=持有现金，1=买入，2=卖出
            current_price: 当前价格
        """
        if action == 0:  # 持有现金，不操作
            pass
        
        elif action == 1:  # 买入
            if self.balance > 0:
                # 计算可买入的最大股数
                max_shares = self.balance / (current_price * (1 + self.transaction_cost))
                # 买入所有可能的股票
                shares_to_buy = int(max_shares)
                
                if shares_to_buy > 0:
                    # 计算买入成本
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    # 更新余额和持仓
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    
                    # 记录交易
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
        
        elif action == 2:  # 卖出
            if self.shares_held > 0:
                # 卖出所有持仓
                shares_to_sell = self.shares_held
                
                # 计算卖出收入
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                # 更新余额和持仓
                self.balance += revenue
                self.shares_held = 0
                
                # 记录交易
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue
                })
        
        # 更新资产价值和总价值
        self.asset_value = self.shares_held * current_price
        self.total_value = self.balance + self.asset_value
    
    def _calculate_reward(self):
        """
        计算奖励
        
        Returns:
            float: 奖励值
        """
        if self.reward_function == 'returns':
            # 使用每日收益率作为奖励
            if len(self.returns) > 0:
                return self.returns[-1]
            else:
                return 0.0
        
        elif self.reward_function == 'sharpe':
            # 使用夏普比率作为奖励（需要足够的历史数据）
            if len(self.returns) > 10:
                mean_return = np.mean(self.returns[-10:])
                std_return = np.std(self.returns[-10:])
                
                # 避免除以零
                if std_return == 0:
                    return mean_return
                
                sharpe = mean_return / std_return
                return sharpe
            else:
                return 0.0
        
        elif self.reward_function == 'sortino':
            # 使用索提诺比率作为奖励（只考虑下行风险）
            if len(self.returns) > 10:
                mean_return = np.mean(self.returns[-10:])
                # 只计算负收益的标准差
                neg_returns = [r for r in self.returns[-10:] if r < 0]
                
                # 避免没有负收益的情况
                if len(neg_returns) == 0:
                    return mean_return
                
                downside_std = np.std(neg_returns)
                
                # 避免除以零
                if downside_std == 0:
                    return mean_return
                
                sortino = mean_return / downside_std
                return sortino
            else:
                return 0.0
        
        else:
            return 0.0
    
    def render(self, mode='human'):
        """渲染环境状态（可视化）"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Asset value: ${self.asset_value:.2f}")
            print(f"Total value: ${self.total_value:.2f}")
            if self.returns:
                print(f"Last return: {self.returns[-1]:.4f}")
    
    def plot_performance(self, benchmark_data=None):
        """
        绘制策略表现
        
        Args:
            benchmark_data: 基准数据，如果提供，将与策略表现进行比较
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制投资组合价值曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_values, label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        # 绘制收益率曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.returns, label='Strategy Returns')
        
        # 如果有基准数据，也绘制基准收益率
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()
            plt.plot(benchmark_returns, label='Benchmark Returns')
        
        plt.title('Daily Returns')
        plt.xlabel('Steps')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class TensorboardCallback(BaseCallback):
    """用于记录额外信息到Tensorboard的回调"""
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        # 记录当前价值
        self.logger.record('portfolio/value', self.training_env.get_attr('total_value')[0])
        # 记录持有的股票数量
        self.logger.record('portfolio/shares_held', self.training_env.get_attr('shares_held')[0])
        # 记录当前余额
        self.logger.record('portfolio/balance', self.training_env.get_attr('balance')[0])
        return True


class RLTrader:
    """强化学习交易者类，用于训练和评估RL交易策略"""
    
    def __init__(self, config: Dict = None):
        """
        初始化RL交易者
        
        Args:
            config: 配置字典，默认使用RL_CONFIG
        """
        self.config = config or RL_CONFIG
        self.model = None
        self.env = None
        self.train_env = None
        self.eval_env = None
        
        # 创建模型保存目录
        os.makedirs('models/rl', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)
        
        logger.info("RL交易者初始化完成")
    
    def create_environment(self, data: pd.DataFrame, is_training: bool = True) -> gym.Env:
        """
        创建交易环境
        
        Args:
            data: 股票数据
            is_training: 是否用于训练
            
        Returns:
            gym.Env: 创建的环境
        """
        # 分割数据为训练集和测试集
        if is_training:
            # 使用前80%的数据进行训练
            train_size = int(len(data) * 0.8)
            env_data = data.iloc[:train_size]
        else:
            # 使用后20%的数据进行测试
            train_size = int(len(data) * 0.8)
            env_data = data.iloc[train_size:]
        
        # 创建环境
        env = TradingEnvironment(
            data=env_data,
            initial_balance=self.config['environment'].get('initial_balance', 10000.0),
            transaction_cost=self.config['environment'].get('transaction_cost', 0.001),
            reward_function=self.config['environment'].get('reward_function', 'sharpe'),
            window_size=self.config['environment'].get('window_size', 10)
        )
        
        # 包装环境以记录性能指标
        env = Monitor(env, os.path.join('results/metrics', 'rl_trading_monitor'))
        
        return env
    
    def create_model(self, env: gym.Env) -> Any:
        """
        创建RL模型
        
        Args:
            env: 交易环境
            
        Returns:
            Any: 创建的模型
        """
        algorithm_type = self.config['algorithm']['type']
        params = self.config['algorithm']['params']
        
        # 根据算法类型创建模型
        if algorithm_type == 'ppo':
            model = PPO(
                policy='MlpPolicy',
                env=env,
                **params
            )
        elif algorithm_type == 'a2c':
            model = A2C(
                policy='MlpPolicy',
                env=env,
                **params
            )
        elif algorithm_type == 'dqn':
            model = DQN(
                policy='MlpPolicy',
                env=env,
                **params
            )
        else:
            logger.error(f"不支持的算法类型: {algorithm_type}")
            raise ValueError(f"不支持的算法类型: {algorithm_type}")
        
        return model
    
    def train(self, data: pd.DataFrame) -> Any:
        """
        训练RL交易策略
        
        Args:
            data: 股票数据
            
        Returns:
            Any: 训练好的模型
        """
        logger.info("开始训练RL交易策略")
        
        # 创建训练环境
        self.train_env = self.create_environment(data, is_training=True)
        
        # 创建向量化环境
        vec_env = DummyVecEnv([lambda: self.train_env])
        
        # 创建模型
        self.model = self.create_model(vec_env)
        
        # 创建评估环境
        self.eval_env = self.create_environment(data, is_training=False)
        
        # 创建评估回调
        eval_callback = EvalCallback(
            eval_env=self.eval_env,
            n_eval_episodes=10,
            eval_freq=1000,
            log_path='results/metrics',
            best_model_save_path='models/rl'
        )
        
        # 创建Tensorboard回调
        tensorboard_callback = TensorboardCallback()
        
        # 训练模型
        total_timesteps = self.config['training']['total_timesteps']
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, tensorboard_callback]
        )
        
        logger.info(f"RL交易策略训练完成，共训练 {total_timesteps} 步")
        
        # 保存模型
        self.save_model()
        
        return self.model
    
    def save_model(self, path: str = 'models/rl/rl_model'):
        """
        保存RL模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            logger.error("没有训练好的模型可保存")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        self.model.save(path)
        
        logger.info(f"RL模型已保存到 {path}")
    
    def load_model(self, path: str = 'models/rl/rl_model', env: gym.Env = None):
        """
        加载RL模型
        
        Args:
            path: 加载路径
            env: 交易环境，如果为None则使用self.train_env
        """
        if not os.path.exists(path + '.zip'):
            logger.error(f"模型文件 {path}.zip 不存在")
            return
        
        # 确定环境
        if env is None:
            if self.train_env is None:
                logger.error("没有创建交易环境")
                return
            env = self.train_env
        
        # 确定算法类型
        algorithm_type = self.config['algorithm']['type']
        
        # 加载模型
        if algorithm_type == 'ppo':
            self.model = PPO.load(path, env=env)
        elif algorithm_type == 'a2c':
            self.model = A2C.load(path, env=env)
        elif algorithm_type == 'dqn':
            self.model = DQN.load(path, env=env)
        else:
            logger.error(f"不支持的算法类型: {algorithm_type}")
            return
        
        logger.info(f"已加载RL模型从 {path}")
    
    def evaluate(self, data: pd.DataFrame, n_episodes: int = 1) -> Dict:
        """
        评估RL交易策略
        
        Args:
            data: 股票数据
            n_episodes: 评估回合数
            
        Returns:
            Dict: 评估结果
        """
        if self.model is None:
            logger.error("没有训练好的模型可评估")
            return {}
        
        logger.info(f"开始评估RL交易策略，回合数: {n_episodes}")
        
        # 创建评估环境
        if self.eval_env is None:
            self.eval_env = self.create_environment(data, is_training=False)
        
        # 评估模型
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=n_episodes)
        
        # 重置环境以保存详细结果
        obs = self.eval_env.reset()
        done = False
        
        # 运行一个完整的回合
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.eval_env.step(action)
        
        # 获取评估结果
        portfolio_values = self.eval_env.portfolio_values
        returns = self.eval_env.returns
        trades = self.eval_env.trades
        
        # 计算各项指标
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = total_return * (252 / len(portfolio_values))
        
        # 计算夏普比率
        if len(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # 计算胜率
        if len(trades) > 0:
            profitable_trades = sum(1 for trade in trades if trade.get('action') == 'sell' and trade.get('revenue', 0) > 0)
            win_rate = profitable_trades / len(trades) if len(trades) > 0 else 0
        else:
            win_rate = 0
        
        # 保存评估结果
        evaluation_results = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'n_trades': len(trades)
        }
        
        # 保存评估结果到文件
        with open('results/metrics/rl_evaluation.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        logger.info(f"评估完成，平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # 绘制性能图表
        self.plot_performance(data)
        
        return evaluation_results
    
    def plot_performance(self, data: pd.DataFrame):
        """
        绘制策略表现
        
        Args:
            data: 股票数据
        """
        if self.eval_env is None:
            logger.error("没有评估环境，无法绘制性能图表")
            return
        
        # 提取收盘价作为基准
        benchmark_data = data['Close']
        
        # 使用环境的方法绘制性能
        self.eval_env.plot_performance(benchmark_data)
        
        # 保存图表
        plt.savefig('results/figures/rl_performance.png')
        plt.close()
        
        # 绘制交易动作
        self.plot_trades(data)
    
    def plot_trades(self, data: pd.DataFrame):
        """
        绘制交易动作
        
        Args:
            data: 股票数据
        """
        if self.eval_env is None or not hasattr(self.eval_env, 'trades'):
            logger.error("没有交易数据，无法绘制交易图表")
            return
        
        trades = self.eval_env.trades
        
        if not trades:
            logger.warning("没有交易记录，无法绘制交易图表")
            return
        
        # 绘制价格和交易点
        plt.figure(figsize=(12, 6))
        
        # 使用测试数据的收盘价
        plt.plot(data['Close'].values, label='Close Price')
        
        # 标记买入点和卖出点
        for trade in trades:
            step = trade['step']
            if step < len(data):
                if trade['action'] == 'buy':
                    plt.scatter(step, data['Close'].iloc[step], color='green', marker='^', s=100, label='Buy')
                elif trade['action'] == 'sell':
                    plt.scatter(step, data['Close'].iloc[step], color='red', marker='v', s=100, label='Sell')
        
        # 去除重复的图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title('Trading Actions')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.grid(True)
        
        # 保存图表
        plt.savefig('results/figures/rl_trades.png')
        plt.close()
    
    def run_trading_simulation(self, data: pd.DataFrame, initial_balance: float = 10000.0) -> Dict:
        """
        运行交易模拟
        
        Args:
            data: 股票数据
            initial_balance: 初始资金
            
        Returns:
            Dict: 模拟结果
        """
        if self.model is None:
            logger.error("没有训练好的模型可用于模拟")
            return {}
        
        logger.info("开始运行交易模拟")
        
        # 创建模拟环境
        sim_env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            transaction_cost=self.config['environment'].get('transaction_cost', 0.001),
            reward_function=self.config['environment'].get('reward_function', 'sharpe'),
            window_size=self.config['environment'].get('window_size', 10)
        )
        
        # 重置环境
        obs = sim_env.reset()
        done = False
        total_reward = 0
        
        # 记录每一步的状态
        states = []
        
        # 运行模拟
        while not done:
            # 预测动作
            action, _ = self.model.predict(obs)
            
            # 执行动作
            obs, reward, done, info = sim_env.step(action)
            
            # 累计奖励
            total_reward += reward
            
            # 记录状态
            states.append(info)
        
        # 计算模拟结果
        final_balance = sim_env.balance
        final_asset_value = sim_env.asset_value
        total_value = sim_env.total_value
        total_return = (total_value / initial_balance) - 1
        
        # 记录交易详情
        trades = sim_env.trades
        
        # 保存模拟结果
        simulation_results = {
            'initial_balance': float(initial_balance),
            'final_balance': float(final_balance),
            'final_asset_value': float(final_asset_value),
            'total_value': float(total_value),
            'total_return': float(total_return),
            'total_reward': float(total_reward),
            'n_trades': len(trades),
            'trades': trades
        }
        
        # 保存模拟结果到文件
        with open('results/metrics/rl_simulation.json', 'w') as f:
            json.dump(simulation_results, f, indent=4)
        
        logger.info(f"交易模拟完成，总回报: {total_return:.2%}")
        
        # 绘制模拟结果
        sim_env.plot_performance(data['Close'])
        plt.savefig('results/figures/rl_simulation.png')
        plt.close()
        
        return simulation_results
    
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 10) -> Dict:
        """
        优化超参数
        
        Args:
            data: 股票数据
            n_trials: 尝试次数
            
        Returns:
            Dict: 最优超参数
        """
        # 注意：完整的超参数优化需要使用如Optuna这样的库
        # 这里简化处理，只尝试几种不同的参数组合
        
        logger.info(f"开始超参数优化，尝试次数: {n_trials}")
        
        # 创建基础环境
        base_env = self.create_environment(data, is_training=True)
        
        # 定义要尝试的参数
        learning_rates = [0.0001, 0.0003, 0.001]
        n_steps_options = [128, 256, 512, 1024, 2048]
        batch_sizes = [32, 64, 128]
        
        best_reward = -float('inf')
        best_params = None
        
        # 尝试不同参数组合
        trial = 0
        
        for lr in learning_rates:
            for n_steps in n_steps_options:
                for batch_size in batch_sizes:
                    if trial >= n_trials:
                        break
                    
                    trial += 1
                    
                    # 更新参数
                    params = {
                        'learning_rate': lr,
                        'n_steps': n_steps,
                        'batch_size': batch_size,
                        'gamma': 0.99,
                        'gae_lambda': 0.95
                    }
                    
                    # 创建模型
                    model = PPO(
                        policy='MlpPolicy',
                        env=base_env,
                        **params
                    )
                    
                    # 训练模型
                    model.learn(total_timesteps=10000)
                    
                    # 评估模型
                    mean_reward, _ = evaluate_policy(model, base_env, n_eval_episodes=5)
                    
                    logger.info(f"试验 {trial}/{n_trials}: lr={lr}, n_steps={n_steps}, batch_size={batch_size}, reward={mean_reward:.2f}")
                    
                    # 更新最优参数
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_params = params.copy()
        
        logger.info(f"超参数优化完成，最优参数: {best_params}, 最优奖励: {best_reward:.2f}")
        
        # 保存最优参数
        with open('results/metrics/rl_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        return best_params


def main():
    """主函数"""
    import os
    import glob
    import pandas as pd
    
    # 加载处理后的特征数据
    processed_data = {}
    if os.path.exists('data/processed/features'):
        for file in glob.glob('data/processed/features/*.csv'):
            symbol = os.path.basename(file).replace('_features.csv', '').replace('_', '.')
            data = pd.read_csv(file, index_col=0, parse_dates=True)
            processed_data[symbol] = data
    
    if not processed_data:
        logger.error("未找到处理后的特征数据，请先运行feature_engineering.py处理数据")
        return
    
    # 选择第一只股票用于示例
    symbol = list(processed_data.keys())[0]
    stock_data = processed_data[symbol]
    
    print(f"使用 {symbol} 的数据训练RL交易策略")
    
    # 创建RL交易者
    trader = RLTrader()
    
    # 训练模型
    trader.train(stock_data)
    
    # 评估模型
    evaluation_results = trader.evaluate(stock_data)
    
    # 打印评估结果
    print("\nRL交易策略评估结果:")
    print(f"  总收益率: {evaluation_results['total_return']:.2%}")
    print(f"  年化收益率: {evaluation_results['annual_return']:.2%}")
    print(f"  夏普比率: {evaluation_results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {evaluation_results['max_drawdown']:.2%}")
    print(f"  胜率: {evaluation_results['win_rate']:.2%}")
    print(f"  交易次数: {evaluation_results['n_trades']}")
    
    # 运行交易模拟
    simulation_results = trader.run_trading_simulation(stock_data)
    
    print("\n交易模拟结果:")
    print(f"  初始资金: ${simulation_results['initial_balance']:.2f}")
    print(f"  最终资金: ${simulation_results['final_balance']:.2f}")
    print(f"  最终资产价值: ${simulation_results['final_asset_value']:.2f}")
    print(f"  总价值: ${simulation_results['total_value']:.2f}")
    print(f"  总收益率: {simulation_results['total_return']:.2%}")
    print(f"  交易次数: {simulation_results['n_trades']}")


if __name__ == "__main__":
    main() 