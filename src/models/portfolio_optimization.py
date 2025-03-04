"""
资产配置模块

该模块负责根据机器学习模型的预测结果进行资产配置，使用现代投资组合理论进行投资组合优化。
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import cvxpy as cp

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置
from config.config import ALLOCATION_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """投资组合优化类，用于基于预测收益和风险进行资产配置"""
    
    def __init__(self, config: Dict = None):
        """
        初始化投资组合优化器
        
        Args:
            config: 配置字典，默认使用ALLOCATION_CONFIG
        """
        self.config = config or ALLOCATION_CONFIG
        self.predicted_returns = None
        self.covariance_matrix = None
        self.optimal_weights = None
        self.optimization_results = {}
        
        # 创建结果保存目录
        os.makedirs('results/metrics', exist_ok=True)
        
        logger.info("投资组合优化器初始化完成")
    
    def estimate_returns(self, historical_returns: pd.DataFrame, 
                        predicted_returns: pd.DataFrame = None,
                        alpha: float = 0.5) -> pd.Series:
        """
        估计预期收益率
        
        Args:
            historical_returns: 历史收益率DataFrame，索引为日期，列为资产
            predicted_returns: 预测收益率DataFrame，索引为日期，列为资产
            alpha: 预测收益率的权重，历史收益率的权重为1-alpha
            
        Returns:
            pd.Series: 预期收益率序列，索引为资产
        """
        # 计算历史平均收益率
        hist_mean_return = historical_returns.mean()
        
        # 如果没有提供预测收益率，则仅使用历史收益率
        if predicted_returns is None:
            logger.info("未提供预测收益率，使用历史平均收益率")
            return hist_mean_return
        
        # 计算预测收益率的平均值
        pred_mean_return = predicted_returns.mean()
        
        # 组合历史收益率和预测收益率
        expected_returns = (1 - alpha) * hist_mean_return + alpha * pred_mean_return
        
        return expected_returns
    
    def estimate_covariance(self, historical_returns: pd.DataFrame, 
                           time_decay: bool = True) -> pd.DataFrame:
        """
        估计协方差矩阵
        
        Args:
            historical_returns: 历史收益率DataFrame，索引为日期，列为资产
            time_decay: 是否使用时间衰减，最近的数据权重更高
            
        Returns:
            pd.DataFrame: 协方差矩阵
        """
        if time_decay:
            # 使用指数加权协方差矩阵，更近期的数据权重更高
            halflife = 21  # 半衰期为21天（约一个月）
            return historical_returns.ewm(halflife=halflife).cov().iloc[-len(historical_returns.columns):]
        else:
            # 使用普通协方差矩阵
            return historical_returns.cov()
    
    def prepare_optimization_data(self, stock_data: Dict[str, pd.DataFrame],
                                predictions: Dict[str, pd.DataFrame],
                                window_size: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
        """
        准备优化数据
        
        Args:
            stock_data: 股票数据字典，键为股票代码，值为价格数据
            predictions: 预测数据字典，键为股票代码，值为预测数据
            window_size: 用于计算历史统计量的窗口大小
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: (预期收益率, 协方差矩阵)
        """
        logger.info("准备投资组合优化数据")
        
        # 提取最近的价格和预测数据
        recent_prices = {}
        recent_predictions = {}
        
        for symbol, data in stock_data.items():
            # 确保数据按时间排序
            data = data.sort_index()
            
            # 仅使用最近的数据
            recent_data = data.iloc[-window_size:]
            
            if len(recent_data) > 0:
                recent_prices[symbol] = recent_data
            
            # 提取预测数据
            if symbol in predictions:
                pred_data = predictions[symbol].sort_index()
                if len(pred_data) > 0:
                    recent_predictions[symbol] = pred_data
        
        # 计算每日收益率
        returns_data = {}
        for symbol, data in recent_prices.items():
            if 'Close' in data.columns:
                returns_data[symbol] = data['Close'].pct_change().dropna()
        
        # 合并所有收益率数据
        historical_returns = pd.DataFrame(returns_data)
        
        # 检查是否有足够的数据
        if len(historical_returns) < window_size / 2:
            logger.warning(f"历史收益率数据不足，仅有 {len(historical_returns)} 条记录")
        
        # 合并所有预测数据
        predicted_returns_data = {}
        for symbol, data in recent_predictions.items():
            if 'predicted_return' in data.columns:
                predicted_returns_data[symbol] = data['predicted_return']
        
        predicted_returns = None
        if predicted_returns_data:
            predicted_returns = pd.DataFrame(predicted_returns_data)
        
        # 估计预期收益率
        alpha = 0.7  # 预测收益率的权重
        expected_returns = self.estimate_returns(historical_returns, predicted_returns, alpha)
        self.predicted_returns = expected_returns
        
        # 估计协方差矩阵
        covariance_matrix = self.estimate_covariance(historical_returns, time_decay=True)
        self.covariance_matrix = covariance_matrix
        
        logger.info(f"优化数据准备完成，资产数量: {len(expected_returns)}")
        
        return expected_returns, covariance_matrix
    
    def optimize_mean_variance(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame,
                             objective: str = 'sharpe_ratio', risk_free_rate: float = 0.0,
                             target_return: float = None, target_risk: float = None) -> pd.Series:
        """
        均值-方差优化
        
        Args:
            expected_returns: 预期收益率序列，索引为资产
            covariance_matrix: 协方差矩阵
            objective: 优化目标，'sharpe_ratio'、'min_variance'或'max_return'
            risk_free_rate: 无风险利率，用于计算夏普比率
            target_return: 目标收益率，用于风险最小化优化
            target_risk: 目标风险，用于收益率最大化优化
            
        Returns:
            pd.Series: 最优权重序列，索引为资产
        """
        logger.info(f"开始均值-方差优化，目标: {objective}")
        
        n_assets = len(expected_returns)
        
        # 定义目标函数和约束条件
        if objective == 'sharpe_ratio':
            # 最大化夏普比率
            def objective_function(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                # 最小化问题，所以返回负夏普比率
                return -sharpe_ratio
        elif objective == 'min_variance':
            # 最小化风险
            def objective_function(weights):
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                return portfolio_volatility
        elif objective == 'max_return':
            # 最大化收益
            def objective_function(weights):
                portfolio_return = np.sum(expected_returns * weights)
                # 最小化问题，所以返回负收益率
                return -portfolio_return
        else:
            logger.error(f"不支持的优化目标: {objective}")
            return pd.Series(index=expected_returns.index)
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1
        
        if target_return is not None and objective == 'min_variance':
            # 目标收益率约束
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(expected_returns * x) - target_return
            })
        
        if target_risk is not None and objective == 'max_return':
            # 目标风险约束
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(covariance_matrix, x))) - target_risk
            })
        
        # 权重范围约束
        bounds = []
        for asset in expected_returns.index:
            min_weight = self.config['optimization']['constraints'].get('min_weight', 0.0)
            max_weight = self.config['optimization']['constraints'].get('max_weight', 1.0)
            bounds.append((min_weight, max_weight))
        
        # 行业约束（在这里简化处理，实际应用中需要更复杂的处理）
        # 这部分代码在生产环境中需要根据实际情况进行调整
        
        # 初始权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            fun=objective_function,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            # 获取最优权重
            optimal_weights = pd.Series(result['x'], index=expected_returns.index)
            
            # 计算投资组合收益率和风险
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            logger.info(f"优化成功，投资组合预期年化收益率: {portfolio_return*252:.4f}, "
                      f"年化波动率: {portfolio_volatility*np.sqrt(252):.4f}, "
                      f"夏普比率: {sharpe_ratio*np.sqrt(252):.4f}")
            
            # 保存结果
            self.optimal_weights = optimal_weights
            self.optimization_results = {
                'objective': objective,
                'portfolio_return': portfolio_return * 252,  # 年化
                'portfolio_volatility': portfolio_volatility * np.sqrt(252),  # 年化
                'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # 年化
                'weights': optimal_weights.to_dict()
            }
            
            return optimal_weights
        else:
            logger.error(f"优化失败: {result['message']}")
            return pd.Series(initial_weights, index=expected_returns.index)
    
    def optimize_risk_parity(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        风险平价优化
        
        Args:
            covariance_matrix: 协方差矩阵
            
        Returns:
            pd.Series: 最优权重序列，索引为资产
        """
        logger.info("开始风险平价优化")
        
        n_assets = covariance_matrix.shape[0]
        
        # 定义风险贡献目标函数
        def risk_contribution_objective(weights):
            weights = np.array(weights).reshape(-1, 1)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))[0, 0]
            marginal_risk_contribution = np.dot(covariance_matrix, weights) / portfolio_volatility
            risk_contribution = np.multiply(marginal_risk_contribution, weights)
            target_risk_contribution = portfolio_volatility / n_assets
            return np.sum(np.square(risk_contribution - target_risk_contribution))
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1
        
        # 权重范围约束
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # 初始权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            fun=risk_contribution_objective,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            # 获取最优权重
            optimal_weights = pd.Series(result['x'], index=covariance_matrix.index)
            
            # 计算投资组合风险
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
            
            logger.info(f"优化成功，投资组合年化波动率: {portfolio_volatility*np.sqrt(252):.4f}")
            
            # 保存结果
            self.optimal_weights = optimal_weights
            self.optimization_results = {
                'objective': 'risk_parity',
                'portfolio_volatility': portfolio_volatility * np.sqrt(252),  # 年化
                'weights': optimal_weights.to_dict()
            }
            
            return optimal_weights
        else:
            logger.error(f"优化失败: {result['message']}")
            return pd.Series(initial_weights, index=covariance_matrix.index)
    
    def optimize_conditional_value_at_risk(self, returns: pd.DataFrame, 
                                         alpha: float = 0.05, 
                                         target_return: float = None) -> pd.Series:
        """
        条件风险价值（CVaR）优化
        
        Args:
            returns: 历史收益率DataFrame，索引为日期，列为资产
            alpha: 置信水平，默认0.05（95%置信度）
            target_return: 目标收益率
            
        Returns:
            pd.Series: 最优权重序列，索引为资产
        """
        logger.info(f"开始条件风险价值优化，置信水平: {1-alpha:.1%}")
        
        n_assets = returns.shape[1]
        n_samples = returns.shape[0]
        
        # 使用CVXPY进行优化
        # 变量
        w = cp.Variable(n_assets)
        a = cp.Variable()  # CVaR辅助变量
        u = cp.Variable(n_samples)  # 超额损失变量
        
        # 计算每个样本的投资组合收益率
        portfolio_returns = returns.values @ w
        
        # 约束条件
        constraints = [
            cp.sum(w) == 1,  # 权重和为1
            w >= 0,  # 非负权重
            u >= 0,  # 非负超额损失
        ]
        
        # CVaR约束
        for i in range(n_samples):
            constraints.append(-portfolio_returns[i] - a <= u[i])
        
        # 目标收益率约束
        if target_return is not None:
            constraints.append(cp.sum(cp.multiply(w, returns.mean().values)) >= target_return)
        
        # 目标函数：最小化CVaR
        objective = cp.Minimize(a + (1 / (alpha * n_samples)) * cp.sum(u))
        
        # 求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal' or problem.status == 'optimal_inaccurate':
            # 获取最优权重
            optimal_weights = pd.Series(w.value, index=returns.columns)
            
            # 计算投资组合预期收益率和CVaR
            portfolio_return = np.sum(returns.mean().values * optimal_weights)
            
            logger.info(f"优化成功，投资组合预期年化收益率: {portfolio_return*252:.4f}, "
                      f"CVaR: {a.value:.4f}")
            
            # 保存结果
            self.optimal_weights = optimal_weights
            self.optimization_results = {
                'objective': 'cvar',
                'portfolio_return': portfolio_return * 252,  # 年化
                'cvar': a.value,
                'weights': optimal_weights.to_dict()
            }
            
            return optimal_weights
        else:
            logger.error(f"优化失败: {problem.status}")
            # 返回等权重组合
            return pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)
    
    def optimize_portfolio(self, stock_data: Dict[str, pd.DataFrame],
                         predictions: Dict[str, pd.DataFrame] = None,
                         method: str = None,
                         risk_free_rate: float = 0.0,
                         target_return: float = None,
                         target_risk: float = None) -> pd.Series:
        """
        优化投资组合
        
        Args:
            stock_data: 股票数据字典，键为股票代码，值为价格数据
            predictions: 预测数据字典，键为股票代码，值为预测数据
            method: 优化方法，如果为None则使用配置文件中的方法
            risk_free_rate: 无风险利率
            target_return: 目标收益率
            target_risk: 目标风险
            
        Returns:
            pd.Series: 最优权重序列，索引为资产
        """
        # 使用配置中的方法
        if method is None:
            method = self.config['optimization']['method']
        
        # 准备优化数据
        expected_returns, covariance_matrix = self.prepare_optimization_data(stock_data, predictions or {})
        
        # 根据方法选择优化算法
        if method == 'mean_variance':
            objective = self.config['optimization']['objective']
            return self.optimize_mean_variance(
                expected_returns, 
                covariance_matrix,
                objective=objective,
                risk_free_rate=risk_free_rate,
                target_return=target_return,
                target_risk=target_risk
            )
        elif method == 'risk_parity':
            return self.optimize_risk_parity(covariance_matrix)
        elif method == 'cvar':
            # 计算每日收益率
            returns_data = {}
            for symbol, data in stock_data.items():
                if 'Close' in data.columns:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
            
            returns = pd.DataFrame(returns_data)
            
            return self.optimize_conditional_value_at_risk(
                returns,
                alpha=self.config['risk_management']['confidence_level'],
                target_return=target_return
            )
        else:
            logger.error(f"不支持的优化方法: {method}")
            # 返回等权重组合
            return pd.Series(np.ones(len(expected_returns)) / len(expected_returns), index=expected_returns.index)
    
    def plot_efficient_frontier(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, 
                              n_points: int = 100, risk_free_rate: float = 0.0) -> None:
        """
        绘制有效前沿
        
        Args:
            expected_returns: 预期收益率序列，索引为资产
            covariance_matrix: 协方差矩阵
            n_points: 有效前沿上的点数
            risk_free_rate: 无风险利率
        """
        logger.info("绘制有效前沿")
        
        # 计算最小方差组合
        min_vol_weights = self.optimize_mean_variance(
            expected_returns, 
            covariance_matrix,
            objective='min_variance'
        )
        min_vol_return = np.sum(expected_returns * min_vol_weights)
        min_vol_volatility = np.sqrt(np.dot(min_vol_weights.T, np.dot(covariance_matrix, min_vol_weights)))
        
        # 计算最大回报组合
        max_ret_weights = self.optimize_mean_variance(
            expected_returns, 
            covariance_matrix,
            objective='max_return'
        )
        max_ret_return = np.sum(expected_returns * max_ret_weights)
        max_ret_volatility = np.sqrt(np.dot(max_ret_weights.T, np.dot(covariance_matrix, max_ret_weights)))
        
        # 在最小方差和最大回报之间计算一系列目标回报
        target_returns = np.linspace(min_vol_return, max_ret_return, n_points)
        efficient_frontier_volatilities = []
        
        for target_return in target_returns:
            weights = self.optimize_mean_variance(
                expected_returns, 
                covariance_matrix,
                objective='min_variance',
                target_return=target_return
            )
            volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            efficient_frontier_volatilities.append(volatility)
        
        # 计算最优夏普比率组合
        sharpe_weights = self.optimize_mean_variance(
            expected_returns, 
            covariance_matrix,
            objective='sharpe_ratio',
            risk_free_rate=risk_free_rate
        )
        sharpe_return = np.sum(expected_returns * sharpe_weights)
        sharpe_volatility = np.sqrt(np.dot(sharpe_weights.T, np.dot(covariance_matrix, sharpe_weights)))
        
        # 绘制有效前沿
        plt.figure(figsize=(12, 8))
        
        # 有效前沿
        plt.plot(efficient_frontier_volatilities, target_returns, 'b-', linewidth=2, label='Efficient Frontier')
        
        # 最小方差组合
        plt.scatter(min_vol_volatility, min_vol_return, marker='o', color='g', s=100, label='Minimum Volatility')
        
        # 最大回报组合
        plt.scatter(max_ret_volatility, max_ret_return, marker='o', color='r', s=100, label='Maximum Return')
        
        # 最优夏普比率组合
        plt.scatter(sharpe_volatility, sharpe_return, marker='*', color='gold', s=200, label='Optimal Sharpe Ratio')
        
        # 显示个别资产
        for i, asset in enumerate(expected_returns.index):
            asset_volatility = np.sqrt(covariance_matrix.iloc[i, i])
            asset_return = expected_returns[asset]
            plt.scatter(asset_volatility, asset_return, marker='x', s=50, label=asset)
        
        # 资本市场线
        if risk_free_rate is not None:
            max_sharpe = (sharpe_return - risk_free_rate) / sharpe_volatility
            cml_x = np.linspace(0, max(max_ret_volatility, sharpe_volatility) * 1.2, 100)
            cml_y = risk_free_rate + max_sharpe * cml_x
            plt.plot(cml_x, cml_y, 'r--', label='Capital Market Line')
        
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/efficient_frontier.png')
        plt.close()
        
        logger.info("有效前沿图表已保存")
    
    def plot_optimal_allocation(self) -> None:
        """绘制最优资产配置饼图"""
        if self.optimal_weights is None:
            logger.error("未找到最优权重，无法绘制饼图")
            return
        
        # 过滤掉权重接近于0的资产
        threshold = 0.01  # 1%阈值
        significant_weights = self.optimal_weights[self.optimal_weights > threshold]
        
        if len(significant_weights) == 0:
            logger.warning("没有显著权重的资产，无法绘制饼图")
            return
        
        # 如果有剩余的小权重，将它们合并为"其他"
        if len(significant_weights) < len(self.optimal_weights):
            others_weight = 1.0 - significant_weights.sum()
            if others_weight > 0:
                significant_weights['Others'] = others_weight
        
        # 绘制饼图
        plt.figure(figsize=(10, 8))
        plt.pie(significant_weights, labels=significant_weights.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # 确保饼图是圆的
        plt.title('Optimal Portfolio Allocation')
        
        # 保存图表
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/optimal_allocation.png')
        plt.close()
        
        logger.info("最优资产配置饼图已保存")
    
    def save_optimization_results(self, filepath: str = 'results/metrics/portfolio_optimization.json') -> None:
        """
        保存优化结果
        
        Args:
            filepath: 保存路径
        """
        if not self.optimization_results:
            logger.error("未找到优化结果，无法保存")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 添加时间戳
        results = self.optimization_results.copy()
        results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"优化结果已保存到 {filepath}")
    
    def load_optimization_results(self, filepath: str = 'results/metrics/portfolio_optimization.json') -> Dict:
        """
        加载优化结果
        
        Args:
            filepath: 加载路径
            
        Returns:
            Dict: 优化结果字典
        """
        if not os.path.exists(filepath):
            logger.error(f"未找到优化结果文件: {filepath}")
            return {}
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.optimization_results = results
        
        # 重建最优权重
        if 'weights' in results:
            self.optimal_weights = pd.Series(results['weights'])
        
        logger.info(f"已加载优化结果，优化方法: {results.get('objective', 'unknown')}")
        
        return results
    
    def rebalance_portfolio(self, current_weights: pd.Series, target_weights: pd.Series, 
                          threshold: float = 0.05) -> pd.Series:
        """
        投资组合再平衡
        
        Args:
            current_weights: 当前权重序列，索引为资产
            target_weights: 目标权重序列，索引为资产
            threshold: 再平衡阈值，当权重差异超过该阈值时进行调整
            
        Returns:
            pd.Series: 再平衡后的权重序列，索引为资产
        """
        logger.info(f"开始投资组合再平衡，阈值: {threshold:.1%}")
        
        # 确保索引一致
        common_assets = current_weights.index.intersection(target_weights.index)
        current = current_weights[common_assets]
        target = target_weights[common_assets]
        
        # 计算权重差异
        weight_diff = (target - current).abs()
        
        # 找出需要调整的资产
        assets_to_adjust = weight_diff[weight_diff > threshold].index
        
        if len(assets_to_adjust) == 0:
            logger.info("没有资产需要调整")
            return current_weights
        
        # 计算调整后的权重
        new_weights = current_weights.copy()
        for asset in assets_to_adjust:
            new_weights[asset] = target[asset]
        
        # 归一化权重
        new_weights = new_weights / new_weights.sum()
        
        logger.info(f"再平衡完成，调整了 {len(assets_to_adjust)} 个资产的权重")
        
        return new_weights


def main():
    """主函数"""
    import os
    import glob
    import pandas as pd
    import numpy as np
    
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
    
    # 创建模拟预测数据
    predictions = {}
    for symbol, data in processed_data.items():
        if 'future_return_5d' in data.columns:
            # 使用实际未来收益率加上一些噪声作为"预测"
            noise = np.random.normal(0, 0.002, len(data))
            pred_return = data['future_return_5d'] + noise
            
            # 创建预测DataFrame
            pred_df = pd.DataFrame({
                'predicted_return': pred_return
            }, index=data.index)
            
            predictions[symbol] = pred_df
    
    # 提取股票价格数据
    stock_data = {}
    for symbol, data in processed_data.items():
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            # 提取OHLCV数据
            price_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            stock_data[symbol] = price_data
    
    # 创建投资组合优化器
    optimizer = PortfolioOptimizer()
    
    # 优化投资组合
    optimal_weights = optimizer.optimize_portfolio(
        stock_data,
        predictions,
        method='mean_variance',
        risk_free_rate=0.02 / 252  # 假设年化无风险利率为2%
    )
    
    # 打印最优权重
    print("\n最优投资组合权重:")
    for asset, weight in optimal_weights.sort_values(ascending=False).items():
        if weight > 0.01:  # 只显示权重大于1%的资产
            print(f"  {asset}: {weight:.2%}")
    
    # 绘制有效前沿
    optimizer.plot_efficient_frontier(
        optimizer.predicted_returns,
        optimizer.covariance_matrix,
        risk_free_rate=0.02 / 252
    )
    
    # 绘制最优资产配置饼图
    optimizer.plot_optimal_allocation()
    
    # 保存优化结果
    optimizer.save_optimization_results()
    
    print("\n投资组合优化完成，结果已保存")


if __name__ == "__main__":
    main() 