"""
特征工程模块

该模块负责计算股票的技术指标和因子，为机器学习模型提供输入特征。
"""

import os
import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List, Tuple, Optional, Union
import sys
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置
from config.config import FACTOR_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类，用于计算股票的技术指标和因子"""
    
    def __init__(self, config: Dict = None):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典，默认使用FACTOR_CONFIG
        """
        self.config = config or FACTOR_CONFIG
        
        # 创建数据目录
        os.makedirs('data/processed/features', exist_ok=True)
        
        logger.info("特征工程器初始化完成")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 股票价格数据，包含Open, High, Low, Close, Volume列
            
        Returns:
            pd.DataFrame: 添加了技术指标的数据框
        """
        # 确保数据包含必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"数据缺少必要的列: {required_columns}")
            return data
        
        # 创建结果数据框
        result = data.copy()
        
        # 计算动量指标
        if 'momentum_1m' in self.config['technical_factors']:
            result['momentum_1m'] = result['Close'].pct_change(21)  # 约21个交易日
        
        if 'momentum_3m' in self.config['technical_factors']:
            result['momentum_3m'] = result['Close'].pct_change(63)  # 约63个交易日
        
        if 'momentum_6m' in self.config['technical_factors']:
            result['momentum_6m'] = result['Close'].pct_change(126)  # 约126个交易日
        
        # 计算波动率指标
        if 'volatility_1m' in self.config['technical_factors']:
            result['volatility_1m'] = result['Close'].pct_change().rolling(21).std()
        
        if 'volatility_3m' in self.config['technical_factors']:
            result['volatility_3m'] = result['Close'].pct_change().rolling(63).std()
        
        # 计算RSI
        if 'rsi_14' in self.config['technical_factors']:
            result['rsi_14'] = talib.RSI(result['Close'].values, timeperiod=14)
        
        # 计算MACD
        if 'macd' in self.config['technical_factors']:
            macd, macd_signal, macd_hist = talib.MACD(
                result['Close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            result['macd'] = macd
            result['macd_signal'] = macd_signal
            result['macd_hist'] = macd_hist
        
        # 计算布林带
        if 'bollinger_band' in self.config['technical_factors']:
            upper, middle, lower = talib.BBANDS(
                result['Close'].values, 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower
            # 计算价格相对于布林带的位置
            result['bb_position'] = (result['Close'] - lower) / (upper - lower)
        
        # 计算交易量相关指标
        if 'volume_ratio' in self.config['technical_factors']:
            result['volume_ratio'] = result['Volume'] / result['Volume'].rolling(20).mean()
        
        if 'turnover_rate' in self.config['technical_factors']:
            # 注意：实际计算换手率需要流通股本数据，这里简化处理
            result['turnover_rate'] = result['Volume'] / result['Volume'].rolling(252).mean()
        
        # 添加更多技术指标
        # 移动平均线
        result['sma_5'] = talib.SMA(result['Close'].values, timeperiod=5)
        result['sma_10'] = talib.SMA(result['Close'].values, timeperiod=10)
        result['sma_20'] = talib.SMA(result['Close'].values, timeperiod=20)
        result['sma_60'] = talib.SMA(result['Close'].values, timeperiod=60)
        
        # 指数移动平均线
        result['ema_5'] = talib.EMA(result['Close'].values, timeperiod=5)
        result['ema_10'] = talib.EMA(result['Close'].values, timeperiod=10)
        result['ema_20'] = talib.EMA(result['Close'].values, timeperiod=20)
        result['ema_60'] = talib.EMA(result['Close'].values, timeperiod=60)
        
        # 计算均线差值和交叉信号
        result['sma_5_10_diff'] = result['sma_5'] - result['sma_10']
        result['sma_10_20_diff'] = result['sma_10'] - result['sma_20']
        result['sma_20_60_diff'] = result['sma_20'] - result['sma_60']
        
        # 计算KDJ指标
        result['k'], result['d'] = talib.STOCH(
            result['High'].values, 
            result['Low'].values, 
            result['Close'].values, 
            fastk_period=9, 
            slowk_period=3, 
            slowk_matype=0, 
            slowd_period=3, 
            slowd_matype=0
        )
        result['j'] = 3 * result['k'] - 2 * result['d']
        
        # 计算ATR（真实波动幅度均值）
        result['atr'] = talib.ATR(
            result['High'].values, 
            result['Low'].values, 
            result['Close'].values, 
            timeperiod=14
        )
        
        # 计算CCI（顺势指标）
        result['cci'] = talib.CCI(
            result['High'].values, 
            result['Low'].values, 
            result['Close'].values, 
            timeperiod=14
        )
        
        # 计算OBV（能量潮）
        result['obv'] = talib.OBV(result['Close'].values, result['Volume'].values)
        
        # 计算ADX（平均趋向指数）
        result['adx'] = talib.ADX(
            result['High'].values, 
            result['Low'].values, 
            result['Close'].values, 
            timeperiod=14
        )
        
        logger.info(f"已计算 {len(result.columns) - len(data.columns)} 个技术指标")
        
        return result
    
    def calculate_fundamental_factors(self, data: pd.DataFrame, 
                                     financial_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算基本面因子
        
        Args:
            data: 股票价格数据
            financial_data: 财务数据，如果为None，则使用模拟数据
            
        Returns:
            pd.DataFrame: 添加了基本面因子的数据框
        """
        # 创建结果数据框
        result = data.copy()
        
        # 如果没有提供财务数据，则使用模拟数据
        if financial_data is None:
            logger.warning("未提供财务数据，使用模拟数据")
            
            # 模拟市盈率 (P/E)
            if 'pe_ratio' in self.config['fundamental_factors']:
                # 模拟一个在10-30之间波动的市盈率
                result['pe_ratio'] = np.random.uniform(10, 30, len(result))
                # 添加一些趋势和季节性
                t = np.arange(len(result)) / 252  # 转换为年
                result['pe_ratio'] = result['pe_ratio'] + 5 * np.sin(2 * np.pi * t) + t
            
            # 模拟市净率 (P/B)
            if 'pb_ratio' in self.config['fundamental_factors']:
                # 模拟一个在1-5之间波动的市净率
                result['pb_ratio'] = np.random.uniform(1, 5, len(result))
                # 添加一些趋势
                result['pb_ratio'] = result['pb_ratio'] + t * 0.5
            
            # 模拟市销率 (P/S)
            if 'ps_ratio' in self.config['fundamental_factors']:
                # 模拟一个在2-8之间波动的市销率
                result['ps_ratio'] = np.random.uniform(2, 8, len(result))
                # 添加一些趋势和季节性
                result['ps_ratio'] = result['ps_ratio'] + 2 * np.sin(2 * np.pi * t * 0.5) + t * 0.3
            
            # 模拟ROE
            if 'roe' in self.config['fundamental_factors']:
                # 模拟一个在0.05-0.25之间波动的ROE
                result['roe'] = np.random.uniform(0.05, 0.25, len(result))
                # 添加季度性变化（财报发布）
                quarter = (np.arange(len(result)) % 63) / 63  # 假设每季度63个交易日
                result['roe'] = result['roe'] + 0.05 * np.sin(2 * np.pi * quarter)
            
            # 模拟ROA
            if 'roa' in self.config['fundamental_factors']:
                # 模拟一个在0.02-0.15之间波动的ROA
                result['roa'] = np.random.uniform(0.02, 0.15, len(result))
                # 添加季度性变化
                result['roa'] = result['roa'] + 0.03 * np.sin(2 * np.pi * quarter)
            
            # 模拟毛利率
            if 'gross_margin' in self.config['fundamental_factors']:
                # 模拟一个在0.3-0.6之间波动的毛利率
                result['gross_margin'] = np.random.uniform(0.3, 0.6, len(result))
                # 添加季度性变化
                result['gross_margin'] = result['gross_margin'] + 0.05 * np.sin(2 * np.pi * quarter)
            
            # 模拟负债权益比
            if 'debt_to_equity' in self.config['fundamental_factors']:
                # 模拟一个在0.5-2.0之间波动的负债权益比
                result['debt_to_equity'] = np.random.uniform(0.5, 2.0, len(result))
                # 添加一些趋势
                result['debt_to_equity'] = result['debt_to_equity'] - t * 0.1  # 假设负债在减少
            
            # 模拟流动比率
            if 'current_ratio' in self.config['fundamental_factors']:
                # 模拟一个在1.2-3.0之间波动的流动比率
                result['current_ratio'] = np.random.uniform(1.2, 3.0, len(result))
                # 添加一些趋势
                result['current_ratio'] = result['current_ratio'] + t * 0.2  # 假设流动性在提高
            
            # 模拟收入增长率
            if 'revenue_growth' in self.config['fundamental_factors']:
                # 模拟一个在0.05-0.3之间波动的收入增长率
                result['revenue_growth'] = np.random.uniform(0.05, 0.3, len(result))
                # 添加季度性变化和长期趋势
                result['revenue_growth'] = result['revenue_growth'] + 0.1 * np.sin(2 * np.pi * quarter) - t * 0.02
            
            # 模拟利润增长率
            if 'profit_growth' in self.config['fundamental_factors']:
                # 模拟一个在0.03-0.25之间波动的利润增长率
                result['profit_growth'] = np.random.uniform(0.03, 0.25, len(result))
                # 添加季度性变化和长期趋势
                result['profit_growth'] = result['profit_growth'] + 0.08 * np.sin(2 * np.pi * quarter) - t * 0.015
            
            # 模拟研发投入强度
            if 'r_and_d_intensity' in self.config['fundamental_factors']:
                # 模拟一个在0.08-0.2之间波动的研发投入强度（研发费用/收入）
                result['r_and_d_intensity'] = np.random.uniform(0.08, 0.2, len(result))
                # 添加季度性变化和长期趋势
                result['r_and_d_intensity'] = result['r_and_d_intensity'] + 0.02 * np.sin(2 * np.pi * quarter) + t * 0.01
        else:
            # 使用实际财务数据计算因子
            # 这部分需要根据实际财务数据的格式进行调整
            logger.info("使用实际财务数据计算基本面因子")
            # 示例代码，需要根据实际数据调整
            # result = pd.merge(result, financial_data, left_index=True, right_index=True, how='left')
        
        logger.info(f"已计算 {len(self.config['fundamental_factors'])} 个基本面因子")
        
        return result
    
    def calculate_industry_specific_factors(self, data: pd.DataFrame, 
                                          industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算行业特定因子
        
        Args:
            data: 股票价格数据
            industry_data: 行业数据，如果为None，则使用模拟数据
            
        Returns:
            pd.DataFrame: 添加了行业特定因子的数据框
        """
        # 创建结果数据框
        result = data.copy()
        
        # 如果没有提供行业数据，则使用模拟数据
        if industry_data is None:
            logger.warning("未提供行业数据，使用模拟数据")
            
            # 时间变量
            t = np.arange(len(result)) / 252  # 转换为年
            
            # 模拟市场份额
            if 'market_share' in self.config['industry_specific_factors']:
                # 模拟一个在0.05-0.3之间的市场份额
                result['market_share'] = np.random.uniform(0.05, 0.3, 1)[0]
                # 添加一些小的随机波动和趋势
                result['market_share'] = result['market_share'] + np.random.normal(0, 0.01, len(result)) + t * 0.01
                # 确保值在合理范围内
                result['market_share'] = result['market_share'].clip(0.01, 0.5)
            
            # 模拟产品周期
            if 'product_cycle' in self.config['industry_specific_factors']:
                # 模拟产品周期位置（0-1之间的值，表示在产品周期中的位置）
                cycle_length = 2  # 假设产品周期为2年
                result['product_cycle'] = (t % cycle_length) / cycle_length
                # 添加一些随机波动
                result['product_cycle'] = result['product_cycle'] + np.random.normal(0, 0.05, len(result))
                # 确保值在0-1之间
                result['product_cycle'] = result['product_cycle'] % 1
            
            # 模拟技术领先度
            if 'tech_leadership' in self.config['industry_specific_factors']:
                # 模拟一个在0.6-0.9之间的技术领先度
                result['tech_leadership'] = np.random.uniform(0.6, 0.9, 1)[0]
                # 添加一些小的随机波动和趋势
                result['tech_leadership'] = result['tech_leadership'] + np.random.normal(0, 0.02, len(result)) + t * 0.02
                # 确保值在0-1之间
                result['tech_leadership'] = result['tech_leadership'].clip(0, 1)
            
            # 模拟客户集中度
            if 'customer_concentration' in self.config['industry_specific_factors']:
                # 模拟一个在0.2-0.7之间的客户集中度
                result['customer_concentration'] = np.random.uniform(0.2, 0.7, 1)[0]
                # 添加一些小的随机波动和缓慢变化
                result['customer_concentration'] = result['customer_concentration'] + np.random.normal(0, 0.01, len(result)) - t * 0.01
                # 确保值在0-1之间
                result['customer_concentration'] = result['customer_concentration'].clip(0, 1)
        else:
            # 使用实际行业数据计算因子
            logger.info("使用实际行业数据计算行业特定因子")
            # 示例代码，需要根据实际数据调整
            # result = pd.merge(result, industry_data, left_index=True, right_index=True, how='left')
        
        logger.info(f"已计算 {len(self.config['industry_specific_factors'])} 个行业特定因子")
        
        return result
    
    def calculate_sentiment_factors(self, data: pd.DataFrame, 
                                   sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算情感因子
        
        Args:
            data: 股票价格数据
            sentiment_data: 情感数据，如果为None，则使用模拟数据
            
        Returns:
            pd.DataFrame: 添加了情感因子的数据框
        """
        # 创建结果数据框
        result = data.copy()
        
        # 如果没有提供情感数据，则使用模拟数据
        if sentiment_data is None:
            logger.warning("未提供情感数据，使用模拟数据")
            
            # 模拟新闻情感
            if 'news_sentiment' in self.config['sentiment_factors']:
                # 生成一个在-1到1之间的随机情感得分，均值为0.1（略微正面）
                result['news_sentiment'] = np.random.normal(0.1, 0.3, len(result))
                # 添加一些自相关性（情感通常会持续一段时间）
                for i in range(1, len(result)):
                    result.iloc[i, result.columns.get_loc('news_sentiment')] = \
                        0.7 * result.iloc[i-1, result.columns.get_loc('news_sentiment')] + \
                        0.3 * result.iloc[i, result.columns.get_loc('news_sentiment')]
                # 确保值在-1到1之间
                result['news_sentiment'] = result['news_sentiment'].clip(-1, 1)
            
            # 模拟分析师评级
            if 'analyst_ratings' in self.config['sentiment_factors']:
                # 生成一个在1到5之间的随机评级，均值为3.5
                result['analyst_ratings'] = np.random.normal(3.5, 0.5, len(result))
                # 添加一些自相关性（评级通常会持续一段时间）
                for i in range(1, len(result)):
                    result.iloc[i, result.columns.get_loc('analyst_ratings')] = \
                        0.9 * result.iloc[i-1, result.columns.get_loc('analyst_ratings')] + \
                        0.1 * result.iloc[i, result.columns.get_loc('analyst_ratings')]
                # 确保值在1到5之间
                result['analyst_ratings'] = result['analyst_ratings'].clip(1, 5)
            
            # 模拟社交媒体情感
            if 'social_media_sentiment' in self.config['sentiment_factors']:
                # 生成一个在-1到1之间的随机情感得分，均值为0（中性）
                result['social_media_sentiment'] = np.random.normal(0, 0.4, len(result))
                # 添加一些自相关性和更多的波动性（社交媒体情感变化更快）
                for i in range(1, len(result)):
                    result.iloc[i, result.columns.get_loc('social_media_sentiment')] = \
                        0.5 * result.iloc[i-1, result.columns.get_loc('social_media_sentiment')] + \
                        0.5 * result.iloc[i, result.columns.get_loc('social_media_sentiment')]
                # 确保值在-1到1之间
                result['social_media_sentiment'] = result['social_media_sentiment'].clip(-1, 1)
        else:
            # 使用实际情感数据计算因子
            logger.info("使用实际情感数据计算情感因子")
            # 示例代码，需要根据实际数据调整
            # result = pd.merge(result, sentiment_data, left_index=True, right_index=True, how='left')
        
        logger.info(f"已计算 {len(self.config['sentiment_factors'])} 个情感因子")
        
        return result
    
    def calculate_target_variable(self, data: pd.DataFrame, horizon: str = '5d') -> pd.DataFrame:
        """
        计算目标变量（未来收益率）
        
        Args:
            data: 股票价格数据
            horizon: 预测周期，如'1d'表示1天，'5d'表示5天，'21d'表示21天（约1个月）
            
        Returns:
            pd.DataFrame: 添加了目标变量的数据框
        """
        # 创建结果数据框
        result = data.copy()
        
        # 解析预测周期
        if horizon.endswith('d'):
            days = int(horizon[:-1])
        elif horizon.endswith('w'):
            days = int(horizon[:-1]) * 5  # 假设每周5个交易日
        elif horizon.endswith('m'):
            days = int(horizon[:-1]) * 21  # 假设每月21个交易日
        else:
            logger.error(f"不支持的预测周期格式: {horizon}")
            days = 5  # 默认为5天
        
        # 计算未来收益率
        result[f'future_return_{horizon}'] = result['Close'].pct_change(days).shift(-days)
        
        # 计算未来收益率的方向（上涨/下跌）
        result[f'future_direction_{horizon}'] = (result[f'future_return_{horizon}'] > 0).astype(int)
        
        logger.info(f"已计算目标变量: 未来{horizon}收益率和方向")
        
        return result
    
    def process_stock_data(self, stock_data: Dict[str, pd.DataFrame], 
                          include_fundamental: bool = True,
                          include_industry: bool = True,
                          include_sentiment: bool = True,
                          target_horizon: str = '5d') -> Dict[str, pd.DataFrame]:
        """
        处理所有股票数据，计算特征和目标变量
        
        Args:
            stock_data: 股票数据字典，键为股票代码，值为价格数据框
            include_fundamental: 是否包含基本面因子
            include_industry: 是否包含行业特定因子
            include_sentiment: 是否包含情感因子
            target_horizon: 目标变量的预测周期
            
        Returns:
            Dict[str, pd.DataFrame]: 处理后的股票数据字典
        """
        processed_data = {}
        
        logger.info(f"开始处理 {len(stock_data)} 只股票的数据")
        
        for symbol, data in tqdm(stock_data.items(), desc="处理股票数据"):
            try:
                # 计算技术指标
                result = self.calculate_technical_indicators(data)
                
                # 计算基本面因子
                if include_fundamental:
                    result = self.calculate_fundamental_factors(result)
                
                # 计算行业特定因子
                if include_industry:
                    result = self.calculate_industry_specific_factors(result)
                
                # 计算情感因子
                if include_sentiment:
                    result = self.calculate_sentiment_factors(result)
                
                # 计算目标变量
                result = self.calculate_target_variable(result, horizon=target_horizon)
                
                # 删除包含NaN的行
                result = result.dropna()
                
                # 保存处理后的数据
                processed_data[symbol] = result
                
                # 保存到CSV
                os.makedirs('data/processed/features', exist_ok=True)
                result.to_csv(f'data/processed/features/{symbol.replace(".", "_")}_features.csv')
                
                logger.info(f"已处理 {symbol} 的数据，共 {len(result)} 条记录，{len(result.columns)} 个特征")
            
            except Exception as e:
                logger.error(f"处理 {symbol} 的数据时出错: {str(e)}")
        
        logger.info(f"数据处理完成，共处理 {len(processed_data)} 只股票的数据")
        
        return processed_data


def main():
    """主函数"""
    import os
    import glob
    
    # 加载原始数据
    raw_data = {}
    for market in ['US', 'HK']:
        if os.path.exists(f'data/raw/{market}'):
            for file in glob.glob(f'data/raw/{market}/*.csv'):
                symbol = os.path.basename(file).replace('_', '.').replace('.csv', '')
                data = pd.read_csv(file, index_col=0, parse_dates=True)
                raw_data[symbol] = data
    
    if not raw_data:
        logger.error("未找到原始数据，请先运行stock_selection.py获取数据")
        return
    
    # 创建特征工程器
    engineer = FeatureEngineer()
    
    # 处理股票数据
    processed_data = engineer.process_stock_data(
        raw_data,
        include_fundamental=True,
        include_industry=True,
        include_sentiment=True,
        target_horizon='5d'
    )
    
    print(f"共处理 {len(processed_data)} 只股票的数据")
    
    # 打印第一只股票的特征
    if processed_data:
        symbol = list(processed_data.keys())[0]
        print(f"\n{symbol} 的特征示例:")
        print(processed_data[symbol].head())
        print(f"\n特征列表: {processed_data[symbol].columns.tolist()}")


if __name__ == "__main__":
    main() 