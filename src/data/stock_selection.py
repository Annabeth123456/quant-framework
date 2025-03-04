"""
股票选择模块

该模块负责选择AI、半导体和芯片行业的代表性公司，涵盖上中下游各环节。
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import logging
from typing import Dict, List, Tuple, Optional
import sys
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置
from config.config import DATA_CONFIG, FACTOR_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockSelector:
    """股票选择器类，用于选择AI、半导体和芯片行业的代表性公司"""
    
    def __init__(self, config: Dict = None):
        """
        初始化股票选择器
        
        Args:
            config: 配置字典，默认使用DATA_CONFIG
        """
        self.config = config or DATA_CONFIG
        self.selected_stocks = {
            'US': {segment: [] for segment in self.config['industry_segments']},
            'HK': {segment: [] for segment in self.config['industry_segments']}
        }
        self.stock_data = {}
        
        # 创建数据目录
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        logger.info("股票选择器初始化完成")
    
    def get_predefined_stocks(self) -> Dict:
        """
        获取预定义的股票列表
        
        由于实际获取完整的行业股票需要付费数据源，这里提供一个预定义的列表作为示例
        
        Returns:
            Dict: 预定义的股票字典，按市场和行业分类
        """
        predefined_stocks = {
            'US': {
                'upstream': [
                    {'symbol': 'AMAT', 'name': 'Applied Materials', 'segment': 'equipment'},
                    {'symbol': 'LRCX', 'name': 'Lam Research', 'segment': 'equipment'},
                    {'symbol': 'KLAC', 'name': 'KLA Corporation', 'segment': 'equipment'},
                    {'symbol': 'ASML', 'name': 'ASML Holding', 'segment': 'equipment'},
                    {'symbol': 'EMR', 'name': 'Emerson Electric', 'segment': 'equipment'},
                ],
                'midstream': [
                    {'symbol': 'TSM', 'name': 'Taiwan Semiconductor', 'segment': 'foundry'},
                    {'symbol': 'UMC', 'name': 'United Microelectronics', 'segment': 'foundry'},
                    {'symbol': 'INTC', 'name': 'Intel Corporation', 'segment': 'IDM'},
                    {'symbol': 'TXN', 'name': 'Texas Instruments', 'segment': 'IDM'},
                    {'symbol': 'MU', 'name': 'Micron Technology', 'segment': 'memory'},
                ],
                'downstream': [
                    {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'segment': 'chip_design'},
                    {'symbol': 'AMD', 'name': 'Advanced Micro Devices', 'segment': 'chip_design'},
                    {'symbol': 'QCOM', 'name': 'Qualcomm', 'segment': 'chip_design'},
                    {'symbol': 'AVGO', 'name': 'Broadcom', 'segment': 'chip_design'},
                    {'symbol': 'MRVL', 'name': 'Marvell Technology', 'segment': 'chip_design'},
                    {'symbol': 'GOOGL', 'name': 'Alphabet', 'segment': 'AI_application'},
                    {'symbol': 'MSFT', 'name': 'Microsoft', 'segment': 'AI_application'},
                ]
            },
            'HK': {
                'upstream': [
                    {'symbol': '00522.HK', 'name': 'ASM Pacific Technology', 'segment': 'equipment'},
                ],
                'midstream': [
                    {'symbol': '00981.HK', 'name': 'SMIC', 'segment': 'foundry'},
                    {'symbol': '01347.HK', 'name': 'Hua Hong Semiconductor', 'segment': 'foundry'},
                ],
                'downstream': [
                    {'symbol': '09618.HK', 'name': 'JD.com', 'segment': 'AI_application'},
                    {'symbol': '09988.HK', 'name': 'Alibaba', 'segment': 'AI_application'},
                    {'symbol': '00700.HK', 'name': 'Tencent', 'segment': 'AI_application'},
                    {'symbol': '09999.HK', 'name': 'NetEase', 'segment': 'AI_application'},
                ]
            }
        }
        
        return predefined_stocks
    
    def select_stocks_by_market_cap(self, market: str, segment: str, top_n: int = 5) -> List[Dict]:
        """
        根据市值选择股票
        
        Args:
            market: 市场，'US'或'HK'
            segment: 行业段，'upstream'、'midstream'或'downstream'
            top_n: 选择前n个股票
            
        Returns:
            List[Dict]: 选择的股票列表
        """
        # 在实际应用中，这里应该从数据源获取股票列表并按市值排序
        # 由于这需要付费数据源，这里使用预定义的列表作为示例
        predefined_stocks = self.get_predefined_stocks()
        
        if market in predefined_stocks and segment in predefined_stocks[market]:
            # 模拟按市值排序
            stocks = predefined_stocks[market][segment]
            # 实际应用中应该获取真实市值数据
            return stocks[:min(top_n, len(stocks))]
        else:
            logger.warning(f"未找到市场 {market} 的 {segment} 段股票")
            return []
    
    def select_stocks_by_factor(self, market: str, segment: str, 
                               factors: List[str], top_n: int = 5) -> List[Dict]:
        """
        根据因子选择股票
        
        Args:
            market: 市场，'US'或'HK'
            segment: 行业段，'upstream'、'midstream'或'downstream'
            factors: 因子列表
            top_n: 选择前n个股票
            
        Returns:
            List[Dict]: 选择的股票列表
        """
        # 在实际应用中，这里应该计算因子值并排序
        # 由于这需要详细的财务和市场数据，这里使用预定义的列表作为示例
        return self.select_stocks_by_market_cap(market, segment, top_n)
    
    def select_representative_stocks(self, method: str = 'market_cap', 
                                    top_n: int = 5) -> Dict:
        """
        选择代表性股票
        
        Args:
            method: 选择方法，'market_cap'或'factor'
            top_n: 每个细分行业选择的股票数量
            
        Returns:
            Dict: 选择的股票字典，按市场和行业分类
        """
        logger.info(f"开始选择代表性股票，方法: {method}, 每个细分行业选择 {top_n} 只股票")
        
        for market in self.config['markets']:
            for segment in self.config['industry_segments']:
                if method == 'market_cap':
                    selected = self.select_stocks_by_market_cap(market, segment, top_n)
                elif method == 'factor':
                    # 使用基本面因子选择股票
                    selected = self.select_stocks_by_factor(
                        market, segment, 
                        FACTOR_CONFIG['fundamental_factors'], 
                        top_n
                    )
                else:
                    logger.error(f"不支持的选股方法: {method}")
                    continue
                
                self.selected_stocks[market][segment].extend(selected)
                
                logger.info(f"已选择 {market} 市场 {segment} 段的 {len(selected)} 只股票")
        
        # 保存选择的股票
        self.save_selected_stocks()
        
        return self.selected_stocks
    
    def save_selected_stocks(self, filepath: str = 'data/processed/selected_stocks.json'):
        """
        保存选择的股票
        
        Args:
            filepath: 保存路径
        """
        with open(filepath, 'w') as f:
            json.dump(self.selected_stocks, f, indent=4)
        
        logger.info(f"已保存选择的股票到 {filepath}")
    
    def load_selected_stocks(self, filepath: str = 'data/processed/selected_stocks.json') -> Dict:
        """
        加载选择的股票
        
        Args:
            filepath: 加载路径
            
        Returns:
            Dict: 选择的股票字典
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.selected_stocks = json.load(f)
            
            logger.info(f"已加载选择的股票从 {filepath}")
            return self.selected_stocks
        else:
            logger.warning(f"文件 {filepath} 不存在，无法加载选择的股票")
            return self.selected_stocks
    
    def fetch_stock_data(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        获取选择的股票的历史数据
        
        Args:
            start_date: 开始日期，默认使用配置中的start_date
            end_date: 结束日期，默认使用配置中的end_date
            
        Returns:
            Dict: 股票历史数据字典
        """
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']
        
        logger.info(f"开始获取股票历史数据，时间范围: {start_date} 到 {end_date}")
        
        # 确保已选择股票
        if not any(any(segment) for market in self.selected_stocks.values() for segment in market.values()):
            logger.warning("未选择任何股票，请先调用select_representative_stocks方法")
            return {}
        
        # 获取所有股票的历史数据
        for market in self.config['markets']:
            for segment in self.config['industry_segments']:
                for stock in self.selected_stocks[market][segment]:
                    symbol = stock['symbol']
                    
                    try:
                        if market == 'US':
                            # 使用yfinance获取美股数据
                            data = yf.download(symbol, start=start_date, end=end_date)
                        elif market == 'HK':
                            # 使用akshare获取港股数据
                            # 注意：实际使用时可能需要调整akshare的API调用
                            try:
                                # 尝试使用akshare获取港股数据
                                data = ak.stock_hk_daily(symbol=symbol.replace('.HK', ''))
                                data.index = pd.to_datetime(data['日期'])
                                data = data[(data.index >= start_date) & (data.index <= end_date)]
                                # 重命名列以匹配yfinance格式
                                data = data.rename(columns={
                                    '开盘': 'Open', '收盘': 'Close', 
                                    '最高': 'High', '最低': 'Low',
                                    '成交量': 'Volume'
                                })
                            except:
                                # 如果akshare获取失败，尝试使用yfinance
                                logger.warning(f"使用akshare获取 {symbol} 数据失败，尝试使用yfinance")
                                data = yf.download(symbol, start=start_date, end=end_date)
                        
                        # 保存数据
                        if not data.empty:
                            self.stock_data[symbol] = data
                            # 保存到CSV
                            os.makedirs(f'data/raw/{market}', exist_ok=True)
                            data.to_csv(f'data/raw/{market}/{symbol.replace(".", "_")}.csv')
                            logger.info(f"已获取并保存 {symbol} 的历史数据，共 {len(data)} 条记录")
                        else:
                            logger.warning(f"获取 {symbol} 的历史数据为空")
                    
                    except Exception as e:
                        logger.error(f"获取 {symbol} 的历史数据时出错: {str(e)}")
        
        logger.info(f"股票历史数据获取完成，共获取 {len(self.stock_data)} 只股票的数据")
        
        return self.stock_data
    
    def get_stock_list_flat(self) -> List[Dict]:
        """
        获取扁平化的股票列表
        
        Returns:
            List[Dict]: 扁平化的股票列表
        """
        flat_list = []
        
        for market in self.config['markets']:
            for segment in self.config['industry_segments']:
                for stock in self.selected_stocks[market][segment]:
                    stock_info = stock.copy()
                    stock_info['market'] = market
                    stock_info['industry_segment'] = segment
                    flat_list.append(stock_info)
        
        return flat_list


def main():
    """主函数"""
    # 创建股票选择器
    selector = StockSelector()
    
    # 选择代表性股票
    selected_stocks = selector.select_representative_stocks(method='market_cap', top_n=5)
    
    # 打印选择的股票
    for market in selected_stocks:
        print(f"\n{market} 市场:")
        for segment in selected_stocks[market]:
            print(f"  {segment} 段:")
            for stock in selected_stocks[market][segment]:
                print(f"    {stock['symbol']}: {stock['name']} ({stock['segment']})")
    
    # 获取股票历史数据
    stock_data = selector.fetch_stock_data()
    
    print(f"\n共获取 {len(stock_data)} 只股票的历史数据")


if __name__ == "__main__":
    main() 