"""
机器学习模型模块

该模块实现了多种机器学习模型，用于预测股票的未来收益率或价格走势。
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习库
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

# 深度学习库
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置
from config.config import ML_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子，保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)


class FeatureSelector:
    """特征选择类，用于从原始特征中选择最重要的特征"""
    
    def __init__(self, method: str = 'rfe', n_features: int = 20):
        """
        初始化特征选择器
        
        Args:
            method: 特征选择方法，'rfe'表示递归特征消除
            n_features: 选择的特征数量
        """
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_features = []
        
        logger.info(f"特征选择器初始化完成，方法: {method}, 特征数量: {n_features}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        拟合特征选择器
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            List[str]: 选择的特征列表
        """
        logger.info(f"开始选择特征，原始特征数量: {X.shape[1]}")
        
        if self.method == 'rfe':
            # 使用线性回归作为基础模型
            estimator = LinearRegression()
            self.selector = RFE(estimator, n_features_to_select=self.n_features, step=1)
            self.selector.fit(X, y)
            
            # 获取选择的特征
            self.selected_features = [X.columns[i] for i in range(len(X.columns)) 
                                    if self.selector.support_[i]]
        else:
            logger.warning(f"不支持的特征选择方法: {self.method}")
            # 如果方法不支持，则选择前n个特征
            self.selected_features = list(X.columns[:self.n_features])
        
        logger.info(f"特征选择完成，选择了 {len(self.selected_features)} 个特征")
        
        return self.selected_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换特征矩阵，只保留选择的特征
        
        Args:
            X: 特征矩阵
            
        Returns:
            pd.DataFrame: 转换后的特征矩阵
        """
        if self.selector is not None and self.method == 'rfe':
            # 使用RFE选择器转换
            X_selected = self.selector.transform(X)
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        else:
            # 直接选择特征列
            return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        拟合并转换特征矩阵
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            pd.DataFrame: 转换后的特征矩阵
        """
        self.fit(X, y)
        return self.transform(X)


class ModelTrainer:
    """模型训练类，用于训练和评估机器学习模型"""
    
    def __init__(self, config: Dict = None):
        """
        初始化模型训练器
        
        Args:
            config: 配置字典，默认使用ML_CONFIG
        """
        self.config = config or ML_CONFIG
        self.models = {}
        self.feature_selector = None
        self.scaler = None
        self.feature_importance = {}
        
        # 创建模型保存目录
        os.makedirs('models', exist_ok=True)
        
        logger.info("模型训练器初始化完成")
    
    def prepare_data(self, data: Dict[str, pd.DataFrame], 
                   target_col: str = 'future_return_5d',
                   train_ratio: float = 0.8,
                   normalize: bool = True,
                   select_features: bool = True) -> Tuple[Dict, Dict, List[str]]:
        """
        准备训练数据
        
        Args:
            data: 股票数据字典，键为股票代码，值为特征数据框
            target_col: 目标变量列名
            train_ratio: 训练集比例
            normalize: 是否标准化特征
            select_features: 是否选择特征
            
        Returns:
            Tuple[Dict, Dict, List[str]]: (训练数据字典, 测试数据字典, 特征列表)
        """
        logger.info(f"开始准备训练数据，股票数量: {len(data)}")
        
        train_data = {}
        test_data = {}
        all_features = []
        
        # 对每只股票分别处理
        for symbol, df in data.items():
            # 确保数据按时间排序
            df = df.sort_index()
            
            # 丢弃包含NaN的行
            df = df.dropna()
            
            # 分离特征和目标变量
            if target_col not in df.columns:
                logger.warning(f"{symbol} 的数据中没有目标变量 {target_col}，跳过")
                continue
            
            y = df[target_col]
            
            # 排除不作为特征的列
            exclude_cols = [col for col in df.columns if col.startswith('future_') or 
                           col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            X = df.drop(exclude_cols, axis=1)
            
            # 记录所有特征
            all_features.extend(X.columns)
            
            # 分割训练集和测试集
            train_size = int(len(df) * train_ratio)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            train_data[symbol] = {'X': X_train, 'y': y_train}
            test_data[symbol] = {'X': X_test, 'y': y_test}
        
        # 获取唯一特征列表
        all_features = sorted(list(set(all_features)))
        
        # 标准化特征
        if normalize:
            logger.info("标准化特征")
            self.scaler = StandardScaler()
            
            # 对所有股票的训练集特征拟合标准化器
            combined_X_train = pd.concat([train_data[symbol]['X'] for symbol in train_data])
            self.scaler.fit(combined_X_train)
            
            # 转换所有数据
            for symbol in train_data:
                # 标准化训练集
                X_train_scaled = pd.DataFrame(
                    self.scaler.transform(train_data[symbol]['X']),
                    columns=train_data[symbol]['X'].columns,
                    index=train_data[symbol]['X'].index
                )
                train_data[symbol]['X'] = X_train_scaled
                
                # 标准化测试集
                X_test_scaled = pd.DataFrame(
                    self.scaler.transform(test_data[symbol]['X']),
                    columns=test_data[symbol]['X'].columns,
                    index=test_data[symbol]['X'].index
                )
                test_data[symbol]['X'] = X_test_scaled
        
        # 特征选择
        if select_features:
            logger.info("选择特征")
            n_features = min(20, len(all_features))
            self.feature_selector = FeatureSelector(
                method=self.config['feature_engineering']['feature_selection'],
                n_features=n_features
            )
            
            # 对所有股票的训练集特征拟合特征选择器
            combined_X_train = pd.concat([train_data[symbol]['X'] for symbol in train_data])
            combined_y_train = pd.concat([train_data[symbol]['y'] for symbol in train_data])
            selected_features = self.feature_selector.fit(combined_X_train, combined_y_train)
            
            # 转换所有数据
            for symbol in train_data:
                # 转换训练集
                train_data[symbol]['X'] = train_data[symbol]['X'][selected_features]
                
                # 转换测试集
                test_data[symbol]['X'] = test_data[symbol]['X'][selected_features]
            
            all_features = selected_features
        
        logger.info(f"数据准备完成，训练集: {len(train_data)}只股票，测试集: {len(test_data)}只股票，特征数: {len(all_features)}")
        
        return train_data, test_data, all_features
    
    def build_linear_model(self) -> Any:
        """
        构建线性回归模型
        
        Returns:
            Any: 线性回归模型
        """
        model_type = self.config['models']['linear']['type']
        params = self.config['models']['linear']['params']
        
        if model_type == 'linear_regression':
            model = LinearRegression(**params)
        elif model_type == 'ridge':
            model = Ridge(**params)
        elif model_type == 'lasso':
            model = Lasso(**params)
        else:
            logger.warning(f"不支持的线性模型类型: {model_type}，使用默认线性回归")
            model = LinearRegression()
        
        return model
    
    def build_tree_model(self) -> Any:
        """
        构建树模型
        
        Returns:
            Any: 树模型
        """
        model_type = self.config['models']['tree']['type']
        params = self.config['models']['tree']['params']
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(**params)
        else:
            logger.warning(f"不支持的树模型类型: {model_type}，使用默认随机森林")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        return model
    
    def build_boosting_model(self) -> Any:
        """
        构建提升模型
        
        Returns:
            Any: 提升模型
        """
        model_type = self.config['models']['boosting']['type']
        params = self.config['models']['boosting']['params']
        
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(**params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**params)
        else:
            logger.warning(f"不支持的提升模型类型: {model_type}，使用默认XGBoost")
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
        return model
    
    def build_deep_learning_model(self, input_dim: int) -> Any:
        """
        构建深度学习模型
        
        Args:
            input_dim: 输入维度
            
        Returns:
            Any: 深度学习模型
        """
        model_type = self.config['models']['deep_learning']['type']
        params = self.config['models']['deep_learning']['params']
        
        if model_type == 'lstm':
            # 创建LSTM模型
            model = Sequential()
            model.add(LSTM(
                units=params['units'],
                input_shape=(input_dim, 1),
                return_sequences=True
            ))
            model.add(Dropout(params['dropout']))
            model.add(LSTM(units=params['units'] // 2))
            model.add(Dropout(params['dropout']))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error'
            )
        else:
            logger.warning(f"不支持的深度学习模型类型: {model_type}，使用默认前馈神经网络")
            # 创建简单的前馈神经网络
            model = Sequential()
            model.add(Dense(64, activation='relu', input_dim=input_dim))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error'
            )
        
        return model
    
    def train_models(self, train_data: Dict, features: List[str]) -> Dict:
        """
        训练模型
        
        Args:
            train_data: 训练数据字典，键为股票代码，值为包含'X'和'y'的字典
            features: 特征列表
            
        Returns:
            Dict: 训练好的模型字典
        """
        logger.info("开始训练模型")
        
        # 合并所有股票的训练数据
        combined_X = pd.concat([train_data[symbol]['X'] for symbol in train_data])
        combined_y = pd.concat([train_data[symbol]['y'] for symbol in train_data])
        
        # 训练线性模型
        logger.info("训练线性模型")
        linear_model = self.build_linear_model()
        linear_model.fit(combined_X, combined_y)
        self.models['linear'] = linear_model
        
        # 保存特征重要性（如果是线性回归）
        if hasattr(linear_model, 'coef_'):
            self.feature_importance['linear'] = dict(zip(features, linear_model.coef_))
        
        # 训练树模型
        logger.info("训练树模型")
        tree_model = self.build_tree_model()
        tree_model.fit(combined_X, combined_y)
        self.models['tree'] = tree_model
        
        # 保存特征重要性
        if hasattr(tree_model, 'feature_importances_'):
            self.feature_importance['tree'] = dict(zip(features, tree_model.feature_importances_))
        
        # 训练提升模型
        logger.info("训练提升模型")
        boosting_model = self.build_boosting_model()
        boosting_model.fit(combined_X, combined_y)
        self.models['boosting'] = boosting_model
        
        # 保存特征重要性
        if hasattr(boosting_model, 'feature_importances_'):
            self.feature_importance['boosting'] = dict(zip(features, boosting_model.feature_importances_))
        
        # 训练深度学习模型（如果数据量足够）
        if len(combined_X) >= 1000:
            logger.info("训练深度学习模型")
            # 调整数据格式以适应LSTM
            model_type = self.config['models']['deep_learning']['type']
            params = self.config['models']['deep_learning']['params']
            
            if model_type == 'lstm':
                # 重塑数据为3D格式: [样本, 时间步, 特征]
                X_reshaped = combined_X.values.reshape(combined_X.shape[0], combined_X.shape[1], 1)
                
                # 构建LSTM模型
                dl_model = self.build_deep_learning_model(combined_X.shape[1])
                
                # 创建早停回调
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # 训练模型
                dl_model.fit(
                    X_reshaped, combined_y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
            else:
                # 构建前馈神经网络
                dl_model = self.build_deep_learning_model(combined_X.shape[1])
                
                # 创建早停回调
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # 训练模型
                dl_model.fit(
                    combined_X, combined_y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
            
            self.models['deep_learning'] = dl_model
        else:
            logger.warning("数据量不足，跳过深度学习模型训练")
        
        logger.info(f"模型训练完成，共训练了 {len(self.models)} 个模型")
        
        return self.models
    
    def evaluate_models(self, test_data: Dict) -> Dict:
        """
        评估模型
        
        Args:
            test_data: 测试数据字典，键为股票代码，值为包含'X'和'y'的字典
            
        Returns:
            Dict: 评估结果字典
        """
        logger.info("开始评估模型")
        
        evaluation_results = {}
        
        # 合并所有股票的测试数据
        combined_X = pd.concat([test_data[symbol]['X'] for symbol in test_data])
        combined_y = pd.concat([test_data[symbol]['y'] for symbol in test_data])
        
        # 计算基准模型（平均预测）的MSE
        y_mean = combined_y.mean()
        benchmark_mse = mean_squared_error(combined_y, [y_mean] * len(combined_y))
        benchmark_rmse = np.sqrt(benchmark_mse)
        
        # 评估每个模型
        for model_name, model in self.models.items():
            logger.info(f"评估 {model_name} 模型")
            
            try:
                # 预测
                if model_name == 'deep_learning' and self.config['models']['deep_learning']['type'] == 'lstm':
                    # 重塑数据为3D格式: [样本, 时间步, 特征]
                    X_reshaped = combined_X.values.reshape(combined_X.shape[0], combined_X.shape[1], 1)
                    y_pred = model.predict(X_reshaped).flatten()
                else:
                    y_pred = model.predict(combined_X)
                
                # 计算评估指标
                mse = mean_squared_error(combined_y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(combined_y, y_pred)
                r2 = r2_score(combined_y, y_pred)
                
                # 记录结果
                evaluation_results[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'improvement_over_benchmark': (benchmark_rmse - rmse) / benchmark_rmse * 100
                }
                
                logger.info(f"{model_name} 模型性能: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")
                
                # 预测每只股票的性能
                stock_performance = {}
                for symbol in test_data:
                    X_test = test_data[symbol]['X']
                    y_test = test_data[symbol]['y']
                    
                    if model_name == 'deep_learning' and self.config['models']['deep_learning']['type'] == 'lstm':
                        # 重塑数据为3D格式
                        X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
                        y_pred = model.predict(X_test_reshaped).flatten()
                    else:
                        y_pred = model.predict(X_test)
                    
                    # 计算评估指标
                    stock_mse = mean_squared_error(y_test, y_pred)
                    stock_rmse = np.sqrt(stock_mse)
                    stock_mae = mean_absolute_error(y_test, y_pred)
                    stock_r2 = r2_score(y_test, y_pred)
                    
                    stock_performance[symbol] = {
                        'mse': stock_mse,
                        'rmse': stock_rmse,
                        'mae': stock_mae,
                        'r2': stock_r2
                    }
                
                evaluation_results[model_name]['stock_performance'] = stock_performance
            
            except Exception as e:
                logger.error(f"评估 {model_name} 模型时出错: {str(e)}")
                evaluation_results[model_name] = {'error': str(e)}
        
        logger.info("模型评估完成")
        
        return evaluation_results
    
    def save_models(self, path: str = 'models') -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        logger.info(f"开始保存模型到 {path}")
        
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存标准化器
        if self.scaler is not None:
            with open(f"{path}/scaler_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # 保存特征选择器
        if self.feature_selector is not None:
            with open(f"{path}/feature_selector_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        # 保存模型
        for model_name, model in self.models.items():
            if model_name == 'deep_learning':
                # 保存Keras模型
                model.save(f"{path}/{model_name}_{timestamp}.h5")
            else:
                # 保存sklearn模型
                with open(f"{path}/{model_name}_{timestamp}.pkl", 'wb') as f:
                    pickle.dump(model, f)
        
        # 保存特征重要性
        with open(f"{path}/feature_importance_{timestamp}.json", 'w') as f:
            # 将ndarray转换为list
            importance_dict = {}
            for model_name, importance in self.feature_importance.items():
                importance_dict[model_name] = {
                    feature: float(value) if isinstance(value, np.float32) or isinstance(value, np.float64) else value
                    for feature, value in importance.items()
                }
            json.dump(importance_dict, f, indent=4)
        
        logger.info(f"模型保存完成，时间戳: {timestamp}")
    
    def load_models(self, path: str = 'models', timestamp: str = None) -> Dict:
        """
        加载模型
        
        Args:
            path: 加载路径
            timestamp: 时间戳，如果为None则加载最新的模型
            
        Returns:
            Dict: 加载的模型字典
        """
        logger.info(f"开始从 {path} 加载模型")
        
        # 如果没有指定时间戳，则查找最新的模型
        if timestamp is None:
            # 查找所有模型文件
            model_files = [f for f in os.listdir(path) if f.endswith('.pkl') or f.endswith('.h5')]
            if not model_files:
                logger.error("未找到模型文件")
                return {}
            
            # 提取时间戳
            timestamps = set()
            for f in model_files:
                parts = f.split('_')
                if len(parts) >= 2:
                    ts = parts[-1].split('.')[0]
                    timestamps.add(ts)
            
            if not timestamps:
                logger.error("无法提取时间戳")
                return {}
            
            # 选择最新的时间戳
            timestamp = max(timestamps)
        
        logger.info(f"加载时间戳为 {timestamp} 的模型")
        
        # 加载标准化器
        scaler_path = f"{path}/scaler_{timestamp}.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            logger.warning(f"未找到标准化器文件: {scaler_path}")
        
        # 加载特征选择器
        selector_path = f"{path}/feature_selector_{timestamp}.pkl"
        if os.path.exists(selector_path):
            with open(selector_path, 'rb') as f:
                self.feature_selector = pickle.load(f)
        else:
            logger.warning(f"未找到特征选择器文件: {selector_path}")
        
        # 加载模型
        for model_type in ['linear', 'tree', 'boosting']:
            model_path = f"{path}/{model_type}_{timestamp}.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                logger.info(f"已加载 {model_type} 模型")
            else:
                logger.warning(f"未找到模型文件: {model_path}")
        
        # 加载深度学习模型
        dl_model_path = f"{path}/deep_learning_{timestamp}.h5"
        if os.path.exists(dl_model_path):
            self.models['deep_learning'] = load_model(dl_model_path)
            logger.info("已加载深度学习模型")
        else:
            logger.warning(f"未找到深度学习模型文件: {dl_model_path}")
        
        # 加载特征重要性
        importance_path = f"{path}/feature_importance_{timestamp}.json"
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
            logger.info("已加载特征重要性")
        else:
            logger.warning(f"未找到特征重要性文件: {importance_path}")
        
        logger.info(f"模型加载完成，共加载了 {len(self.models)} 个模型")
        
        return self.models
    
    def predict(self, X: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征矩阵
            model_name: 模型名称，'ensemble'表示集成所有模型的预测
            
        Returns:
            np.ndarray: 预测结果
        """
        logger.info(f"使用 {model_name} 模型进行预测")
        
        # 检查模型是否已加载
        if not self.models:
            logger.error("未加载任何模型，无法进行预测")
            return np.array([])
        
        # 预处理特征
        X_processed = X.copy()
        
        # 应用标准化
        if self.scaler is not None:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        # 应用特征选择
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        # 进行预测
        if model_name == 'ensemble':
            # 集成所有模型的预测
            predictions = []
            weights = {'linear': 0.2, 'tree': 0.3, 'boosting': 0.3, 'deep_learning': 0.2}
            
            for model_type, model in self.models.items():
                try:
                    if model_type == 'deep_learning' and self.config['models']['deep_learning']['type'] == 'lstm':
                        # 重塑数据为3D格式
                        X_reshaped = X_processed.values.reshape(X_processed.shape[0], X_processed.shape[1], 1)
                        pred = model.predict(X_reshaped).flatten()
                    else:
                        pred = model.predict(X_processed)
                    
                    # 加权预测
                    weight = weights.get(model_type, 0.25)
                    predictions.append(pred * weight)
                except Exception as e:
                    logger.error(f"使用 {model_type} 模型预测时出错: {str(e)}")
            
            # 合并预测结果
            if predictions:
                return np.sum(predictions, axis=0)
            else:
                logger.error("所有模型预测都失败")
                return np.array([])
        elif model_name in self.models:
            # 使用指定模型预测
            model = self.models[model_name]
            
            try:
                if model_name == 'deep_learning' and self.config['models']['deep_learning']['type'] == 'lstm':
                    # 重塑数据为3D格式
                    X_reshaped = X_processed.values.reshape(X_processed.shape[0], X_processed.shape[1], 1)
                    return model.predict(X_reshaped).flatten()
                else:
                    return model.predict(X_processed)
            except Exception as e:
                logger.error(f"使用 {model_name} 模型预测时出错: {str(e)}")
                return np.array([])
        else:
            logger.error(f"未找到名为 {model_name} 的模型")
            return np.array([])
    
    def plot_feature_importance(self, model_name: str = 'tree', top_n: int = 10) -> None:
        """
        绘制特征重要性图表
        
        Args:
            model_name: 模型名称
            top_n: 显示前n个重要特征
        """
        if model_name not in self.feature_importance:
            logger.error(f"未找到 {model_name} 模型的特征重要性")
            return
        
        # 获取特征重要性
        importance = self.feature_importance[model_name]
        
        # 转换为DataFrame并排序
        df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df)
        plt.title(f'{model_name.capitalize()} Model - Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(f'results/figures/{model_name}_feature_importance.png')
        plt.close()
        
        logger.info(f"已保存 {model_name} 模型的特征重要性图表")


def main():
    """主函数"""
    import os
    import glob
    
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
    
    # 创建模型训练器
    trainer = ModelTrainer()
    
    # 准备训练数据
    train_data, test_data, features = trainer.prepare_data(
        processed_data,
        target_col='future_return_5d',
        train_ratio=0.8,
        normalize=True,
        select_features=True
    )
    
    # 训练模型
    models = trainer.train_models(train_data, features)
    
    # 评估模型
    evaluation_results = trainer.evaluate_models(test_data)
    
    # 打印评估结果
    print("\n模型评估结果:")
    for model_name, results in evaluation_results.items():
        if 'error' in results:
            print(f"  {model_name} 模型评估失败: {results['error']}")
        else:
            print(f"  {model_name} 模型:")
            print(f"    MSE: {results['mse']:.6f}")
            print(f"    RMSE: {results['rmse']:.6f}")
            print(f"    MAE: {results['mae']:.6f}")
            print(f"    R²: {results['r2']:.6f}")
            print(f"    相比基准改进: {results['improvement_over_benchmark']:.2f}%")
    
    # 保存模型
    trainer.save_models()
    
    # 绘制特征重要性图表
    for model_name in ['linear', 'tree', 'boosting']:
        if model_name in trainer.feature_importance:
            trainer.plot_feature_importance(model_name)
    
    print("\n模型训练和评估完成，模型已保存")


if __name__ == "__main__":
    main()
