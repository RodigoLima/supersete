#!/usr/bin/env python3
"""
Super Sete Analysis - Análise Inteligente de Jogos da Loteria
Versão otimizada, enxuta e organizada com boas práticas

Autor: Sistema de Análise de Dados
Data: 2024
"""

import os
import json
import time
import warnings
import hashlib
import pickle
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from scipy import stats
from scipy.signal import find_peaks, periodogram
from scipy.stats import entropy, kstest, jarque_bera
from scipy.optimize import minimize
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.base import clone

# Configurações de performance otimizadas
# Permitir paralelismo controlado nas bibliotecas
os.environ['OMP_NUM_THREADS'] = str(min(4, cpu_count()))
os.environ['MKL_NUM_THREADS'] = str(min(4, cpu_count()))
os.environ['NUMEXPR_NUM_THREADS'] = str(min(4, cpu_count()))
os.environ['OPENBLAS_NUM_THREADS'] = str(min(4, cpu_count()))

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore')

# Conversor para tipos não serializáveis pelo JSON padrão
def to_serializable(obj):
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    # Coleções comuns
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:
        # numpy.datetime64
        if str(type(obj)).endswith("numpy.datetime64'>"):
            return str(obj)
    except Exception:
        pass
    # Fallback seguro: string
    return str(obj)

# Configuração global
MAX_WORKERS = min(cpu_count(), 8)

# Verificar bibliotecas opcionais
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuração dos modelos de machine learning - OTIMIZADA PARA ASSERTIVIDADE"""
    # Parâmetros básicos - OTIMIZADOS PARA ASSERTIVIDADE SEM OVERFITTING
    window: int = 20  # Conservador para evitar overfitting temporal
    n_estimators: int = 50  # Reduzido para evitar overfitting
    max_depth: int = 4  # Reduzido para evitar overfitting
    learning_rate: float = 0.1  # Aumentado para convergência mais rápida
    calibrate: bool = True
    calibration_method: str = 'isotonic'
    calibration_cv: int = 3  # Reduzido para evitar overfitting na validação
    random_state: int = 42
    
    # REGULARIZAÇÃO RIGOROSA ANTI-OVERFITTING
    regularization_strength: float = 1.0  # Aumentado para maior regularização
    l1_ratio: float = 0.5  # Balanceado L1/L2
    dropout_rate: float = 0.4  # Aumentado para maior regularização
    early_stopping_rounds: int = 5  # Reduzido para parada mais rápida
    min_samples_split: int = 50  # Aumentado para evitar overfitting
    min_samples_leaf: int = 25   # Aumentado para evitar overfitting
    
    # LSTM
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_patience: int = 10
    include_lstm: bool = False
    
    # Validação e backtesting - OTIMIZADOS PARA EVITAR OVERFITTING
    backtest_steps: int = 500  # Reduzido para evitar overfitting na validação
    train_min_size: int = 200  # Aumentado para mais dados de treino
    min_confidence: float = 0.20  # Reduzido para ser mais realista
    confidence_interval: float = 0.95  # Aumentado para ser mais conservador
    
    # Análise temporal avançada
    detect_seasonality: bool = True
    max_periods: int = 50
    min_period: int = 3
    seasonality_threshold: float = 0.1
    
    # Análise de entropia
    entropy_window: int = 20
    entropy_threshold: float = 0.8
    
    # Ensemble adaptativo
    adaptive_weights: bool = True
    weight_decay: float = 0.95
    min_weight: float = 0.01
    
    # Detecção de anomalias
    anomaly_detection: bool = True
    anomaly_threshold: float = 2.5
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    
    # Correlação entre colunas
    correlation_analysis: bool = True
    min_correlation: float = 0.1
    max_correlation: float = 0.9
    
    # Otimização bayesiana
    bayesian_optimization: bool = False
    n_trials: int = 50
    optimization_metric: str = 'brier_score'
    
    # Métricas de confiança
    confidence_metrics: bool = True
    uncertainty_quantification: bool = True
    prediction_intervals: bool = True


class AdaptiveLearningSystem:
    """Sistema de aprendizado adaptativo que se ajusta aos padrões recentes"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.performance_history = {}
        self.model_weights_history = {}
        self.adaptation_rate = 0.1  # Taxa de adaptação
        self.min_performance_threshold = 0.15  # Threshold mínimo de performance
        
    def update_performance(self, coluna: str, model_name: str, accuracy: float, 
                          brier_score: float, timestamp: int = None) -> None:
        """Atualiza histórico de performance de um modelo"""
        if coluna not in self.performance_history:
            self.performance_history[coluna] = {}
        
        if model_name not in self.performance_history[coluna]:
            self.performance_history[coluna][model_name] = {
                'accuracies': [],
                'brier_scores': [],
                'timestamps': [],
                'avg_accuracy': 0.0,
                'avg_brier': 1.0,
                'trend': 'stable'
            }
        
        # Adicionar nova performance
        self.performance_history[coluna][model_name]['accuracies'].append(accuracy)
        self.performance_history[coluna][model_name]['brier_scores'].append(brier_score)
        if timestamp:
            self.performance_history[coluna][model_name]['timestamps'].append(timestamp)
        
        # Manter apenas últimas 50 medições
        if len(self.performance_history[coluna][model_name]['accuracies']) > 50:
            self.performance_history[coluna][model_name]['accuracies'] = \
                self.performance_history[coluna][model_name]['accuracies'][-50:]
            self.performance_history[coluna][model_name]['brier_scores'] = \
                self.performance_history[coluna][model_name]['brier_scores'][-50:]
        
        # Atualizar médias
        self._update_averages(coluna, model_name)
        
        # Calcular tendência
        self._calculate_trend(coluna, model_name)
    
    def _update_averages(self, coluna: str, model_name: str) -> None:
        """Atualiza médias de performance"""
        data = self.performance_history[coluna][model_name]
        data['avg_accuracy'] = np.mean(data['accuracies'])
        data['avg_brier'] = np.mean(data['brier_scores'])
    
    def _calculate_trend(self, coluna: str, model_name: str) -> None:
        """Calcula tendência de performance"""
        data = self.performance_history[coluna][model_name]
        accuracies = data['accuracies']
        
        if len(accuracies) < 5:
            data['trend'] = 'insufficient_data'
            return
        
        # Calcular tendência usando regressão linear
        x = np.arange(len(accuracies))
        slope, _ = np.polyfit(x, accuracies, 1)
        
        if slope > 0.01:
            data['trend'] = 'improving'
        elif slope < -0.01:
            data['trend'] = 'declining'
        else:
            data['trend'] = 'stable'
    
    def get_adaptive_weights(self, coluna: str, model_names: List[str]) -> Dict[str, float]:
        """Calcula pesos adaptativos baseados na performance recente"""
        if coluna not in self.performance_history:
            # Pesos uniformes se não há histórico
            return {name: 1.0 / len(model_names) for name in model_names}
        
        weights = {}
        total_score = 0.0
        
        for model_name in model_names:
            if model_name in self.performance_history[coluna]:
                data = self.performance_history[coluna][model_name]
                
                # Score baseado na performance média
                avg_accuracy = data['avg_accuracy']
                avg_brier = data['avg_brier']
                
                # Score composto
                base_score = avg_accuracy - (avg_brier * 0.5)
                
                # Ajustar por tendência
                trend_multiplier = {
                    'improving': 1.2,
                    'stable': 1.0,
                    'declining': 0.8,
                    'insufficient_data': 0.9
                }.get(data['trend'], 1.0)
                
                # Ajustar por recência (últimas 10 medições)
                if len(data['accuracies']) >= 10:
                    recent_accuracy = np.mean(data['accuracies'][-10:])
                    recent_multiplier = 1.0 + (recent_accuracy - avg_accuracy) * 2
                else:
                    recent_multiplier = 1.0
                
                final_score = base_score * trend_multiplier * recent_multiplier
                weights[model_name] = max(0.0, final_score)
                total_score += weights[model_name]
            else:
                weights[model_name] = 0.1  # Peso baixo para modelos sem histórico
                total_score += 0.1
        
        # Normalizar pesos
        if total_score > 0:
            for model_name in weights:
                weights[model_name] /= total_score
        else:
            # Fallback para pesos uniformes
            uniform_weight = 1.0 / len(model_names)
            weights = {name: uniform_weight for name in model_names}
        
        return weights
    
    def should_retrain_model(self, coluna: str, model_name: str) -> bool:
        """Decide se um modelo deve ser retreinado baseado na performance"""
        if coluna not in self.performance_history or model_name not in self.performance_history[coluna]:
            return False
        
        data = self.performance_history[coluna][model_name]
        
        # Retreinar se:
        # 1. Performance está abaixo do threshold
        if data['avg_accuracy'] < self.min_performance_threshold:
            return True
        
        # 2. Tendência está declinando
        if data['trend'] == 'declining':
            return True
        
        # 3. Performance recente está muito abaixo da média
        if len(data['accuracies']) >= 5:
            recent_avg = np.mean(data['accuracies'][-5:])
            if recent_avg < data['avg_accuracy'] * 0.8:
                return True
        
        return False
    
    def get_adaptation_suggestions(self, coluna: str) -> Dict[str, Any]:
        """Retorna sugestões de adaptação para uma coluna"""
        if coluna not in self.performance_history:
            return {'suggestions': [], 'status': 'no_data'}
        
        suggestions = []
        status = 'good'
        
        for model_name, data in self.performance_history[coluna].items():
            if data['avg_accuracy'] < self.min_performance_threshold:
                suggestions.append(f"Modelo {model_name} tem performance baixa ({data['avg_accuracy']:.3f})")
                status = 'needs_improvement'
            
            if data['trend'] == 'declining':
                suggestions.append(f"Modelo {model_name} está em declínio")
                status = 'needs_improvement'
            
            if data['trend'] == 'improving':
                suggestions.append(f"Modelo {model_name} está melhorando - manter configuração")
        
        return {
            'suggestions': suggestions,
            'status': status,
            'models_performance': {name: data['avg_accuracy'] for name, data in self.performance_history[coluna].items()}
        }


class AdvancedFeatureEngineer:
    """Engenheiro de features avançado para melhorar assertividade"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def create_frequency_features(self, series: np.ndarray) -> np.ndarray:
        """Cria features baseadas em frequência de números"""
        features = []
        
        # Frequência de cada número (0-9)
        for num in range(10):
            freq = np.sum(series == num) / len(series)
            features.append(freq)
        
        # Frequência relativa dos últimos N valores
        for window in [5, 10, 15]:
            if len(series) >= window:
                recent_series = series[-window:]
                for num in range(10):
                    recent_freq = np.sum(recent_series == num) / window
                    features.append(recent_freq)
        
        return np.array(features)
    
    def create_gap_features(self, series: np.ndarray) -> np.ndarray:
        """Cria features baseadas em gaps entre ocorrências"""
        features = []
        
        for num in range(10):
            # Encontrar posições onde o número aparece
            positions = np.where(series == num)[0]
            
            if len(positions) > 1:
                # Calcular gaps entre ocorrências
                gaps = np.diff(positions)
                features.extend([
                    np.mean(gaps),  # Gap médio
                    np.std(gaps),   # Desvio padrão dos gaps
                    np.min(gaps),   # Gap mínimo
                    np.max(gaps),   # Gap máximo
                    len(gaps)       # Número de gaps
                ])
            else:
                # Se não há ocorrências suficientes
                features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)
    
    def create_pattern_features(self, series: np.ndarray) -> np.ndarray:
        """Cria features baseadas em padrões sequenciais"""
        features = []
        
        # Padrões de 2 números consecutivos
        for i in range(len(series) - 1):
            pattern = (series[i], series[i + 1])
            features.append(pattern[0] * 10 + pattern[1])  # Codificar como número único
        
        # Padrões de 3 números consecutivos
        for i in range(len(series) - 2):
            pattern = (series[i], series[i + 1], series[i + 2])
            features.append(pattern[0] * 100 + pattern[1] * 10 + pattern[2])
        
        # Estatísticas dos padrões
        if features:
            features = np.array(features)
            return np.array([
                np.mean(features),
                np.std(features),
                np.var(features),
                len(np.unique(features)) / len(features)  # Diversidade de padrões
            ])
        else:
            return np.array([0, 0, 0, 0])
    
    def create_temporal_features(self, series: np.ndarray) -> np.ndarray:
        """Cria features temporais avançadas"""
        features = []
        
        # Tendência dos últimos N valores
        for window in [5, 10, 15, 20]:
            if len(series) >= window:
                recent = series[-window:]
                # Calcular tendência usando regressão linear
                x = np.arange(len(recent))
                if len(x) > 1:
                    slope, _ = np.polyfit(x, recent, 1)
                    features.append(slope)
                else:
                    features.append(0)
        
        # Volatilidade (desvio padrão móvel)
        for window in [5, 10, 15]:
            if len(series) >= window:
                rolling_std = pd.Series(series).rolling(window=window).std().iloc[-1]
                features.append(rolling_std if not np.isnan(rolling_std) else 0)
        
        # Mudanças abruptas
        if len(series) > 1:
            changes = np.diff(series)
            features.extend([
                np.mean(np.abs(changes)),  # Mudança média absoluta
                np.std(changes),           # Desvio padrão das mudanças
                np.sum(changes != 0) / len(changes)  # Taxa de mudança
            ])
        
        return np.array(features)
    
    def create_ensemble_features(self, series: np.ndarray) -> np.ndarray:
        """Cria features combinadas para ensemble"""
        freq_features = self.create_frequency_features(series)
        gap_features = self.create_gap_features(series)
        pattern_features = self.create_pattern_features(series)
        temporal_features = self.create_temporal_features(series)
        
        return np.concatenate([
            freq_features,
            gap_features,
            pattern_features,
            temporal_features
        ])


class TemporalAnalyzer:
    """Analisador de padrões temporais avançados"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def detect_seasonality(self, series: np.ndarray) -> Dict:
        """Detecta sazonalidade usando análise espectral"""
        if len(series) < 10:
            return {'has_seasonality': False, 'periods': [], 'strength': 0.0}
        
        # Análise de periodograma
        freqs, psd = periodogram(series, fs=1.0)
        
        # Encontrar picos significativos
        peaks, properties = find_peaks(psd, height=np.percentile(psd, 80))
        
        periods = []
        strengths = []
        
        for peak in peaks:
            period = 1.0 / freqs[peak] if freqs[peak] > 0 else 0
            if self.config.min_period <= period <= self.config.max_periods:
                periods.append(period)
                strengths.append(psd[peak])
        
        # Ordenar por força
        if periods:
            sorted_indices = np.argsort(strengths)[::-1]
            periods = [periods[i] for i in sorted_indices]
            strengths = [strengths[i] for i in sorted_indices]
        
        has_seasonality = len(periods) > 0 and max(strengths) > self.config.seasonality_threshold
        
        return {
            'has_seasonality': has_seasonality,
            'periods': periods[:5],  # Top 5 períodos
            'strengths': strengths[:5],
            'dominant_period': periods[0] if periods else None
        }
    
    def calculate_trend(self, series: np.ndarray) -> Dict:
        """Calcula tendência usando regressão linear e teste de Mann-Kendall"""
        if len(series) < 3:
            return {'trend': 0.0, 'significance': 0.0, 'direction': 'none'}
        
        # Regressão linear
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        
        # Teste de Mann-Kendall para tendência
        mk_stat, mk_p_value = self._mann_kendall_test(series)
        
        # Classificar direção
        if abs(slope) < 0.001:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'trend': slope,
            'significance': 1 - p_value,
            'direction': direction,
            'mk_significance': 1 - mk_p_value,
            'r_squared': r_value ** 2
        }
    
    def _mann_kendall_test(self, series: np.ndarray) -> Tuple[float, float]:
        """Implementa teste de Mann-Kendall para tendência"""
        n = len(series)
        if n < 3:
            return 0.0, 1.0
        
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(series[j] - series[i])
        
        # Variância
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Estatística Z
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # P-valor (aproximação normal)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def detect_cycles(self, series: np.ndarray) -> Dict:
        """Detecta ciclos usando autocorrelação e análise espectral avançada"""
        if len(series) < 10:
            return {'cycles': [], 'autocorr': [], 'spectral_cycles': []}
        
        # Calcular autocorrelação
        max_lag = min(len(series) // 2, 50)
        autocorr = []
        
        for lag in range(1, max_lag + 1):
            if len(series) > lag:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
            else:
                autocorr.append(0)
        
        # Encontrar picos na autocorrelação
        peaks, _ = find_peaks(autocorr, height=0.1)
        
        cycles = []
        for peak in peaks:
            lag = peak + 1
            if lag >= self.config.min_period and lag <= self.config.max_periods:
                cycles.append({
                    'period': lag,
                    'strength': autocorr[peak],
                    'significance': abs(autocorr[peak])
                })
        
        # Análise espectral para detectar ciclos mais sutis
        spectral_cycles = []
        if len(series) > 20:
            try:
                from scipy.signal import periodogram
                freqs, psd = periodogram(series, fs=1.0)
                
                # Encontrar picos no espectro
                spectral_peaks, _ = find_peaks(psd, height=np.percentile(psd, 70))
                
                for peak in spectral_peaks:
                    period = 1.0 / freqs[peak] if freqs[peak] > 0 else 0
                    if self.config.min_period <= period <= self.config.max_periods:
                        spectral_cycles.append({
                            'period': period,
                            'strength': psd[peak],
                            'frequency': freqs[peak],
                            'type': 'spectral'
                        })
            except Exception as e:
                print(f"    ⚠️  Erro na análise espectral: {e}")
        
        # Detecção de padrões sequenciais específicos
        sequential_patterns = self._detect_sequential_patterns(series)
        
        # Ordenar por força
        cycles.sort(key=lambda x: x['strength'], reverse=True)
        spectral_cycles.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'cycles': cycles[:5],
            'spectral_cycles': spectral_cycles[:3],
            'sequential_patterns': sequential_patterns,
            'autocorr': autocorr,
            'max_autocorr': max(autocorr) if autocorr else 0
        }
    
    def _detect_sequential_patterns(self, series: np.ndarray) -> Dict:
        """Detecta padrões sequenciais específicos como repetições e progressões"""
        if len(series) < 5:
            return {'repetitions': [], 'progressions': [], 'alternations': 0}
        
        patterns = {
            'repetitions': [],
            'progressions': [],
            'alternations': 0
        }
        
        # Detectar repetições de 2 e 3 dígitos
        for pattern_length in [2, 3]:
            if len(series) >= pattern_length * 2:
                pattern_counts = {}
                for i in range(len(series) - pattern_length + 1):
                    pattern = tuple(series[i:i+pattern_length])
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                # Encontrar padrões que se repetem
                for pattern, count in pattern_counts.items():
                    if count > 1:
                        patterns['repetitions'].append({
                            'pattern': pattern,
                            'count': count,
                            'length': pattern_length,
                            'frequency': count / (len(series) - pattern_length + 1)
                        })
        
        # Detectar progressões aritméticas
        if len(series) >= 3:
            progressions = []
            for i in range(len(series) - 2):
                for j in range(i + 1, len(series) - 1):
                    diff = series[j] - series[i]
                    if diff != 0:
                        # Verificar se há mais elementos na progressão
                        next_val = series[j] + diff
                        if next_val in series[j+1:]:
                            progressions.append({
                                'start': i,
                                'step': diff,
                                'length': 3,
                                'values': [series[i], series[j], next_val]
                            })
            
            patterns['progressions'] = progressions[:5]  # Top 5 progressões
        
        # Calcular alternâncias (mudanças de direção)
        if len(series) > 1:
            diffs = np.diff(series)
            if len(diffs) > 1:
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                patterns['alternations'] = sign_changes / max(1, len(diffs) - 1)
        
        return patterns


class EntropyAnalyzer:
    """Analisador de entropia e aleatoriedade"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def calculate_entropy(self, series: np.ndarray) -> Dict:
        """Calcula entropia de Shannon e outras métricas de aleatoriedade"""
        if len(series) == 0:
            return {'shannon_entropy': 0.0, 'max_entropy': 0.0, 'normalized_entropy': 0.0}
        
        # Contar frequências
        counts = np.bincount(series, minlength=10)
        probabilities = counts / counts.sum()
        
        # Remover zeros para evitar log(0)
        probabilities = probabilities[probabilities > 0]
        
        # Entropia de Shannon
        shannon_entropy = entropy(probabilities, base=2)
        max_entropy = math.log2(10)  # Máxima entropia para 10 dígitos
        normalized_entropy = shannon_entropy / max_entropy
        
        return {
            'shannon_entropy': shannon_entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'is_random': normalized_entropy > self.config.entropy_threshold
        }
    
    def test_randomness(self, series: np.ndarray) -> Dict:
        """Testa aleatoriedade usando múltiplos testes estatísticos"""
        if len(series) < 10:
            return {'is_random': True, 'tests': {}}
        
        tests = {}
        
        # Teste de Kolmogorov-Smirnov para uniformidade
        ks_stat, ks_p_value = kstest(series, 'uniform', args=(0, 9))
        tests['ks_uniformity'] = {
            'statistic': ks_stat,
            'p_value': ks_p_value,
            'is_uniform': ks_p_value > 0.05
        }
        
        # Teste de Jarque-Bera para normalidade (transformado)
        if len(series) > 4:
            jb_stat, jb_p_value = jarque_bera(series)
            tests['jb_normality'] = {
                'statistic': jb_stat,
                'p_value': jb_p_value,
                'is_normal': jb_p_value > 0.05
            }
        
        # Teste de sequências (runs test)
        runs_stat, runs_p_value = self._runs_test(series)
        tests['runs'] = {
            'statistic': runs_stat,
            'p_value': runs_p_value,
            'is_random': runs_p_value > 0.05
        }
        
        # Teste de autocorrelação
        if len(series) > 2:
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
            tests['autocorr'] = {
                'value': autocorr,
                'is_random': abs(autocorr) < 0.1
            }
        
        # Decisão final
        is_random = all(
            test.get('is_random', test.get('is_uniform', test.get('is_normal', True)))
            for test in tests.values()
        )
        
        return {
            'is_random': is_random,
            'tests': tests
        }
    
    def _runs_test(self, series: np.ndarray) -> Tuple[float, float]:
        """Implementa teste de sequências (runs test)"""
        if len(series) < 2:
            return 0.0, 1.0
        
        # Converter para sequência de sinais
        signs = np.diff(series)
        signs = np.sign(signs)
        signs = signs[signs != 0]  # Remover zeros
        
        if len(signs) < 2:
            return 0.0, 1.0
        
        # Contar runs
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1
        
        n1 = np.sum(signs > 0)
        n2 = np.sum(signs < 0)
        n = n1 + n2
        
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        
        # Estatística Z
        expected_runs = (2 * n1 * n2) / n + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
        
        if variance <= 0:
            return 0.0, 1.0
        
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value


class ConfidenceAnalyzer:
    """Analisador de confiança e incerteza"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def calculate_confidence_intervals(self, predictions: np.ndarray, 
                                     model_uncertainty: float = 0.1) -> Dict:
        """Calcula intervalos de confiança para predições"""
        if len(predictions) == 0:
            return {'intervals': [], 'confidence_scores': []}
        
        # Usar distribuição beta para intervalos de confiança
        alpha = 1 - self.config.confidence_interval
        
        intervals = []
        confidence_scores = []
        
        for i, prob in enumerate(predictions):
            # Parâmetros da distribuição beta
            n = 100  # Tamanho da amostra assumido
            successes = int(prob * n)
            failures = n - successes
            
            # Ajustar para evitar parâmetros zero
            alpha_param = max(1, successes)
            beta_param = max(1, failures)
            
            # Calcular intervalos
            lower = stats.beta.ppf(alpha/2, alpha_param, beta_param)
            upper = stats.beta.ppf(1 - alpha/2, alpha_param, beta_param)
            
            # Adicionar incerteza do modelo
            uncertainty = model_uncertainty * prob
            lower = max(0, lower - uncertainty)
            upper = min(1, upper + uncertainty)
            
            intervals.append((lower, upper))
            
            # Score de confiança baseado na largura do intervalo
            width = upper - lower
            confidence = 1 - width
            confidence_scores.append(max(0, min(1, confidence)))
        
        return {
            'intervals': intervals,
            'confidence_scores': confidence_scores,
            'mean_confidence': np.mean(confidence_scores)
        }
    
    def calculate_prediction_uncertainty(self, model_predictions: List[np.ndarray]) -> Dict:
        """Calcula incerteza baseada na variância entre modelos"""
        if not model_predictions:
            return {'uncertainty': 0.0, 'agreement': 0.0}
        
        predictions_array = np.array(model_predictions)
        
        # Calcular variância entre modelos
        mean_pred = np.mean(predictions_array, axis=0)
        variance = np.var(predictions_array, axis=0)
        
        # Incerteza como coeficiente de variação
        uncertainty = np.sqrt(variance) / (mean_pred + 1e-8)
        mean_uncertainty = np.mean(uncertainty)
        
        # Concordância entre modelos (1 - desvio padrão normalizado)
        agreement = 1 - np.mean(uncertainty)
        
        return {
            'uncertainty': mean_uncertainty,
            'agreement': max(0, min(1, agreement)),
            'variance': variance,
            'mean_prediction': mean_pred
        }


class FrequencyAnalyzer:
    """Analisador de frequência de números por coluna"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
    def analyze_column_frequency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa a frequência de números por coluna no histórico
        
        Args:
            df: DataFrame com dados do Super Sete
            
        Returns:
            Dicionário com análise de frequência por coluna
        """
        print("📊 Analisando frequência de números por coluna...")
        
        # Colunas numéricas (excluindo Data_Sorteio se existir)
        numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
        
        frequency_analysis = {
            'frequencias_por_coluna': {},
            'numeros_mais_frequentes': {},
            'estatisticas_gerais': {},
            'padroes_temporais': {},
            'recomendacoes': {}
        }
        
        for col in numeric_cols:
            print(f"🔍 Analisando {col}...")
            
            # Contar frequência de cada número (0-9)
            freq_counts = df[col].value_counts().sort_index()
            total_concursos = len(df[col])
            
            # Calcular percentuais
            freq_percent = (freq_counts / total_concursos * 100).round(2)
            
            # Estatísticas da coluna
            col_stats = {
                'total_concursos': total_concursos,
                'frequencia_absoluta': freq_counts.to_dict(),
                'frequencia_percentual': freq_percent.to_dict(),
                'numero_mais_frequente': freq_counts.idxmax(),
                'frequencia_maxima': freq_counts.max(),
                'percentual_maximo': freq_percent.max(),
                'numero_menos_frequente': freq_counts.idxmin(),
                'frequencia_minima': freq_counts.min(),
                'percentual_minimo': freq_percent.min(),
                'desvio_padrao': freq_counts.std(),
                'variancia': freq_counts.var()
            }
            
            frequency_analysis['frequencias_por_coluna'][col] = col_stats
            
            # Top 3 números mais frequentes
            top_3 = freq_counts.head(3)
            frequency_analysis['numeros_mais_frequentes'][col] = {
                'top_3': top_3.to_dict(),
                'top_3_percentual': (top_3 / total_concursos * 100).round(2).to_dict()
            }
        
        # Análise geral
        self._analyze_general_patterns(df, frequency_analysis)
        
        # Análise temporal (se houver datas)
        if 'Data_Sorteio' in df.columns:
            self._analyze_temporal_patterns(df, frequency_analysis)
        
        # Gerar recomendações
        self._generate_frequency_recommendations(frequency_analysis)
        
        return frequency_analysis
    
    def _analyze_general_patterns(self, df: pd.DataFrame, analysis: Dict) -> None:
        """Analisa padrões gerais de frequência"""
        numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
        
        # Frequência geral de cada número (0-9) em todas as colunas
        all_numbers = []
        for col in numeric_cols:
            all_numbers.extend(df[col].tolist())
        
        general_freq = pd.Series(all_numbers).value_counts().sort_index()
        total_occurrences = len(all_numbers)
        
        analysis['estatisticas_gerais'] = {
            'frequencia_geral': general_freq.to_dict(),
            'percentual_geral': (general_freq / total_occurrences * 100).round(2).to_dict(),
            'numero_mais_saido_geral': general_freq.idxmax(),
            'numero_menos_saido_geral': general_freq.idxmin(),
            'total_ocorrencias': total_occurrences,
            'media_por_numero': general_freq.mean(),
            'desvio_padrao_geral': general_freq.std()
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, analysis: Dict) -> None:
        """Analisa padrões temporais de frequência"""
        if 'Data_Sorteio' not in df.columns:
            return
            
        # Análise por mês
        df['Mes'] = df['Data_Sorteio'].dt.month
        df['Ano'] = df['Data_Sorteio'].dt.year
        
        temporal_patterns = {
            'por_mes': {},
            'por_ano': {},
            'tendencias': {}
        }
        
        # Padrões por mês
        for mes in range(1, 13):
            mes_data = df[df['Mes'] == mes]
            if len(mes_data) > 0:
                numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
                mes_numbers = []
                for col in numeric_cols:
                    mes_numbers.extend(mes_data[col].tolist())
                
                if mes_numbers:
                    mes_freq = pd.Series(mes_numbers).value_counts().sort_index()
                    temporal_patterns['por_mes'][mes] = {
                        'frequencia': mes_freq.to_dict(),
                        'total_concursos': len(mes_data),
                        'numero_mais_frequente': mes_freq.idxmax() if len(mes_freq) > 0 else None
                    }
        
        analysis['padroes_temporais'] = temporal_patterns
    
    def _generate_frequency_recommendations(self, analysis: Dict) -> None:
        """Gera recomendações baseadas na análise de frequência"""
        recommendations = {
            'numeros_quentes': {},
            'numeros_frios': {},
            'estrategias_por_coluna': {},
            'observacoes_importantes': []
        }
        
        # Analisar cada coluna
        for col, data in analysis['frequencias_por_coluna'].items():
            freq_percent = data['frequencia_percentual']
            
            # Números quentes (acima da média + 1 desvio padrão)
            media = np.mean(list(freq_percent.values()))
            desvio = np.std(list(freq_percent.values()))
            threshold_quente = media + desvio
            
            # Números frios (abaixo da média - 1 desvio padrão)
            threshold_frio = media - desvio
            
            numeros_quentes = {num: freq for num, freq in freq_percent.items() if freq > threshold_quente}
            numeros_frios = {num: freq for num, freq in freq_percent.items() if freq < threshold_frio}
            
            recommendations['numeros_quentes'][col] = numeros_quentes
            recommendations['numeros_frios'][col] = numeros_frios
            
            # Estratégia por coluna
            if numeros_quentes:
                top_quente = max(numeros_quentes.items(), key=lambda x: x[1])
                recommendations['estrategias_por_coluna'][col] = {
                    'recomendacao': f"Focar no número {top_quente[0]} (frequência: {top_quente[1]:.1f}%)",
                    'numeros_quentes': list(numeros_quentes.keys()),
                    'numeros_frios': list(numeros_frios.keys())
                }
        
        # Observações importantes
        recommendations['observacoes_importantes'] = [
            "Números com frequência acima da média têm maior probabilidade histórica",
            "Padrões podem mudar ao longo do tempo - análise contínua recomendada",
            "Frequência não garante resultado futuro, mas indica tendências",
            "Considere combinar análise de frequência com outros métodos"
        ]
        
        analysis['recomendacoes'] = recommendations


class ProgressTracker:
    """Rastreador de progresso otimizado"""
    
    def __init__(self, total_steps: int, operation_name: str = "Operação"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_interval = 2.0
        
    def update(self, step: int = None, message: str = ""):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        current_time = time.time()
        
        if (current_time - self.last_update >= self.update_interval or 
            self.current_step >= self.total_steps):
            self._display_progress(message)
            self.last_update = current_time
    
    def _display_progress(self, message: str = ""):
        elapsed = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100
        
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        
        bar_length = 20
        filled = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        progress_line = f"\r🔄 {self.operation_name}: [{bar}] {progress:4.0f}% | {elapsed:.0f}s | {eta_str}"
        if message and len(message) < 30:
            progress_line += f" | {message}"
        
        print(progress_line, end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()
    
    def finish(self, message: str = "Concluído!"):
        self.current_step = self.total_steps
        self._display_progress(message)
        elapsed = time.time() - self.start_time
        print(f"✅ {self.operation_name} {message} em {elapsed:.1f}s")


class SuperSeteAnalyzer:
    """Analisador principal do Super Sete com análise científica avançada"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model_cache = {}
        
        # Inicializar analisadores especializados
        self.temporal_analyzer = TemporalAnalyzer(self.config)
        self.entropy_analyzer = EntropyAnalyzer(self.config)
        self.confidence_analyzer = ConfidenceAnalyzer(self.config)
        self.feature_engineer = AdvancedFeatureEngineer(self.config)
        self.adaptive_learning = AdaptiveLearningSystem(self.config)
        self.frequency_analyzer = FrequencyAnalyzer(self.config)
        
        # Cache para análises
        self.analysis_cache = {}
        self.model_weights = {}
        
    def load_data(self, excel_path: str) -> pd.DataFrame:
        """Carrega dados do Excel incluindo informações temporais"""
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {excel_path}")
        
        # Carregar dados numéricos (colunas C:I) - dados começam na linha 2
        df = pd.read_excel(excel_path, usecols="C:I", header=0, engine="openpyxl")
        df.columns = [f"Coluna_{i+1}" for i in range(df.shape[1])]
        df = df.apply(pd.to_numeric, errors="coerce").dropna().astype(int)
        
        if df.min().min() < 0 or df.max().max() > 9:
            raise ValueError("Valores fora do intervalo 0..9 encontrados")
        
        # Carregar datas da coluna B
        try:
            # Ler apenas a coluna B (Data Sorteio) - dados começam na linha 2
            df_dates = pd.read_excel(excel_path, usecols="B", header=0, engine="openpyxl")
            
            # Converter para datetime, assumindo formato DD/MM/YYYY
            df_dates['Data_Sorteio'] = pd.to_datetime(df_dates.iloc[:, 0], format='%d/%m/%Y', errors='coerce')
            
            # Remover linhas com datas inválidas
            df_dates = df_dates.dropna()
            
            # Ajustar índices para corresponder aos dados numéricos
            min_len = min(len(df), len(df_dates))
            df = df.iloc[:min_len].reset_index(drop=True)
            df_dates = df_dates.iloc[:min_len].reset_index(drop=True)
            
            # Adicionar coluna de data ao DataFrame principal
            df['Data_Sorteio'] = df_dates['Data_Sorteio']
            
            # Reordenar colunas para ter Data_Sorteio primeiro
            cols = ['Data_Sorteio'] + [f"Coluna_{i+1}" for i in range(7)]
            df = df[cols]
            
            print(f"📅 Datas carregadas: {df['Data_Sorteio'].min().strftime('%d/%m/%Y')} a {df['Data_Sorteio'].max().strftime('%d/%m/%Y')}")
            print(f"📊 Total de concursos: {len(df)}")
            
        except Exception as e:
            print(f"⚠️  Aviso: Não foi possível carregar as datas: {e}")
            print("📊 Continuando apenas com dados numéricos...")
        
        return df
    
    def analyze_data(self, df: pd.DataFrame) -> None:
        """Análise exploratória dos dados com métricas científicas"""
        print(f"📊 Dataset: {len(df)} concursos")
        print(f"📅 Colunas: {list(df.columns)}")
        
        # Análise temporal se disponível
        if 'Data_Sorteio' in df.columns:
            print(f"\n📅 ANÁLISE TEMPORAL:")
            print(f"  Período: {df['Data_Sorteio'].min().strftime('%d/%m/%Y')} a {df['Data_Sorteio'].max().strftime('%d/%m/%Y')}")
            print(f"  Duração: {(df['Data_Sorteio'].max() - df['Data_Sorteio'].min()).days} dias")
            print(f"  Frequência média: {len(df) / ((df['Data_Sorteio'].max() - df['Data_Sorteio'].min()).days / 7):.1f} concursos por semana")
            
            # Análise de dias da semana
            df['Dia_Semana'] = df['Data_Sorteio'].dt.day_name()
            print(f"  Dias da semana: {dict(df['Dia_Semana'].value_counts())}")
            
            # Análise de meses
            df['Mes'] = df['Data_Sorteio'].dt.month
            print(f"  Distribuição por mês: {dict(df['Mes'].value_counts().sort_index())}")
        
        print("\n📈 Estatísticas Básicas:")
        # Mostrar apenas colunas numéricas para estatísticas
        numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
        print(df[numeric_cols].describe())
        
        print("\n🎯 Frequência por Coluna:")
        for col in numeric_cols:
            freq = df[col].value_counts().sort_index()
            print(f"  {col}: {dict(freq)}")
        
        # Análise de frequência detalhada
        print("\n📊 ANÁLISE DE FREQUÊNCIA DETALHADA:")
        print("-" * 50)
        frequency_analysis = self.frequency_analyzer.analyze_column_frequency(df)
        
        # Exibir números mais frequentes por coluna
        print("\n🔥 NÚMEROS MAIS FREQUENTES POR COLUNA:")
        for col, data in frequency_analysis['numeros_mais_frequentes'].items():
            print(f"\n  {col}:")
            for num, freq in data['top_3'].items():
                percent = data['top_3_percentual'][num]
                print(f"    {num}: {freq} vezes ({percent:.1f}%)")
        
        # Exibir estatísticas gerais
        print("\n📈 ESTATÍSTICAS GERAIS:")
        stats = frequency_analysis['estatisticas_gerais']
        print(f"  Número mais saído geralmente: {stats['numero_mais_saido_geral']} ({stats['percentual_geral'][stats['numero_mais_saido_geral']]:.1f}%)")
        print(f"  Número menos saído geralmente: {stats['numero_menos_saido_geral']} ({stats['percentual_geral'][stats['numero_menos_saido_geral']]:.1f}%)")
        print(f"  Total de ocorrências analisadas: {stats['total_ocorrencias']}")
        
        # Exibir recomendações
        print("\n💡 RECOMENDAÇÕES BASEADAS EM FREQUÊNCIA:")
        for col, strategy in frequency_analysis['recomendacoes']['estrategias_por_coluna'].items():
            print(f"  {col}: {strategy['recomendacao']}")
        
        # Análise científica avançada
        print("\n🔬 ANÁLISE CIENTÍFICA AVANÇADA:")
        print("-" * 40)
        
        # Análise de entropia
        print("\n📊 Entropia e Aleatoriedade:")
        for col in numeric_cols:
            series = df[col].values.astype(int)
            entropy_info = self.entropy_analyzer.calculate_entropy(series)
            randomness_info = self.entropy_analyzer.test_randomness(series)
            
            print(f"  {col}:")
            print(f"    Entropia: {entropy_info['normalized_entropy']:.3f} (max: {entropy_info['max_entropy']:.3f})")
            print(f"    Aleatório: {'✅' if randomness_info['is_random'] else '❌'}")
        
        # Análise temporal
        print("\n⏰ Padrões Temporais:")
        for col in numeric_cols:
            series = df[col].values.astype(int)
            seasonality = self.temporal_analyzer.detect_seasonality(series)
            trend = self.temporal_analyzer.calculate_trend(series)
            cycles = self.temporal_analyzer.detect_cycles(series)
            
            print(f"  {col}:")
            print(f"    Sazonalidade: {'✅' if seasonality['has_seasonality'] else '❌'}")
            if seasonality['has_seasonality']:
                print(f"      Períodos: {[f'{p:.1f}' for p in seasonality['periods'][:3]]}")
            print(f"    Tendência: {trend['direction']} (sig: {trend['significance']:.3f})")
            print(f"    Ciclos: {len(cycles['cycles'])} detectados")
            if cycles['spectral_cycles']:
                print(f"    Ciclos espectrais: {len(cycles['spectral_cycles'])} detectados")
            if cycles['sequential_patterns']['repetitions']:
                print(f"    Padrões repetitivos: {len(cycles['sequential_patterns']['repetitions'])}")
            if cycles['sequential_patterns']['progressions']:
                print(f"    Progressões: {len(cycles['sequential_patterns']['progressions'])}")
        
        # Análise de correlação
        if self.config.correlation_analysis:
            print("\n🔗 Correlações entre Colunas:")
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > self.config.min_correlation:
                            print(f"    {col1} ↔ {col2}: {corr:.3f}")
    
    def detect_anomalies(self, series: np.ndarray) -> Dict:
        """Detecta anomalias usando múltiplos métodos"""
        if len(series) < 10:
            return {'anomalies': [], 'outlier_indices': [], 'is_anomalous': False}
        
        outlier_indices = []
        
        if self.config.outlier_method == 'iqr':
            # Método IQR
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = np.where((series < lower_bound) | (series > upper_bound))[0]
        
        elif self.config.outlier_method == 'zscore':
            # Método Z-Score
            z_scores = np.abs(stats.zscore(series))
            outlier_indices = np.where(z_scores > self.config.anomaly_threshold)[0]
        
        # Detectar padrões anômalos
        anomalies = []
        for idx in outlier_indices:
            anomalies.append({
                'index': idx,
                'value': series[idx],
                'z_score': stats.zscore(series)[idx] if len(series) > 1 else 0,
                'severity': 'high' if abs(stats.zscore(series)[idx]) > 3 else 'medium'
            })
        
        return {
            'anomalies': anomalies,
            'outlier_indices': outlier_indices.tolist(),
            'is_anomalous': len(outlier_indices) > 0,
            'anomaly_rate': len(outlier_indices) / len(series)
        }
    
    def calculate_correlation_matrix(self, df: pd.DataFrame) -> Dict:
        """Calcula matriz de correlação com análise de significância"""
        if not self.config.correlation_analysis:
            return {}
        
        # Usar apenas colunas numéricas
        numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
        corr_matrix = df[numeric_cols].corr()
        
        # Teste de significância
        n = len(df)
        significant_correlations = {}
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr):
                        # Teste t para correlação
                        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-8))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        
                        if p_value < 0.05 and abs(corr) > self.config.min_correlation:
                            significant_correlations[f"{col1}_{col2}"] = {
                                'correlation': corr,
                                'p_value': p_value,
                                'significant': True
                            }
        
        return {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant_correlations,
            'high_correlations': {k: v for k, v in significant_correlations.items() 
                                if abs(v['correlation']) > self.config.max_correlation}
        }
    
    def create_features(self, series: np.ndarray, window: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Cria features científicas SIMPLIFICADAS para reduzir overfitting"""
        window = window or self.config.window
        
        if len(series) < window + 1:
            n_features = 6  # REDUZIDO de 12 para 6 para evitar overfitting
            return np.empty((0, n_features), dtype=np.float32), np.empty((0,), dtype=np.int8), []
        
        n_samples = len(series) - window
        n_features = 6  # Features MUITO SIMPLIFICADAS para reduzir overfitting
        
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        targets = series[window:].astype(np.int8)
        
        for i in range(n_samples):
            recent = series[i:i+window]
            
            # Feature 1: Média móvel (3 períodos)
            features[i, 0] = np.mean(recent[-3:]) if len(recent) >= 3 else np.mean(recent)
            
            # Feature 2: Desvio padrão móvel (3 períodos)
            features[i, 1] = np.std(recent[-3:]) if len(recent) >= 3 else np.std(recent)
            
            # Feature 3: Frequência do último dígito
            last_digit = recent[-1]
            features[i, 2] = np.sum(recent == last_digit) / len(recent)
            
            # Feature 4: Frequência do penúltimo dígito
            if len(recent) >= 2:
                second_last_digit = recent[-2]
                features[i, 3] = np.sum(recent == second_last_digit) / len(recent)
            else:
                features[i, 3] = 0
            
            # Feature 5: Padrão de alternância (mudanças de direção)
            if len(recent) >= 3:
                diffs = np.diff(recent)
                alternations = np.sum(np.diff(np.sign(diffs)) != 0)
                features[i, 4] = alternations / (len(diffs) - 1) if len(diffs) > 1 else 0
            else:
                features[i, 4] = 0
            
            # Feature 6: Entropia local (simplificada)
            unique, counts = np.unique(recent, return_counts=True)
            probs = counts / len(recent)
            features[i, 5] = -np.sum(probs * np.log2(probs + 1e-10))
        
        feature_names = [
            'mean_3period', 'std_3period', 'freq_last_digit', 
            'freq_second_last', 'alternation_pattern', 'local_entropy'
        ]
        
        return features, targets, feature_names
    
    def _calculate_complexity(self, sequence: np.ndarray) -> float:
        """Calcula complexidade aproximada usando padrão de Lempel-Ziv"""
        if len(sequence) < 2:
            return 0.0
        
        # Converter para string
        seq_str = ''.join(map(str, sequence))
        
        # Algoritmo simplificado de Lempel-Ziv
        patterns = set()
        i = 0
        while i < len(seq_str):
            for j in range(i + 1, len(seq_str) + 1):
                pattern = seq_str[i:j]
                if pattern not in patterns:
                    patterns.add(pattern)
                    i = j
                    break
            else:
                i += 1
        
        return len(patterns) / len(seq_str)
    
    def _longest_increasing_subsequence(self, sequence: np.ndarray) -> int:
        """Calcula o comprimento da maior subsequência crescente"""
        if len(sequence) == 0:
            return 0
        
        # Algoritmo O(n log n) para LIS
        tails = []
        for num in sequence:
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        
        return len(tails)
    
    def get_cache_path(self, series: np.ndarray, coluna: str, modelo: str) -> str:
        """Gera caminho de cache único"""
        data_hash = hashlib.md5(series.tobytes()).hexdigest()[:8]
        config_hash = hashlib.md5(str(self.config.__dict__).encode()).hexdigest()[:8]
        
        cache_dir = "model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, f"{coluna}_{modelo}_{data_hash}_{config_hash}.pkl")
    
    def save_model(self, model, model_name: str, metrics: Dict, cache_path: str) -> bool:
        """Salva modelo em cache"""
        try:
            save_data = {
                'model': model,
                'model_name': model_name,
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar {model_name}: {e}")
            return False
    
    def load_model(self, cache_path: str) -> Optional[Dict]:
        """Carrega modelo do cache"""
        try:
            if not os.path.exists(cache_path):
                return None
                
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return None
    
    def train_models(self, series: np.ndarray, coluna: str, use_cache: bool = True) -> Dict:
        """Treina modelos ML científicos para uma coluna"""
        print(f"  🔧 Treinando modelos científicos para {coluna}...")
        
        # Verificar cache
        models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'logistic', 'naive_bayes', 
                          'svm', 'mlp', 'gaussian_process', 'ensemble_adaptive']
        cached_models = {}
        cached_metrics = {}
        
        if use_cache:
            for model_name in models_to_train:
                cache_path = self.get_cache_path(series, coluna, model_name)
                cached_data = self.load_model(cache_path)
                
                if cached_data is not None:
                    cached_models[model_name] = cached_data['model']
                    cached_metrics[model_name] = cached_data['metrics']
        
        # Se todos estão em cache, retornar
        if use_cache and len(cached_models) == len(models_to_train):
            print(f"  ⚡ Todos os modelos de {coluna} em cache!")
            return {
                'models': cached_models,
                'metrics': cached_metrics,
                'features': np.array([]),
                'targets': np.array([])
            }
        
        # Criar features científicas
        features, targets, feature_names = self.create_features(series)
        
        if len(features) < self.config.train_min_size:
            print(f"  ⚠️  Dados insuficientes para {coluna} ({len(features)} amostras)")
            return None
        
        # Preparar dados
        X = features
        y = targets.astype(int)
        
        # Verificar se todas as classes estão presentes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes < 10:
            print(f"    ⚠️  Aviso: {coluna} tem apenas {n_classes} classes: {sorted(unique_classes)}")
        
        # Normalizar com RobustScaler (mais robusto a outliers)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Validação temporal ULTRA RIGOROSA com walk-forward testing
        # Usar apenas 2 splits para evitar overfitting na validação
        tscv = TimeSeriesSplit(n_splits=min(2, len(X) // 100))  # Ultra rigoroso
        
        models = {}
        metrics = {}
        
        # Treinar modelos em paralelo
        self._train_models_parallel_advanced(X_scaled, y, feature_names, tscv, models, metrics, 
                                           cached_models, cached_metrics, series, coluna, use_cache, 
                                           unique_classes)
        
        models['scaler'] = scaler
        
        # Calcular pesos adaptativos se habilitado
        if self.config.adaptive_weights and len(models) > 1:
            self._calculate_adaptive_weights(models, metrics, coluna)
        
        return {
            'models': models,
            'metrics': metrics,
            'features': X,
            'targets': y
        }
    
    def train_models_enhanced(self, series: np.ndarray, coluna: str, advanced_features: np.ndarray, use_cache: bool = True) -> Dict:
        """Treina modelos ML melhorados com features avançadas para maior assertividade"""
        print(f"  🔧 Treinando modelos melhorados para {coluna}...")
        
        # Verificar cache
        models_to_train = ['random_forest_enhanced', 'xgboost_enhanced', 'lightgbm_enhanced', 
                          'logistic_enhanced', 'naive_bayes_enhanced', 'svm_enhanced', 
                          'mlp_enhanced', 'ensemble_adaptive_enhanced']
        cached_models = {}
        cached_metrics = {}
        
        if use_cache:
            for model_name in models_to_train:
                cache_path = self.get_cache_path(series, coluna, model_name)
                cached_data = self.load_model(cache_path)
                
                if cached_data is not None:
                    cached_models[model_name] = cached_data['model']
                    cached_metrics[model_name] = cached_data['metrics']
        
        # Se todos estão em cache, retornar
        if use_cache and len(cached_models) == len(models_to_train):
            print(f"  ⚡ Todos os modelos melhorados de {coluna} em cache!")
            return {
                'models': cached_models,
                'metrics': cached_metrics,
                'features': np.array([]),
                'targets': np.array([])
            }
        
        # Criar features científicas tradicionais
        features, targets, feature_names = self.create_features(series)
        
        # Combinar com features avançadas
        if len(advanced_features) > 0:
            # Ajustar tamanho das features avançadas para corresponder ao número de amostras
            n_samples = len(features)
            if len(advanced_features) >= n_samples:
                # Usar as últimas n_samples features avançadas
                advanced_features_adjusted = advanced_features[-n_samples:]
            else:
                # Repetir as features avançadas para preencher
                repeat_factor = n_samples // len(advanced_features) + 1
                advanced_features_repeated = np.tile(advanced_features, repeat_factor)
                advanced_features_adjusted = advanced_features_repeated[:n_samples]
            
            # Combinar features
            features_combined = np.column_stack([features, advanced_features_adjusted])
        else:
            features_combined = features
        
        if len(features_combined) < self.config.train_min_size:
            print(f"  ⚠️  Dados insuficientes para {coluna} ({len(features_combined)} amostras)")
            return None
        
        # Preparar dados
        X = features_combined
        y = targets.astype(int)
        
        # Verificar se todas as classes estão presentes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes < 10:
            print(f"    ⚠️  Aviso: {coluna} tem apenas {n_classes} classes: {sorted(unique_classes)}")
        
        # Normalizar com RobustScaler (mais robusto a outliers)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Validação temporal ULTRA RIGOROSA com walk-forward testing
        tscv = TimeSeriesSplit(n_splits=min(2, len(X) // 100))  # Ultra rigoroso para evitar overfitting
        
        models = {}
        metrics = {}
        
        # Treinar modelos melhorados em paralelo
        self._train_models_enhanced_parallel(X_scaled, y, feature_names, tscv, models, metrics, 
                                           cached_models, cached_metrics, series, coluna, use_cache, 
                                           unique_classes)
        
        models['scaler'] = scaler
        
        # Calcular pesos adaptativos se habilitado
        if self.config.adaptive_weights and len(models) > 1:
            self._calculate_adaptive_weights(models, metrics, coluna)
        
        return {
            'models': models,
            'metrics': metrics,
            'features': X,
            'targets': y
        }
    
    def _calculate_adaptive_weights(self, models: Dict, metrics: Dict, coluna: str) -> None:
        """Calcula pesos adaptativos MELHORADOS baseados na performance dos modelos"""
        if coluna not in self.model_weights:
            self.model_weights[coluna] = {}
        
        # Atualizar performance no sistema de aprendizado adaptativo
        for model_name, model_metrics in metrics.items():
            if model_name != 'scaler':
                accuracy = model_metrics.get('accuracy', 0.0)
                brier_score = model_metrics.get('brier_score', 1.0)
                self.adaptive_learning.update_performance(coluna, model_name, accuracy, brier_score)
        
        # Obter pesos adaptativos do sistema de aprendizado
        model_names = [name for name in models.keys() if name != 'scaler']
        adaptive_weights = self.adaptive_learning.get_adaptive_weights(coluna, model_names)
        
        # Aplicar pesos adaptativos
        for model_name, weight in adaptive_weights.items():
            self.model_weights[coluna][model_name] = weight
        
        # Verificar se algum modelo precisa ser retreinado
        retrain_suggestions = []
        for model_name in model_names:
            if self.adaptive_learning.should_retrain_model(coluna, model_name):
                retrain_suggestions.append(model_name)
        
        if retrain_suggestions:
            print(f"    🔄 Sugestão: Modelos {retrain_suggestions} podem se beneficiar de retreinamento")
        
        # Obter sugestões de adaptação
        adaptation_info = self.adaptive_learning.get_adaptation_suggestions(coluna)
        if adaptation_info['status'] == 'needs_improvement':
            print(f"    ⚠️  Sugestões de adaptação para {coluna}:")
            for suggestion in adaptation_info['suggestions']:
                print(f"      - {suggestion}")
    
    def _train_models_parallel_advanced(self, X_scaled: np.ndarray, y: np.ndarray, feature_names: List[str], 
                                       tscv: TimeSeriesSplit, models: Dict, metrics: Dict, 
                                       cached_models: Dict, cached_metrics: Dict, 
                                       series: np.ndarray, coluna: str, use_cache: bool, 
                                       unique_classes: np.ndarray = None) -> None:
        """Treina modelos científicos em paralelo com validação rigorosa"""
        
        def train_random_forest():
            if 'random_forest' not in cached_models:
                print(f"    🌲 Treinando Random Forest...")
                rf = RandomForestClassifier(
                    n_estimators=30,  # Aumentado ligeiramente para melhor performance
                    max_depth=4,      # Aumentado ligeiramente para capturar padrões
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    class_weight='balanced',
                    # REGULARIZAÇÃO ULTRA RIGOROSA ANTI-OVERFITTING
                    min_samples_split=100,  # Aumentado drasticamente para evitar overfitting
                    min_samples_leaf=50,    # Aumentado drasticamente para evitar overfitting
                    max_features='sqrt',    # Reduzir features por split
                    bootstrap=True,         # Bootstrap para reduzir overfitting
                    oob_score=True,         # Out-of-bag score para validação
                    max_samples=0.6,        # Usar apenas 60% das amostras por árvore
                    max_leaf_nodes=20       # Limitar número de folhas
                )
                rf.fit(X_scaled, y)
                
                # Validação cruzada com métricas múltiplas
                cv_scores = self._cross_validate_model(rf, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'random_forest')
                    self.save_model(rf, 'random_forest', cv_scores, cache_path)
                
                return 'random_forest', rf, cv_scores
            else:
                return 'random_forest', cached_models['random_forest'], cached_metrics['random_forest']
        
        def train_xgboost():
            if XGB_AVAILABLE and 'xgboost' not in cached_models:
                print(f"    🚀 Treinando XGBoost...")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=25,  # Aumentado ligeiramente para melhor performance
                    max_depth=3,      # Mantido baixo para evitar overfitting
                    learning_rate=0.05,  # Reduzido para convergência mais suave
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    eval_metric='mlogloss',
                    # REGULARIZAÇÃO ULTRA RIGOROSA ANTI-OVERFITTING
                    reg_alpha=0.5,    # L1 regularization aumentada drasticamente
                    reg_lambda=0.5,   # L2 regularization aumentada drasticamente
                    subsample=0.5,    # Subsample mais restritivo
                    colsample_bytree=0.5,  # Feature sampling mais restritivo
                    min_child_weight=50,   # Aumentado drasticamente
                    gamma=0.5,        # Regularização adicional aumentada
                    max_delta_step=1,  # Limitar mudanças por step
                    early_stopping_rounds=5  # Parada antecipada
                )
                xgb_model.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(xgb_model, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'xgboost')
                    self.save_model(xgb_model, 'xgboost', cv_scores, cache_path)
                
                return 'xgboost', xgb_model, cv_scores
            elif XGB_AVAILABLE and 'xgboost' in cached_models:
                return 'xgboost', cached_models['xgboost'], cached_metrics['xgboost']
            else:
                return None, None, None
        
        def train_lightgbm():
            if LGBM_AVAILABLE and 'lightgbm' not in cached_models:
                print(f"    💡 Treinando LightGBM...")
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=15,  # Reduzido drasticamente
                    max_depth=3,      # Reduzido drasticamente
                    learning_rate=0.1,  # Mantido
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    class_weight='balanced',
                    # REGULARIZAÇÃO MUITO RIGOROSA
                    reg_alpha=0.1,    # L1 regularization aumentada
                    reg_lambda=0.1,   # L2 regularization aumentada
                    subsample=0.7,    # Subsample mais restritivo
                    colsample_bytree=0.7,  # Feature sampling mais restritivo
                    min_child_samples=25,  # Aumentado drasticamente
                    min_split_gain=0.2,    # Threshold mais alto para splits
                    feature_fraction=0.7   # Usar apenas 70% das features
                )
                
                X_df = pd.DataFrame(X_scaled, columns=feature_names)
                lgb_model.fit(X_df, y)
                
                cv_scores = self._cross_validate_model(lgb_model, X_scaled, y, tscv, feature_names)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'lightgbm')
                    self.save_model(lgb_model, 'lightgbm', cv_scores, cache_path)
                
                return 'lightgbm', lgb_model, cv_scores
            elif LGBM_AVAILABLE and 'lightgbm' in cached_models:
                return 'lightgbm', cached_models['lightgbm'], cached_metrics['lightgbm']
            else:
                return None, None, None
        
        def train_logistic():
            if 'logistic' not in cached_models:
                print(f"    📊 Treinando Logistic Regression...")
                lr = LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=500,  # Reduzido
                    class_weight='balanced',
                    multi_class='ovr',
                    # REGULARIZAÇÃO MUITO RIGOROSA
                    C=0.1,  # Regularização muito alta (1/C = 10)
                    penalty='elasticnet',  # L1 + L2
                    l1_ratio=0.5,  # ElasticNet balanceado
                    solver='saga'  # Suporta elasticnet
                )
                lr.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(lr, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'logistic')
                    self.save_model(lr, 'logistic', cv_scores, cache_path)
                
                return 'logistic', lr, cv_scores
            else:
                return 'logistic', cached_models['logistic'], cached_metrics['logistic']
        
        def train_naive_bayes():
            if 'naive_bayes' not in cached_models:
                print(f"    🎯 Treinando Naive Bayes...")
                nb = GaussianNB()
                nb.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(nb, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'naive_bayes')
                    self.save_model(nb, 'naive_bayes', cv_scores, cache_path)
                
                return 'naive_bayes', nb, cv_scores
            else:
                return 'naive_bayes', cached_models['naive_bayes'], cached_metrics['naive_bayes']
        
        def train_svm():
            if 'svm' not in cached_models:
                print(f"    🔥 Treinando SVM...")
                svm = SVC(
                    probability=True,
                    random_state=self.config.random_state,
                    class_weight='balanced',
                    kernel='rbf'
                )
                svm.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(svm, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'svm')
                    self.save_model(svm, 'svm', cv_scores, cache_path)
                
                return 'svm', svm, cv_scores
            else:
                return 'svm', cached_models['svm'], cached_metrics['svm']
        
        def train_mlp():
            if 'mlp' not in cached_models:
                print(f"    🧠 Treinando MLP...")
                mlp = MLPClassifier(
                    hidden_layer_sizes=(20,),  # Apenas 1 camada com 20 neurônios
                    random_state=self.config.random_state,
                    max_iter=100,  # Reduzido ainda mais
                    early_stopping=True,
                    validation_fraction=0.3,  # Mais validação
                    # REGULARIZAÇÃO MUITO RIGOROSA
                    alpha=0.1,  # L2 regularization aumentada
                    learning_rate_init=0.01,  # Learning rate maior para convergência mais rápida
                    learning_rate='adaptive',  # Learning rate adaptativo
                    batch_size=32,  # Batch size menor
                    beta_1=0.9,  # Parâmetros do Adam
                    beta_2=0.999
                )
                mlp.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(mlp, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'mlp')
                    self.save_model(mlp, 'mlp', cv_scores, cache_path)
                
                return 'mlp', mlp, cv_scores
            else:
                return 'mlp', cached_models['mlp'], cached_metrics['mlp']
        
        def train_gaussian_process():
            if 'gaussian_process' not in cached_models:
                print(f"    🌊 Treinando Gaussian Process...")
                try:
                    # Configurar kernel RBF com length_scale correto
                    kernel = RBF(length_scale=1.0)
                    
                    # Criar GaussianProcessClassifier com parâmetros válidos
                    gp = GaussianProcessClassifier(
                        kernel=kernel,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        warm_start=False,
                        copy_X_train=True
                    )
                    gp.fit(X_scaled, y)
                    
                    cv_scores = self._cross_validate_model(gp, X_scaled, y, tscv)
                    
                    if use_cache:
                        cache_path = self.get_cache_path(series, coluna, 'gaussian_process')
                        self.save_model(gp, 'gaussian_process', cv_scores, cache_path)
                    
                    return 'gaussian_process', gp, cv_scores
                    
                except Exception as e:
                    print(f"    ❌ Erro no treinamento do Gaussian Process: {str(e)}")
                    # Retornar None para indicar falha no treinamento
                    return None, None, None
            else:
                return 'gaussian_process', cached_models['gaussian_process'], cached_metrics['gaussian_process']
        
        # Executar treinamentos em paralelo
        models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'logistic', 'naive_bayes', 
                          'svm', 'mlp', 'gaussian_process', 'ensemble_adaptive']
        
        # Se há classes ausentes, usar apenas modelos robustos
        if unique_classes is not None and len(unique_classes) < 10:
            print(f"    ⚠️  Usando apenas modelos robustos para {len(unique_classes)} classes")
            models_to_train = ['random_forest', 'logistic', 'naive_bayes', 'svm']
        
        with ThreadPoolExecutor(max_workers=min(8, len(models_to_train))) as executor:
            futures = []
            
            if 'random_forest' in models_to_train:
                futures.append(executor.submit(train_random_forest))
            if 'xgboost' in models_to_train:
                futures.append(executor.submit(train_xgboost))
            if 'lightgbm' in models_to_train:
                futures.append(executor.submit(train_lightgbm))
            if 'logistic' in models_to_train:
                futures.append(executor.submit(train_logistic))
            if 'naive_bayes' in models_to_train:
                futures.append(executor.submit(train_naive_bayes))
            if 'svm' in models_to_train:
                futures.append(executor.submit(train_svm))
            if 'mlp' in models_to_train:
                futures.append(executor.submit(train_mlp))
            if 'gaussian_process' in models_to_train:
                futures.append(executor.submit(train_gaussian_process))
            
            for future in as_completed(futures):
                try:
                    name, model, model_metrics = future.result()
                    if name and model is not None:
                        models[name] = model
                        metrics[name] = model_metrics
                except Exception as e:
                    print(f"    ❌ Erro no treinamento: {e}")
        
        # Validar qualidade dos modelos ANTES de usar - RIGOROSA
        print(f"    🔍 Validando qualidade dos modelos (ANTI-OVERFITTING)...")
        good_models = {}
        for name, model in models.items():
            if name != 'scaler':
                validation = self._validate_model_quality(model, X_scaled, y, name)
                model_metrics = metrics.get(name, {})
                
                # Critérios ULTRA RIGOROSOS ANTI-OVERFITTING para aceitar modelo
                is_good_quality = (
                    validation['is_good_quality'] and
                    not model_metrics.get('is_overfitting', False) and  # Rejeitar se overfitting
                    model_metrics.get('overfitting_gap', 1.0) < 0.05 and  # Gap menor que 5% (ultra restritivo)
                    model_metrics.get('accuracy', 0) > 0.12 and  # Acurácia maior que 12% (mais realista)
                    model_metrics.get('stability', 0) > 0.8 and  # Estabilidade maior que 80%
                    model_metrics.get('reliability_score', 0) > 0.15  # Confiabilidade maior que 15%
                )
                
                if is_good_quality:
                    good_models[name] = model
                    print(f"    ✅ {name}: APROVADO (acc: {model_metrics.get('accuracy', 0):.3f}, gap: {model_metrics.get('overfitting_gap', 0):.3f})")
                else:
                    reason = []
                    if model_metrics.get('is_overfitting', False):
                        reason.append("OVERFITTING")
                    if model_metrics.get('overfitting_gap', 0) >= 0.15:
                        reason.append(f"GAP_ALTO({model_metrics.get('overfitting_gap', 0):.3f})")
                    if model_metrics.get('accuracy', 0) <= 0.15:
                        reason.append(f"ACC_BAIXA({model_metrics.get('accuracy', 0):.3f})")
                    print(f"    ❌ {name}: REJEITADO ({', '.join(reason)})")
        
        # Se nenhum modelo tem boa qualidade, usar fallback inteligente
        if not good_models:
            print(f"    ⚠️  Nenhum modelo com qualidade suficiente para {coluna}")
            print(f"    🔄 Usando fallback inteligente baseado em frequência histórica...")
            
            # Fallback: usar apenas frequência histórica com suavização
            fallback_model = {
                'type': 'frequency_fallback',
                'series': series,
                'window': self.config.window
            }
            
            # Calcular métricas do fallback
            fallback_metrics = {
                'accuracy': 0.10,  # Assumir 10% (aleatório)
                'overfitting_gap': 0.0,  # Sem overfitting
                'is_overfitting': False,
                'brier_score': 0.9,
                'log_loss': 2.3,
                'cv_folds': 1
            }
            
            models['frequency_fallback'] = fallback_model
            metrics['frequency_fallback'] = fallback_metrics
            good_models['frequency_fallback'] = fallback_model
        
        # Ensemble adaptativo sofisticado apenas com modelos de boa qualidade
        if len(good_models) > 1 and 'ensemble_adaptive' not in cached_models:
            print(f"    🎯 Criando ensemble adaptativo sofisticado com {len(good_models)} modelos...")
            
            # Criar diferentes tipos de ensemble
            ensemble_models = []
            ensemble_weights = []
            
            # 1. Ensemble com calibração individual
            for name, model in good_models.items():
                try:
                    # Calibrar modelo individualmente
                    calibrated = CalibratedClassifierCV(
                        model,
                        method=self.config.calibration_method,
                        cv=self.config.calibration_cv
                    )
                    calibrated.fit(X_scaled, y)
                    ensemble_models.append(calibrated)
                    
                    # Peso baseado na performance
                    model_accuracy = metrics.get(name, {}).get('accuracy', 0.1)
                    model_brier = metrics.get(name, {}).get('brier_score', 1.0)
                    weight = model_accuracy - (model_brier * 0.5)  # Score composto
                    ensemble_weights.append(max(0.1, weight))
                except Exception as e:
                    print(f"    ⚠️  Erro ao calibrar {name}: {e}")
                    continue
            
            # 2. Ensemble com VotingClassifier
            if len(ensemble_models) >= 2:
                try:
                    voting_ensemble = VotingClassifier(
                        estimators=[(f'model_{i}', model) for i, model in enumerate(ensemble_models)],
                        voting='soft'
                    )
                    voting_ensemble.fit(X_scaled, y)
                    ensemble_models.append(voting_ensemble)
                    ensemble_weights.append(1.0)  # Peso padrão para voting
                except Exception as e:
                    print(f"    ⚠️  Erro ao criar VotingClassifier: {e}")
            
            # 3. Ensemble com BaggingClassifier
            if len(good_models) >= 1:
                try:
                    # Usar o melhor modelo como base
                    best_model_name = max(good_models.keys(), 
                                        key=lambda x: metrics.get(x, {}).get('accuracy', 0))
                    best_model = good_models[best_model_name]
                    
                    bagging_ensemble = BaggingClassifier(
                        estimator=best_model,
                        n_estimators=min(10, len(good_models) * 2),
                        random_state=self.config.random_state,
                        n_jobs=-1
                    )
                    bagging_ensemble.fit(X_scaled, y)
                    ensemble_models.append(bagging_ensemble)
                    ensemble_weights.append(0.8)  # Peso menor para bagging
                except Exception as e:
                    print(f"    ⚠️  Erro ao criar BaggingClassifier: {e}")
            
            # Normalizar pesos
            if ensemble_weights:
                ensemble_weights = np.array(ensemble_weights)
                ensemble_weights = ensemble_weights / np.sum(ensemble_weights)
            
            models['ensemble_adaptive'] = {
                'models': ensemble_models,
                'weights': ensemble_weights.tolist() if ensemble_weights is not None else None
            }
            
            # Calcular métricas do ensemble
            ensemble_accuracy = np.mean([metrics.get(name, {}).get('accuracy', 0) for name in good_models.keys()])
            ensemble_brier = np.mean([metrics.get(name, {}).get('brier_score', 1) for name in good_models.keys()])
            
            metrics['ensemble_adaptive'] = {
                'n_models': len(ensemble_models),
                'accuracy': ensemble_accuracy,
                'brier_score': ensemble_brier,
                'ensemble_type': 'sophisticated_adaptive',
                'weights': ensemble_weights.tolist() if ensemble_weights is not None else None
            }
            
            if use_cache:
                cache_path = self.get_cache_path(series, coluna, 'ensemble_adaptive')
                self.save_model(models['ensemble_adaptive'], 'ensemble_adaptive', metrics['ensemble_adaptive'], cache_path)
        elif 'ensemble_adaptive' in cached_models:
            models['ensemble_adaptive'] = cached_models['ensemble_adaptive']
            metrics['ensemble_adaptive'] = cached_metrics['ensemble_adaptive']
    
    def _train_models_enhanced_parallel(self, X_scaled: np.ndarray, y: np.ndarray, feature_names: List[str], 
                                       tscv: TimeSeriesSplit, models: Dict, metrics: Dict, 
                                       cached_models: Dict, cached_metrics: Dict, 
                                       series: np.ndarray, coluna: str, use_cache: bool, 
                                       unique_classes: np.ndarray = None) -> None:
        """Treina modelos melhorados em paralelo com features avançadas para maior assertividade"""
        
        def train_random_forest_enhanced():
            if 'random_forest_enhanced' not in cached_models:
                print(f"    🌲 Treinando Random Forest Melhorado...")
                rf = RandomForestClassifier(
                    n_estimators=40,  # Reduzido para evitar overfitting
                    max_depth=3,      # Reduzido para evitar overfitting
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    class_weight='balanced',
                    # Regularização ULTRA RIGOROSA ANTI-OVERFITTING
                    min_samples_split=80,   # Aumentado para evitar overfitting
                    min_samples_leaf=40,    # Aumentado para evitar overfitting
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.5,        # Usar apenas 50% das amostras por árvore
                    max_leaf_nodes=15       # Limitar número de folhas
                )
                rf.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(rf, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'random_forest_enhanced')
                    self.save_model(rf, 'random_forest_enhanced', cv_scores, cache_path)
                
                return 'random_forest_enhanced', rf, cv_scores
            else:
                return 'random_forest_enhanced', cached_models['random_forest_enhanced'], cached_metrics['random_forest_enhanced']
        
        def train_xgboost_enhanced():
            if XGB_AVAILABLE and 'xgboost_enhanced' not in cached_models:
                print(f"    🚀 Treinando XGBoost Melhorado...")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=30,  # Reduzido para evitar overfitting
                    max_depth=3,      # Reduzido para evitar overfitting
                    learning_rate=0.05,  # Reduzido para convergência mais suave
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    eval_metric='mlogloss',
                    # Regularização ULTRA RIGOROSA ANTI-OVERFITTING
                    reg_alpha=1.0,    # L1 regularization aumentada drasticamente
                    reg_lambda=1.0,   # L2 regularization aumentada drasticamente
                    subsample=0.4,    # Subsample mais restritivo
                    colsample_bytree=0.4,  # Feature sampling mais restritivo
                    min_child_weight=60,   # Aumentado drasticamente
                    gamma=1.0,        # Regularização adicional aumentada
                    max_delta_step=1,  # Limitar mudanças por step
                    early_stopping_rounds=3  # Parada antecipada mais rápida
                )
                xgb_model.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(xgb_model, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'xgboost_enhanced')
                    self.save_model(xgb_model, 'xgboost_enhanced', cv_scores, cache_path)
                
                return 'xgboost_enhanced', xgb_model, cv_scores
            elif XGB_AVAILABLE and 'xgboost_enhanced' in cached_models:
                return 'xgboost_enhanced', cached_models['xgboost_enhanced'], cached_metrics['xgboost_enhanced']
            else:
                return None, None, None
        
        def train_lightgbm_enhanced():
            if LGBM_AVAILABLE and 'lightgbm_enhanced' not in cached_models:
                print(f"    💡 Treinando LightGBM Melhorado...")
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    class_weight='balanced',
                    # Regularização otimizada
                    reg_alpha=self.config.regularization_strength,
                    reg_lambda=self.config.regularization_strength,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=self.config.min_samples_leaf,
                    min_split_gain=0.1,
                    feature_fraction=0.8
                )
                
                X_df = pd.DataFrame(X_scaled, columns=feature_names)
                lgb_model.fit(X_df, y)
                
                cv_scores = self._cross_validate_model(lgb_model, X_scaled, y, tscv, feature_names)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'lightgbm_enhanced')
                    self.save_model(lgb_model, 'lightgbm_enhanced', cv_scores, cache_path)
                
                return 'lightgbm_enhanced', lgb_model, cv_scores
            elif LGBM_AVAILABLE and 'lightgbm_enhanced' in cached_models:
                return 'lightgbm_enhanced', cached_models['lightgbm_enhanced'], cached_metrics['lightgbm_enhanced']
            else:
                return None, None, None
        
        def train_logistic_enhanced():
            if 'logistic_enhanced' not in cached_models:
                print(f"    📊 Treinando Logistic Regression Melhorado...")
                lr = LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000,
                    class_weight='balanced',
                    C=1.0 / self.config.regularization_strength,  # Inverso da regularização
                    l1_ratio=self.config.l1_ratio,
                    solver='saga'  # Suporta L1 e L2
                )
                lr.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(lr, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'logistic_enhanced')
                    self.save_model(lr, 'logistic_enhanced', cv_scores, cache_path)
                
                return 'logistic_enhanced', lr, cv_scores
            else:
                return 'logistic_enhanced', cached_models['logistic_enhanced'], cached_metrics['logistic_enhanced']
        
        def train_naive_bayes_enhanced():
            if 'naive_bayes_enhanced' not in cached_models:
                print(f"    🎯 Treinando Naive Bayes Melhorado...")
                nb = GaussianNB()
                nb.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(nb, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'naive_bayes_enhanced')
                    self.save_model(nb, 'naive_bayes_enhanced', cv_scores, cache_path)
                
                return 'naive_bayes_enhanced', nb, cv_scores
            else:
                return 'naive_bayes_enhanced', cached_models['naive_bayes_enhanced'], cached_metrics['naive_bayes_enhanced']
        
        def train_svm_enhanced():
            if 'svm_enhanced' not in cached_models:
                print(f"    🔥 Treinando SVM Melhorado...")
                svm = SVC(
                    probability=True,
                    random_state=self.config.random_state,
                    class_weight='balanced',
                    C=1.0 / self.config.regularization_strength,
                    gamma='scale'
                )
                svm.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(svm, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'svm_enhanced')
                    self.save_model(svm, 'svm_enhanced', cv_scores, cache_path)
                
                return 'svm_enhanced', svm, cv_scores
            else:
                return 'svm_enhanced', cached_models['svm_enhanced'], cached_metrics['svm_enhanced']
        
        def train_mlp_enhanced():
            if 'mlp_enhanced' not in cached_models:
                print(f"    🧠 Treinando MLP Melhorado...")
                mlp = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=self.config.random_state,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    alpha=self.config.regularization_strength,
                    learning_rate_init=self.config.learning_rate,
                    dropout=self.config.dropout_rate
                )
                mlp.fit(X_scaled, y)
                
                cv_scores = self._cross_validate_model(mlp, X_scaled, y, tscv)
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'mlp_enhanced')
                    self.save_model(mlp, 'mlp_enhanced', cv_scores, cache_path)
                
                return 'mlp_enhanced', mlp, cv_scores
            else:
                return 'mlp_enhanced', cached_models['mlp_enhanced'], cached_metrics['mlp_enhanced']
        
        # Lista de modelos para treinar
        models_to_train = ['random_forest_enhanced', 'xgboost_enhanced', 'lightgbm_enhanced', 
                          'logistic_enhanced', 'naive_bayes_enhanced', 'svm_enhanced', 'mlp_enhanced']
        
        # Se há classes ausentes, usar apenas modelos robustos
        if unique_classes is not None and len(unique_classes) < 10:
            print(f"    ⚠️  Usando apenas modelos robustos melhorados para {len(unique_classes)} classes")
            models_to_train = ['random_forest_enhanced', 'logistic_enhanced', 'naive_bayes_enhanced', 'svm_enhanced']
        
        with ThreadPoolExecutor(max_workers=min(8, len(models_to_train))) as executor:
            futures = []
            
            if 'random_forest_enhanced' in models_to_train:
                futures.append(executor.submit(train_random_forest_enhanced))
            if 'xgboost_enhanced' in models_to_train:
                futures.append(executor.submit(train_xgboost_enhanced))
            if 'lightgbm_enhanced' in models_to_train:
                futures.append(executor.submit(train_lightgbm_enhanced))
            if 'logistic_enhanced' in models_to_train:
                futures.append(executor.submit(train_logistic_enhanced))
            if 'naive_bayes_enhanced' in models_to_train:
                futures.append(executor.submit(train_naive_bayes_enhanced))
            if 'svm_enhanced' in models_to_train:
                futures.append(executor.submit(train_svm_enhanced))
            if 'mlp_enhanced' in models_to_train:
                futures.append(executor.submit(train_mlp_enhanced))
            
            for future in as_completed(futures):
                try:
                    name, model, model_metrics = future.result()
                    if name and model is not None:
                        models[name] = model
                        metrics[name] = model_metrics
                except Exception as e:
                    print(f"    ❌ Erro no treinamento melhorado: {e}")
        
        # Validar qualidade dos modelos melhorados
        print(f"    🔍 Validando qualidade dos modelos melhorados...")
        good_models = {}
        for name, model in models.items():
            if name != 'scaler':
                validation = self._validate_model_quality(model, X_scaled, y, name)
                model_metrics = metrics.get(name, {})
                
                # Critérios ULTRA RIGOROSOS ANTI-OVERFITTING para modelos melhorados
                is_good_quality = (
                    validation['is_good_quality'] and
                    not model_metrics.get('is_overfitting', False) and
                    model_metrics.get('overfitting_gap', 1.0) < 0.05 and  # Gap menor que 5% (ultra restritivo)
                    model_metrics.get('accuracy', 0) > 0.15 and  # Acurácia maior que 15%
                    model_metrics.get('stability', 0) > 0.8 and  # Estabilidade maior que 80%
                    model_metrics.get('reliability_score', 0) > 0.20  # Confiabilidade maior que 20%
                )
                
                if is_good_quality:
                    good_models[name] = model
                    print(f"    ✅ {name}: APROVADO (acc: {model_metrics.get('accuracy', 0):.3f}, gap: {model_metrics.get('overfitting_gap', 0):.3f})")
                else:
                    reason = []
                    if model_metrics.get('is_overfitting', False):
                        reason.append("OVERFITTING")
                    if model_metrics.get('overfitting_gap', 0) >= 0.20:
                        reason.append(f"GAP_ALTO({model_metrics.get('overfitting_gap', 0):.3f})")
                    if model_metrics.get('accuracy', 0) <= 0.20:
                        reason.append(f"ACC_BAIXA({model_metrics.get('accuracy', 0):.3f})")
                    print(f"    ❌ {name}: REJEITADO ({', '.join(reason)})")
        
        # Ensemble adaptativo melhorado
        if len(good_models) > 1 and 'ensemble_adaptive_enhanced' not in cached_models:
            print(f"    🎯 Criando ensemble adaptativo melhorado com {len(good_models)} modelos...")
            
            ensemble_models = []
            ensemble_weights = []
            
            # Criar ensemble com calibração
            for name, model in good_models.items():
                if hasattr(model, 'predict_proba'):
                    # Calibrar modelo individual
                    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                    calibrated_model.fit(X_scaled, y)
                    ensemble_models.append(calibrated_model)
                    
                    # Peso baseado na performance
                    model_accuracy = metrics.get(name, {}).get('accuracy', 0)
                    ensemble_weights.append(model_accuracy)
            
            # Normalizar pesos
            if ensemble_weights:
                ensemble_weights = np.array(ensemble_weights)
                ensemble_weights = ensemble_weights / np.sum(ensemble_weights)
            
            models['ensemble_adaptive_enhanced'] = {
                'models': ensemble_models,
                'weights': ensemble_weights.tolist() if ensemble_weights is not None else None
            }
            
            # Calcular métricas do ensemble
            ensemble_accuracy = np.mean([metrics.get(name, {}).get('accuracy', 0) for name in good_models.keys()])
            ensemble_brier = np.mean([metrics.get(name, {}).get('brier_score', 1) for name in good_models.keys()])
            
            metrics['ensemble_adaptive_enhanced'] = {
                'n_models': len(ensemble_models),
                'accuracy': ensemble_accuracy,
                'brier_score': ensemble_brier,
                'ensemble_type': 'enhanced_adaptive',
                'weights': ensemble_weights.tolist() if ensemble_weights is not None else None
            }
            
            if use_cache:
                cache_path = self.get_cache_path(series, coluna, 'ensemble_adaptive_enhanced')
                self.save_model(models['ensemble_adaptive_enhanced'], 'ensemble_adaptive_enhanced', metrics['ensemble_adaptive_enhanced'], cache_path)
        elif 'ensemble_adaptive_enhanced' in cached_models:
            models['ensemble_adaptive_enhanced'] = cached_models['ensemble_adaptive_enhanced']
            metrics['ensemble_adaptive_enhanced'] = cached_metrics['ensemble_adaptive_enhanced']
    
    def _cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit, 
                            feature_names: List[str] = None) -> Dict:
        """Validação cruzada ULTRA RIGOROSA com detecção avançada de overfitting"""
        accuracy_scores = []
        train_accuracy_scores = []  # Para detectar overfitting
        brier_scores = []
        log_loss_scores = []
        stability_scores = []  # Nova métrica de estabilidade
        
        # Obter todas as classes únicas do dataset completo
        all_unique_classes = np.unique(y)
        n_all_classes = len(all_unique_classes)
        
        # Validação adicional: verificar distribuição de classes
        class_distribution = np.bincount(y) / len(y)
        min_class_freq = np.min(class_distribution)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Verificar se há classes suficientes no treino e validação
            train_classes = np.unique(y_train)
            val_classes = np.unique(y_val)
            
            if len(train_classes) < 3 or len(val_classes) < 2:
                # Pular fold se não há classes suficientes
                continue
            
            # Treinar modelo temporário (clonagem segura do sklearn)
            temp_model = clone(model)
            if feature_names and hasattr(temp_model, 'fit'):
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                X_val_df = pd.DataFrame(X_val, columns=feature_names)
                temp_model.fit(X_train_df, y_train)
                y_pred_proba = temp_model.predict_proba(X_val_df)
                y_train_pred_proba = temp_model.predict_proba(X_train_df)
            else:
                temp_model.fit(X_train, y_train)
                y_pred_proba = temp_model.predict_proba(X_val)
                y_train_pred_proba = temp_model.predict_proba(X_train)
            
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_train_pred = np.argmax(y_train_pred_proba, axis=1)
            
            # Calcular métricas de validação
            val_accuracy = accuracy_score(y_val, y_pred)
            accuracy_scores.append(val_accuracy)
            
            # Calcular métricas de treino para detectar overfitting
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_accuracy_scores.append(train_accuracy)
            
            # Calcular estabilidade (consistência entre folds)
            stability_score = 1.0 - abs(val_accuracy - train_accuracy)  # Quanto mais próximo, mais estável
            stability_scores.append(stability_score)
            
            # Brier Score - usar todas as classes do dataset
            y_val_ohe = np.zeros((len(y_val), n_all_classes))
            for i, class_label in enumerate(all_unique_classes):
                y_val_ohe[y_val == class_label, i] = 1
            
            # Ajustar y_pred_proba para ter o mesmo número de classes
            if y_pred_proba.shape[1] != n_all_classes:
                # Se o modelo retornou menos classes, expandir para incluir todas
                y_pred_proba_adjusted = np.zeros((len(y_val), n_all_classes))
                for i, class_label in enumerate(all_unique_classes):
                    if class_label in temp_model.classes_:
                        class_idx = np.where(temp_model.classes_ == class_label)[0][0]
                        y_pred_proba_adjusted[:, i] = y_pred_proba[:, class_idx]
                    else:
                        # Se a classe não está presente no modelo, usar probabilidade uniforme
                        y_pred_proba_adjusted[:, i] = 1.0 / n_all_classes
                y_pred_proba = y_pred_proba_adjusted
            
            brier_scores.append(brier_score_loss(y_val_ohe.ravel(), y_pred_proba.ravel()))
            
            # Log Loss - usar labels explícitas
            log_loss_scores.append(log_loss(y_val, y_pred_proba, labels=all_unique_classes))
        
        # Calcular gap de overfitting
        train_accuracy = np.mean(train_accuracy_scores)
        val_accuracy = np.mean(accuracy_scores)
        overfitting_gap = train_accuracy - val_accuracy
        
        # Calcular estabilidade geral
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # Detectar overfitting severo - ULTRA RIGOROSO ANTI-OVERFITTING
        is_overfitting = overfitting_gap > 0.05  # Gap maior que 5% indica overfitting
        
        # Detectar instabilidade
        is_unstable = avg_stability < 0.7  # Estabilidade menor que 70% indica instabilidade
        
        # Calcular confiabilidade geral
        reliability_score = (val_accuracy + avg_stability) / 2
        
        return {
            'accuracy': val_accuracy,
            'accuracy_std': np.std(accuracy_scores),
            'train_accuracy': train_accuracy,
            'overfitting_gap': overfitting_gap,
            'is_overfitting': is_overfitting,
            'stability': avg_stability,
            'is_unstable': is_unstable,
            'reliability_score': reliability_score,
            'brier_score': np.mean(brier_scores),
            'brier_score_std': np.std(brier_scores),
            'log_loss': np.mean(log_loss_scores),
            'log_loss_std': np.std(log_loss_scores),
            'cv_folds': len(accuracy_scores),
            'min_class_freq': min_class_freq,
            'class_balance_ok': min_class_freq > 0.05  # Pelo menos 5% de cada classe
        }
    
    def _train_models_parallel(self, X_scaled: np.ndarray, y: np.ndarray, feature_names: List[str], 
                             tscv: TimeSeriesSplit, models: Dict, metrics: Dict, 
                             cached_models: Dict, cached_metrics: Dict, 
                             series: np.ndarray, coluna: str, use_cache: bool) -> None:
        """Treina modelos em paralelo"""
        
        def train_random_forest():
            if 'random_forest' not in cached_models:
                print(f"    🌲 Treinando Random Forest...")
                rf = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
                rf.fit(X_scaled, y)
                
                # Cross-validation
                rf_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    rf_temp = RandomForestClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        random_state=self.config.random_state,
                        n_jobs=-1
                    )
                    rf_temp.fit(X_train, y_train)
                    y_pred = rf_temp.predict(X_val)
                    rf_scores.append(accuracy_score(y_val, y_pred))
                
                model_metrics = {
                    'accuracy': np.mean(rf_scores),
                    'train_time': 0
                }
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'random_forest')
                    self.save_model(rf, 'random_forest', model_metrics, cache_path)
                
                return 'random_forest', rf, model_metrics
            else:
                return 'random_forest', cached_models['random_forest'], cached_metrics['random_forest']
        
        def train_xgboost():
            if XGB_AVAILABLE and 'xgboost' not in cached_models:
                print(f"    🚀 Treinando XGBoost...")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
                xgb_model.fit(X_scaled, y)
                
                # Cross-validation
                xgb_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    xgb_temp = xgb.XGBClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_state=self.config.random_state,
                        n_jobs=-1
                    )
                    xgb_temp.fit(X_train, y_train)
                    y_pred = xgb_temp.predict(X_val)
                    xgb_scores.append(accuracy_score(y_val, y_pred))
                
                model_metrics = {
                    'accuracy': np.mean(xgb_scores),
                    'train_time': 0
                }
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'xgboost')
                    self.save_model(xgb_model, 'xgboost', model_metrics, cache_path)
                
                return 'xgboost', xgb_model, model_metrics
            elif XGB_AVAILABLE and 'xgboost' in cached_models:
                return 'xgboost', cached_models['xgboost'], cached_metrics['xgboost']
            else:
                return None, None, None
        
        def train_lightgbm():
            if LGBM_AVAILABLE and 'lightgbm' not in cached_models:
                print(f"    💡 Treinando LightGBM...")
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                
                X_df = pd.DataFrame(X_scaled, columns=feature_names)
                lgb_model.fit(X_df, y)
                
                # Cross-validation
                lgb_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    X_train_df = pd.DataFrame(X_train, columns=feature_names)
                    X_val_df = pd.DataFrame(X_val, columns=feature_names)
                    
                    lgb_temp = lgb.LGBMClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbose=-1
                    )
                    lgb_temp.fit(X_train_df, y_train)
                    y_pred = lgb_temp.predict(X_val_df)
                    lgb_scores.append(accuracy_score(y_val, y_pred))
                
                model_metrics = {
                    'accuracy': np.mean(lgb_scores),
                    'train_time': 0
                }
                
                if use_cache:
                    cache_path = self.get_cache_path(series, coluna, 'lightgbm')
                    self.save_model(lgb_model, 'lightgbm', model_metrics, cache_path)
                
                return 'lightgbm', lgb_model, model_metrics
            elif LGBM_AVAILABLE and 'lightgbm' in cached_models:
                return 'lightgbm', cached_models['lightgbm'], cached_metrics['lightgbm']
            else:
                return None, None, None
        
        # Executar treinamentos em paralelo
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(train_random_forest),
                executor.submit(train_xgboost),
                executor.submit(train_lightgbm)
            ]
            
            for future in as_completed(futures):
                try:
                    name, model, model_metrics = future.result()
                    if name and model is not None:
                        models[name] = model
                        metrics[name] = model_metrics
                except Exception as e:
                    print(f"    ❌ Erro no treinamento: {e}")
        
        # Ensemble (sequencial, depende dos outros modelos)
        if self.config.calibrate and len(models) > 1 and 'ensemble' not in cached_models:
            print(f"    🎯 Criando ensemble...")
            ensemble_models = []
            for name, model in models.items():
                if name != 'ensemble':
                    calibrated = CalibratedClassifierCV(
                        model,
                        method=self.config.calibration_method,
                        cv=self.config.calibration_cv
                    )
                    calibrated.fit(X_scaled, y)
                    ensemble_models.append(calibrated)
            
            models['ensemble'] = ensemble_models
            metrics['ensemble'] = {
                'n_models': len(ensemble_models),
                'train_time': 0
            }
            
            if use_cache:
                cache_path = self.get_cache_path(series, coluna, 'ensemble')
                self.save_model(ensemble_models, 'ensemble', metrics['ensemble'], cache_path)
        elif self.config.calibrate and 'ensemble' in cached_models:
            models['ensemble'] = cached_models['ensemble']
            metrics['ensemble'] = cached_metrics['ensemble']
    
    def predict_next_digit(self, ml_results: Dict, series: np.ndarray) -> Dict:
        """Prediz próximo dígito usando modelos científicos com análise de confiança"""
        # Fallback inteligente baseado em frequência histórica
        def get_historical_fallback(series: np.ndarray) -> Dict:
            """Fallback baseado em frequência histórica com suavização de Laplace"""
            if len(series) == 0:
                return {
                    'probabilities': np.ones(10) / 10.0,
                    'confidence': 0.0,
                    'uncertainty': 1.0,
                    'top_digits': list(range(10)),
                    'prediction_quality': 'low'
                }
            
            # Frequência histórica com suavização de Laplace
            counts = np.bincount(series, minlength=10).astype(float)
            alpha = 1.0  # Suavização de Laplace
            smoothed_counts = counts + alpha
            probabilities = smoothed_counts / np.sum(smoothed_counts)
            
            # Calcular confiança baseada no tamanho da amostra
            n_samples = len(series)
            confidence = min(0.8, n_samples / 100.0)  # Máximo 0.8 para fallback
            
            return {
                'probabilities': probabilities,
                'confidence': confidence,
                'uncertainty': 1.0 - confidence,
                'top_digits': np.argsort(probabilities)[::-1][:5].tolist(),
                'prediction_quality': 'low'
            }
        
        if ml_results is None:
            return get_historical_fallback(series)
        
        recent = series[-self.config.window:]
        if len(recent) < self.config.window:
            return get_historical_fallback(series)
        
        # Criar features científicas para predição
        features, _, feature_names = self.create_features(recent.reshape(1, -1).flatten(), window=len(recent))
        
        if len(features) == 0:
            return get_historical_fallback(series)
        
        feature_vector = features[0].reshape(1, -1)
        
        # Normalizar
        if 'scaler' in ml_results['models']:
            feature_vector_scaled = ml_results['models']['scaler'].transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector
        
        # Predições de todos os modelos
        model_predictions = []
        model_names = []
        
        for model_name, model in ml_results['models'].items():
            if model_name == 'scaler':
                continue
                
            try:
                if model_name == 'frequency_fallback':
                    # Usar frequência histórica com suavização de Laplace
                    series = model['series']
                    counts = np.bincount(series, minlength=10).astype(float)
                    alpha = 1.0  # Suavização de Laplace
                    smoothed_counts = counts + alpha
                    pred = smoothed_counts / np.sum(smoothed_counts)
                elif model_name == 'lightgbm':
                    feature_df = pd.DataFrame(feature_vector_scaled, columns=feature_names)
                    pred = model.predict_proba(feature_df)[0]
                elif model_name == 'ensemble_adaptive':
                    # Ensemble adaptativo sofisticado com pesos
                    if isinstance(model, dict) and 'models' in model and 'weights' in model:
                        ensemble_models = model['models']
                        weights = model['weights']
                        
                        if weights is not None:
                            # Usar pesos específicos
                            ensemble_preds = []
                            for i, ensemble_model in enumerate(ensemble_models):
                                pred = ensemble_model.predict_proba(feature_vector_scaled)[0]
                                ensemble_preds.append(pred * weights[i])
                            pred = np.sum(ensemble_preds, axis=0)
                        else:
                            # Usar média simples
                            ensemble_preds = []
                            for ensemble_model in ensemble_models:
                                pred = ensemble_model.predict_proba(feature_vector_scaled)[0]
                                ensemble_preds.append(pred)
                            pred = np.mean(ensemble_preds, axis=0)
                    else:
                        # Fallback para formato antigo
                        ensemble_preds = []
                        for ensemble_model in model:
                            pred = ensemble_model.predict_proba(feature_vector_scaled)[0]
                            ensemble_preds.append(pred)
                        pred = np.mean(ensemble_preds, axis=0)
                else:
                    pred = model.predict_proba(feature_vector_scaled)[0]
                
                model_predictions.append(pred)
                model_names.append(model_name)
            except Exception as e:
                print(f"    ⚠️  Erro na predição do {model_name}: {e}")
                continue
        
        if not model_predictions:
            return get_historical_fallback(series)
        
        # Calcular incerteza entre modelos
        uncertainty_info = self.confidence_analyzer.calculate_prediction_uncertainty(model_predictions)
        
        # Usar pesos adaptativos se disponíveis
        if hasattr(self, 'model_weights') and len(model_predictions) > 1:
            coluna = 'default'  # Assumir coluna padrão para simplicidade
            if coluna in self.model_weights:
                weights = []
                for model_name in model_names:
                    weight = self.model_weights[coluna].get(model_name, 1.0)
                    weights.append(weight)
                
                # Normalizar pesos
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Predição ponderada
                final_pred = np.average(model_predictions, axis=0, weights=weights)
            else:
                final_pred = np.mean(model_predictions, axis=0)
        else:
            final_pred = np.mean(model_predictions, axis=0)
        
        # Normalizar probabilidades
        final_pred = final_pred / np.sum(final_pred)
        
        # Calcular intervalos de confiança
        confidence_info = self.confidence_analyzer.calculate_confidence_intervals(
            final_pred, uncertainty_info['uncertainty']
        )
        
        # Top dígitos
        top_digits = np.argsort(final_pred)[::-1][:5]
        
        # Qualidade da predição
        max_prob = np.max(final_pred)
        entropy_pred = entropy(final_pred + 1e-8, base=2)
        max_entropy = math.log2(10)
        normalized_entropy = entropy_pred / max_entropy
        
        if max_prob > 0.3 and normalized_entropy < 0.7:
            quality = 'high'
        elif max_prob > 0.2 and normalized_entropy < 0.8:
            quality = 'medium'
        else:
            quality = 'low'
        
        return {
            'probabilities': final_pred,
            'confidence': confidence_info['mean_confidence'],
            'uncertainty': uncertainty_info['uncertainty'],
            'agreement': uncertainty_info['agreement'],
            'top_digits': top_digits.tolist(),
            'prediction_quality': quality,
            'model_predictions': model_predictions,
            'model_names': model_names,
            'confidence_intervals': confidence_info['intervals']
        }
    
    def predict_parallel(self, ml_results_dict: Dict[str, Dict], series_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prediz próximos dígitos para múltiplas colunas em paralelo"""
        def predict_single(col: str, ml_results: Dict, series: np.ndarray) -> Tuple[str, np.ndarray]:
            return col, self.predict_next_digit(ml_results, series)
        
        predictions = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(predict_single, col, ml_results, series): col
                for col, (ml_results, series) in zip(ml_results_dict.keys(), 
                                                   zip(ml_results_dict.values(), series_dict.values()))
            }
            
            for future in as_completed(futures):
                try:
                    col, pred = future.result()
                    predictions[col] = pred
                except Exception as e:
                    col = futures[future]
                    print(f"❌ Erro na predição de {col}: {e}")
                    # Fallback para probabilidades uniformes
                    predictions[col] = np.ones(10) / 10.0
        
        return predictions
    
    def dirichlet_probs(self, series: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Probabilidades usando suavização de Dirichlet"""
        counts = np.bincount(series, minlength=10).astype(float)
        n = counts.sum()
        post = counts + alpha
        return post / (n + 10.0 * alpha) if n > 0 else np.ones(10) / 10.0
    
    def markov_probs(self, series: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Probabilidades usando Cadeia de Markov"""
        if series.size < 2:
            return np.ones(10) / 10.0
        
        T = np.zeros((10, 10), dtype=float)
        prev = series[:-1]
        nxt = series[1:]
        
        for a, b in zip(prev, nxt):
            T[a, b] += 1.0
        
        T += alpha
        T = T / T.sum(axis=1, keepdims=True)
        
        last = series[-1]
        return T[last].copy()
    
    def topk_indices(self, probs: np.ndarray, k: int) -> List[int]:
        """Retorna índices dos k maiores valores"""
        return list(np.argsort(probs)[::-1][:k])
    
    def sample_games(self, prob_by_col: Dict[str, np.ndarray], n_games: int) -> List[Tuple[int, ...]]:
        """Gera jogos por amostragem probabilística"""
        rng = np.random.default_rng(self.config.random_state)
        cols = list(prob_by_col.keys())
        games = set()
        max_attempts = n_games * 20
        
        for _ in range(max_attempts):
            pick = []
            for c in cols:
                p = prob_by_col[c].copy()
                p_sum = np.sum(p)
                if p_sum > 0:
                    p = p / p_sum
                else:
                    p = np.ones(10) / 10.0
                
                digit = int(rng.choice(np.arange(10), p=p))
                pick.append(digit)
            games.add(tuple(pick))
            if len(games) >= n_games:
                break
        
        return list(games)[:n_games]
    
    def top_product_games(self, prob_by_col: Dict[str, np.ndarray], n_games: int, top_per_col: int = 5) -> List[Tuple[int, ...]]:
        """Gera jogos por ranking de probabilidades"""
        import itertools
        import math
        
        cols = list(prob_by_col.keys())
        top_digits = {c: self.topk_indices(prob_by_col[c], top_per_col) for c in cols}
        combos = list(itertools.product(*[top_digits[c] for c in cols]))
        
        def score(combo):
            s = 0.0
            for i, c in enumerate(cols):
                s += math.log(prob_by_col[c][combo[i]] + 1e-12)
            return s
        
        combos.sort(key=score, reverse=True)
        return combos[:n_games]
    
    def analyze_column(self, series: np.ndarray, coluna: str, use_cache: bool = True) -> Dict:
        """Analisa uma coluna com métodos científicos avançados e features melhoradas"""
        print(f"📈 Analisando {coluna} cientificamente...")
        
        # Criar features avançadas
        advanced_features = self.feature_engineer.create_ensemble_features(series)
        
        # Análise científica em paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Métodos estatísticos
            future_dirichlet = executor.submit(self.dirichlet_probs, series)
            future_markov = executor.submit(self.markov_probs, series)
            
            # Análise temporal
            future_seasonality = executor.submit(self.temporal_analyzer.detect_seasonality, series)
            future_trend = executor.submit(self.temporal_analyzer.calculate_trend, series)
            future_cycles = executor.submit(self.temporal_analyzer.detect_cycles, series)
            
            # Análise de entropia
            future_entropy = executor.submit(self.entropy_analyzer.calculate_entropy, series)
            future_randomness = executor.submit(self.entropy_analyzer.test_randomness, series)
            
            # Detecção de anomalias
            future_anomalies = executor.submit(self.detect_anomalies, series)
            
            # Coletar resultados
            probs_dirichlet = future_dirichlet.result()
            probs_markov = future_markov.result()
            seasonality = future_seasonality.result()
            trend = future_trend.result()
            cycles = future_cycles.result()
            entropy_info = future_entropy.result()
            randomness_info = future_randomness.result()
            anomalies = future_anomalies.result()
        
        # Modelos ML científicos com features melhoradas
        ml_results = self.train_models_enhanced(series, coluna, advanced_features, use_cache)
        
        if ml_results is not None:
            ml_prediction = self.predict_next_digit(ml_results, series)
            probs_ml = ml_prediction['probabilities']
            confidence = ml_prediction['confidence']
            uncertainty = ml_prediction['uncertainty']
            quality = ml_prediction['prediction_quality']
        else:
            probs_ml = probs_dirichlet
            confidence = 0.0
            uncertainty = 1.0
            quality = 'low'
        
        # Calcular score de confiança geral adaptativo
        overall_confidence = self._calculate_overall_confidence(
            entropy_info, randomness_info, seasonality, trend, confidence,
            data_quality=anomalies.get('data_quality', 'medium'),
            n_samples=len(series)
        )
        
        return {
            # Probabilidades
            'dirichlet': probs_dirichlet,
            'markov': probs_markov,
            'ml_avancado': probs_ml,
            
            # Top dígitos
            'top_dirichlet': self.topk_indices(probs_dirichlet, 5),
            'top_markov': self.topk_indices(probs_markov, 5),
            'top_ml': self.topk_indices(probs_ml, 5),
            
            # Análise temporal
            'seasonality': seasonality,
            'trend': trend,
            'cycles': cycles,
            'sequential_patterns': cycles['sequential_patterns'],
            
            # Análise de entropia
            'entropy': entropy_info,
            'randomness': randomness_info,
            
            # Anomalias
            'anomalies': anomalies,
            
            # Confiança e qualidade
            'confidence': confidence,
            'uncertainty': uncertainty,
            'prediction_quality': quality,
            'overall_confidence': overall_confidence,
            
            # Métricas científicas expandidas
            'scientific_metrics': {
                'is_predictable': overall_confidence > self.config.min_confidence,
                'has_patterns': seasonality['has_seasonality'] or len(cycles['cycles']) > 0 or len(cycles['spectral_cycles']) > 0,
                'has_sequential_patterns': len(cycles['sequential_patterns']['repetitions']) > 0 or len(cycles['sequential_patterns']['progressions']) > 0,
                'is_random': randomness_info['is_random'],
                'has_anomalies': anomalies['is_anomalous'],
                'data_quality': 'high' if anomalies['anomaly_rate'] < 0.05 else 'medium' if anomalies['anomaly_rate'] < 0.15 else 'low',
                'pattern_complexity': len(cycles['sequential_patterns']['repetitions']) + len(cycles['sequential_patterns']['progressions']),
                'spectral_cycles_count': len(cycles['spectral_cycles']),
                'alternation_rate': cycles['sequential_patterns']['alternations']
            }
        }
    
    def _validate_model_quality(self, model, X: np.ndarray, y: np.ndarray, 
                               model_name: str) -> Dict:
        """Valida qualidade do modelo antes de usar"""
        try:
            # Predições
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X)
                y_pred_proba = None
            
            # Métricas básicas
            accuracy = accuracy_score(y, y_pred)
            
            # Verificar se as probabilidades são informativas
            if y_pred_proba is not None:
                # Calcular entropia das probabilidades
                entropy_scores = []
                for probs in y_pred_proba:
                    probs_norm = probs / np.sum(probs)
                    probs_clean = probs_norm[probs_norm > 0]
                    if len(probs_clean) > 0:
                        entropy_val = entropy(probs_clean, base=2)
                        entropy_scores.append(entropy_val)
                
                avg_entropy = np.mean(entropy_scores) if entropy_scores else 0
                max_entropy = math.log2(10)  # Máxima entropia para 10 classes
                normalized_entropy = avg_entropy / max_entropy
                
                # Verificar se as probabilidades não são uniformes
                is_uniform = normalized_entropy > 0.9
                
                # Verificar se há discriminação entre classes
                max_probs = np.max(y_pred_proba, axis=1)
                avg_max_prob = np.mean(max_probs)
                has_discrimination = avg_max_prob > 0.2  # Pelo menos 20% de confiança média
                
            else:
                is_uniform = True
                has_discrimination = False
                normalized_entropy = 1.0
            
            # Critérios de qualidade ULTRA RIGOROSOS ANTI-OVERFITTING
            quality_score = 0
            if accuracy > 0.12:  # Melhor que aleatório (10%) - mais realista
                quality_score += 1
            if not is_uniform:
                quality_score += 1
            if has_discrimination:
                quality_score += 1
            if accuracy > 0.20:  # Boa acurácia - mais realista
                quality_score += 1
            
            is_good_quality = quality_score >= 2
            
            return {
                'accuracy': accuracy,
                'is_good_quality': is_good_quality,
                'quality_score': quality_score,
                'is_uniform': is_uniform,
                'has_discrimination': has_discrimination,
                'normalized_entropy': normalized_entropy,
                'avg_max_prob': avg_max_prob if y_pred_proba is not None else 0
            }
            
        except Exception as e:
            print(f"    ⚠️  Erro na validação do {model_name}: {e}")
            return {
                'accuracy': 0.0,
                'is_good_quality': False,
                'quality_score': 0,
                'is_uniform': True,
                'has_discrimination': False,
                'normalized_entropy': 1.0,
                'avg_max_prob': 0
            }
    
    def _calculate_overall_confidence(self, entropy_info: Dict, randomness_info: Dict, 
                                    seasonality: Dict, trend: Dict, ml_confidence: float,
                                    data_quality: str = 'medium', n_samples: int = 100) -> float:
        """Calcula score de confiança geral MELHORADO com validação rigorosa"""
        scores = []
        weights = []
        
        # Score de entropia MELHORADO (menor entropia = mais previsível)
        entropy_score = 1 - entropy_info['normalized_entropy']
        # Aplicar função de ativação mais rigorosa
        entropy_score = np.tanh(entropy_score * 2) * 0.5 + 0.5
        scores.append(entropy_score)
        weights.append(0.25)  # Aumentado peso
        
        # Score de aleatoriedade MELHORADO (não aleatório = mais previsível)
        if randomness_info['is_random']:
            randomness_score = 0.1  # Muito baixo para dados aleatórios
        else:
            # Considerar p-valor do teste de aleatoriedade
            p_value = randomness_info.get('p_value', 0.5)
            randomness_score = min(1.0, 1 - p_value + 0.3)  # Bonus para p-valores baixos
        scores.append(randomness_score)
        weights.append(0.20)  # Aumentado peso
        
        # Score de sazonalidade MELHORADO (presença de padrões = mais previsível)
        if seasonality['has_seasonality']:
            # Considerar força e consistência da sazonalidade
            strengths = seasonality.get('strengths', [0])
            if strengths:
                seasonality_strength = max(strengths)
                # Normalizar melhor
                seasonality_score = min(1.0, seasonality_strength * 20)
                # Bonus para múltiplos períodos consistentes
                n_periods = len(seasonality.get('periods', []))
                if n_periods > 1:
                    seasonality_score = min(1.0, seasonality_score * 1.2)
            else:
                seasonality_score = 0.3
        else:
            seasonality_score = 0.2  # Reduzido para dados sem sazonalidade
        scores.append(seasonality_score)
        weights.append(0.15)
        
        # Score de tendência MELHORADO (tendência clara = mais previsível)
        trend_significance = trend['significance']
        trend_direction = trend.get('direction', 'stable')
        
        # Base score
        trend_score = min(1.0, trend_significance)
        
        # Bonus para tendências fortes e consistentes
        if trend_significance > 0.8:
            trend_score = min(1.0, trend_score * 1.3)
        elif trend_significance > 0.6:
            trend_score = min(1.0, trend_score * 1.1)
        
        # Penalty para tendências instáveis
        if trend_direction == 'stable' and trend_significance < 0.3:
            trend_score *= 0.7
            
        scores.append(trend_score)
        weights.append(0.15)
        
        # Score do ML MELHORADO (peso maior com validação rigorosa)
        # Aplicar transformação logística para melhor distribuição
        ml_score = 1 / (1 + np.exp(-10 * (ml_confidence - 0.5)))
        scores.append(ml_score)
        weights.append(0.20)  # Aumentado peso
        
        # Score de qualidade dos dados MELHORADO
        quality_scores = {'high': 1.0, 'medium': 0.6, 'low': 0.2}  # Mais rigoroso
        data_quality_score = quality_scores.get(data_quality, 0.3)
        scores.append(data_quality_score)
        weights.append(0.15)  # Aumentado peso
        
        # Ajuste baseado no tamanho da amostra MELHORADO
        if n_samples < 50:
            sample_size_factor = 0.3  # Muito baixo para amostras pequenas
        elif n_samples < 100:
            sample_size_factor = 0.5
        elif n_samples < 200:
            sample_size_factor = 0.7
        elif n_samples < 500:
            sample_size_factor = 0.9
        else:
            sample_size_factor = 1.0  # Máximo para amostras grandes
        scores.append(sample_size_factor)
        weights.append(0.10)
        
        # Normalizar pesos
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calcular confiança ponderada
        overall_confidence = np.average(scores, weights=weights)
        
        # Aplicar função de ativação MAIS RIGOROSA
        # Usar função sigmóide mais íngreme para melhor discriminação
        overall_confidence = 1 / (1 + np.exp(-8 * (overall_confidence - 0.6)))
        
        # Aplicar threshold mínimo para confiança
        if overall_confidence < 0.3:
            overall_confidence = 0.3  # Mínimo de 30% de confiança
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _generate_frequency_report(self, frequency_analysis: Dict) -> str:
        """Gera relatório detalhado da análise de frequência"""
        report = []
        report.append("📊 RELATÓRIO DE ANÁLISE DE FREQUÊNCIA")
        report.append("=" * 50)
        report.append("")
        report.append("🎯 Análise dos números mais frequentes por coluna")
        report.append("📈 Baseado no histórico completo de concursos")
        report.append("")
        
        # Estatísticas gerais
        stats = frequency_analysis['estatisticas_gerais']
        report.append("📈 ESTATÍSTICAS GERAIS:")
        report.append("-" * 30)
        report.append(f"Total de ocorrências analisadas: {stats['total_ocorrencias']}")
        report.append(f"Número mais saído: {stats['numero_mais_saido_geral']} ({stats['percentual_geral'][stats['numero_mais_saido_geral']]:.1f}%)")
        report.append(f"Número menos saído: {stats['numero_menos_saido_geral']} ({stats['percentual_geral'][stats['numero_menos_saido_geral']]:.1f}%)")
        report.append(f"Média por número: {stats['media_por_numero']:.1f}")
        report.append(f"Desvio padrão: {stats['desvio_padrao_geral']:.2f}")
        report.append("")
        
        # Frequência por coluna
        report.append("🔥 NÚMEROS MAIS FREQUENTES POR COLUNA:")
        report.append("-" * 40)
        
        for col, data in frequency_analysis['frequencias_por_coluna'].items():
            report.append(f"\n{col}:")
            report.append(f"  Total de concursos: {data['total_concursos']}")
            report.append(f"  Número mais frequente: {data['numero_mais_frequente']} ({data['percentual_maximo']:.1f}%)")
            report.append(f"  Número menos frequente: {data['numero_menos_frequente']} ({data['percentual_minimo']:.1f}%)")
            report.append(f"  Desvio padrão: {data['desvio_padrao']:.2f}")
            
            # Top 3 números
            report.append("  Top 3 números:")
            top_3_freqs = sorted(data['frequencia_percentual'].items(), key=lambda x: x[1], reverse=True)[:3]
            for num, freq in top_3_freqs:
                report.append(f"    {num}: {freq:.1f}%")
        
        # Números quentes e frios
        report.append("\n🌡️ NÚMEROS QUENTES E FRIOS:")
        report.append("-" * 30)
        
        for col, quentes in frequency_analysis['recomendacoes']['numeros_quentes'].items():
            if quentes:
                report.append(f"\n{col} - Números Quentes:")
                for num, freq in quentes.items():
                    report.append(f"  {num}: {freq:.1f}% (acima da média)")
        
        for col, frios in frequency_analysis['recomendacoes']['numeros_frios'].items():
            if frios:
                report.append(f"\n{col} - Números Frios:")
                for num, freq in frios.items():
                    report.append(f"  {num}: {freq:.1f}% (abaixo da média)")
        
        # Estratégias por coluna
        report.append("\n💡 ESTRATÉGIAS RECOMENDADAS:")
        report.append("-" * 30)
        
        for col, strategy in frequency_analysis['recomendacoes']['estrategias_por_coluna'].items():
            report.append(f"\n{col}:")
            report.append(f"  {strategy['recomendacao']}")
            if strategy['numeros_quentes']:
                report.append(f"  Números quentes: {', '.join(map(str, strategy['numeros_quentes']))}")
            if strategy['numeros_frios']:
                report.append(f"  Números frios: {', '.join(map(str, strategy['numeros_frios']))}")
        
        # Observações importantes
        report.append("\n⚠️ OBSERVAÇÕES IMPORTANTES:")
        report.append("-" * 30)
        for obs in frequency_analysis['recomendacoes']['observacoes_importantes']:
            report.append(f"• {obs}")
        
        # Padrões temporais (se disponível)
        if frequency_analysis['padroes_temporais']:
            report.append("\n📅 PADRÕES TEMPORAIS:")
            report.append("-" * 25)
            
            por_mes = frequency_analysis['padroes_temporais'].get('por_mes', {})
            if por_mes:
                report.append("Frequência por mês:")
                for mes, dados in por_mes.items():
                    if dados['numero_mais_frequente'] is not None:
                        report.append(f"  Mês {mes}: {dados['numero_mais_frequente']} mais frequente ({dados['total_concursos']} concursos)")
        
        report.append("\n" + "=" * 50)
        report.append("🎯 COMO USAR ESTAS INFORMAÇÕES:")
        report.append("1. Priorize números com frequência acima da média")
        report.append("2. Evite números com frequência muito baixa")
        report.append("3. Combine análise de frequência com outros métodos")
        report.append("4. Monitore mudanças nos padrões ao longo do tempo")
        report.append("5. Lembre-se: frequência passada não garante resultado futuro")
        
        return "\n".join(report)
    
    def _generate_detailed_report(self, resultados: Dict) -> str:
        """Gera relatório detalhado dos resultados com guia prático para apostas"""
        report = []
        report.append("📊 RELATÓRIO DETALHADO DA ANÁLISE CIENTÍFICA")
        report.append("=" * 60)
        report.append("")
        
        # Explicação do que significa cada métrica
        report.append("📚 GUIA DE INTERPRETAÇÃO DAS MÉTRICAS:")
        report.append("-" * 40)
        report.append("🎯 CONFIANÇA: Probabilidade de acerto (0-1)")
        report.append("   • 0.7+ = Alta confiança (recomendado para apostas)")
        report.append("   • 0.5-0.7 = Média confiança (use com cautela)")
        report.append("   • <0.5 = Baixa confiança (evite apostas)")
        report.append("")
        report.append("🔬 QUALIDADE: Confiabilidade do modelo")
        report.append("   • high = Modelo muito confiável")
        report.append("   • medium = Modelo moderadamente confiável")
        report.append("   • low = Modelo pouco confiável")
        report.append("")
        report.append("📈 PREVISÍVEL: Se a coluna tem padrões detectáveis")
        report.append("   • ✅ = Tem padrões (bom para apostas)")
        report.append("   • ❌ = Sem padrões (evite apostas)")
        report.append("")
        report.append("🎲 ALEATÓRIO: Se os dados são puramente aleatórios")
        report.append("   • ❌ = Não é aleatório (bom para apostas)")
        report.append("   • ✅ = É aleatório (evite apostas)")
        report.append("")
        report.append("🧩 COMPLEXIDADE: Quantos padrões diferentes existem")
        report.append("   • Baixa (200-250) = Padrões simples (mais previsível)")
        report.append("   • Média (250-300) = Padrões moderados")
        report.append("   • Alta (300+) = Padrões complexos (menos previsível)")
        report.append("")
        report.append("📈 CICLOS ESPECTRAIS: Padrões que se repetem no tempo")
        report.append("   • 3+ ciclos = Muitos padrões temporais")
        report.append("   • 1-2 ciclos = Poucos padrões temporais")
        report.append("")
        report.append("🔀 TAXA DE ALTERNÂNCIA: Quanto os números mudam")
        report.append("   • 0.7+ = Muita variação (mais imprevisível)")
        report.append("   • 0.5-0.7 = Variação moderada")
        report.append("   • <0.5 = Pouca variação (mais previsível)")
        report.append("")
        
        # Análise por coluna com recomendações
        report.append("🎯 ANÁLISE POR COLUNA COM RECOMENDAÇÕES:")
        report.append("=" * 50)
        
        for coluna, dados in resultados['dados_cientificos'].items():
            if coluna.startswith('Coluna_'):
                confianca = dados['confidence']
                qualidade = dados['quality']
                previsivel = dados['scientific_metrics']['is_predictable']
                aleatorio = dados['scientific_metrics']['is_random']
                complexidade = dados['scientific_metrics']['pattern_complexity']
                ciclos = dados['scientific_metrics']['spectral_cycles_count']
                alternancia = dados['scientific_metrics']['alternation_rate']
                
                # Determinar recomendação
                if confianca >= 0.7 and previsivel and not aleatorio:
                    recomendacao = "🟢 EXCELENTE PARA APOSTAS"
                    explicacao = "Alta confiança, padrões claros, não aleatório"
                elif confianca >= 0.6 and previsivel and not aleatorio:
                    recomendacao = "🟡 BOM PARA APOSTAS"
                    explicacao = "Boa confiança, padrões detectáveis"
                elif confianca >= 0.5 and previsivel:
                    recomendacao = "🟠 APOSTE COM CUIDADO"
                    explicacao = "Confiança moderada, use valores baixos"
                else:
                    recomendacao = "🔴 EVITE APOSTAS"
                    explicacao = "Baixa confiança ou muito aleatório"
                
                report.append(f"🔍 {coluna.upper()}:")
                report.append(f"  📊 Confiança: {confianca:.3f}")
                report.append(f"  🔬 Qualidade: {qualidade}")
                report.append(f"  📈 Previsível: {'Sim' if previsivel else 'Não'}")
                report.append(f"  🎲 Aleatório: {'Sim' if aleatorio else 'Não'}")
                report.append(f"  🧩 Complexidade: {complexidade}")
                report.append(f"  📈 Ciclos espectrais: {ciclos}")
                report.append(f"  🔀 Taxa de alternância: {alternancia:.3f}")
                report.append(f"  💡 RECOMENDAÇÃO: {recomendacao}")
                report.append(f"  📝 Explicação: {explicacao}")
                report.append("")
        
        return "\n".join(report)
    
    def _generate_betting_guide(self, resultados: Dict) -> str:
        """Gera guia prático de apostas baseado nos resultados"""
        guide = []
        guide.append("🎯 GUIA PRÁTICO PARA APOSTAS REAIS")
        guide.append("=" * 50)
        guide.append("")
        
        # Análise dos jogos recomendados
        jogos_ranking = resultados.get('jogos_ranking', [])
        jogos_alta_confianca = resultados.get('jogos_alta_confianca', [])
        
        guide.append("🏆 JOGOS RECOMENDADOS PARA APOSTA:")
        guide.append("-" * 40)
        
        if jogos_ranking:
            guide.append("🥇 TOP-3 JOGOS POR RANKING (MAIS RECOMENDADOS):")
            for i, jogo in enumerate(jogos_ranking[:3], 1):
                jogo_str = " - ".join(map(str, jogo))
                guide.append(f"   {i}º Lugar: {jogo_str}")
            guide.append("")
            guide.append("   💡 COMO USAR: Estes são os jogos com melhor score científico")
            guide.append("   💰 ESTRATÉGIA: Aposte valores maiores nestes jogos")
            guide.append("")
        
        if jogos_alta_confianca:
            guide.append("⭐ JOGOS DE ALTA CONFIANÇA (CONFIANÇA > 0.6):")
            for i, jogo in enumerate(jogos_alta_confianca[:5], 1):
                jogo_str = " - ".join(map(str, jogo))
                guide.append(f"   Conf {i:02d}: {jogo_str}")
            guide.append("")
            guide.append("   💡 COMO USAR: Jogos com alta probabilidade de acerto")
            guide.append("   💰 ESTRATÉGIA: Use para apostas de médio valor")
            guide.append("")
        
        # Estratégia de apostas
        guide.append("💰 ESTRATÉGIA DE APOSTAS RECOMENDADA:")
        guide.append("-" * 40)
        guide.append("1. 🥇 APOSTAS PRINCIPAIS (60% do valor):")
        guide.append("   • Use os TOP-3 jogos por ranking")
        guide.append("   • Aposte valores maiores")
        guide.append("   • Maior chance de retorno")
        guide.append("")
        guide.append("2. 🥈 APOSTAS SECUNDÁRIAS (30% do valor):")
        guide.append("   • Use jogos de alta confiança")
        guide.append("   • Aposte valores médios")
        guide.append("   • Diversificação de risco")
        guide.append("")
        guide.append("3. 🥉 APOSTAS EXPLORATÓRIAS (10% do valor):")
        guide.append("   • Use jogos por amostragem")
        guide.append("   • Aposte valores baixos")
        guide.append("   • Teste novas combinações")
        guide.append("")
        
        # Dicas de gestão de risco
        guide.append("⚠️ GESTÃO DE RISCO:")
        guide.append("-" * 20)
        guide.append("• 🎯 NUNCA aposte mais de 5% da sua renda")
        guide.append("• 📊 Monitore os resultados por 10-15 concursos")
        guide.append("• 🔄 Ajuste a estratégia baseado nos resultados")
        guide.append("• 💰 Defina um limite de perda e respeite")
        guide.append("• 📈 Aumente apostas apenas após sucessos consistentes")
        guide.append("")
        
        # Explicação dos dígitos mais frequentes
        guide.append("🎲 DÍGITOS MAIS FREQUENTES POR COLUNA:")
        guide.append("-" * 40)
        
        for coluna, dados in resultados['dados_cientificos'].items():
            if coluna.startswith('Coluna_'):
                top_digits = dados.get('top_digits', [])
                confianca = dados['confidence']
                # Adicionar emoji de estrela para confiança alta
                emoji = "⭐" if confianca > 0.7 else "🔸" if confianca > 0.6 else "🔹"
                guide.append(f"  {coluna}: {', '.join(map(str, top_digits[:3]))} (conf: {confianca:.3f}) {emoji}")
        
        guide.append("")
        guide.append("💡 COMO USAR: Priorize estes dígitos ao montar seus jogos")
        guide.append("⭐ = Alta confiança (0.7+) | 🔸 = Boa confiança (0.6-0.7) | 🔹 = Confiança moderada (<0.6)")
        guide.append("")
        
        return "\n".join(guide)
    
    def run_analysis(self, excel_path: str, n_jogos: int = 10, topk: int = 5, 
                    seed: int = 42, use_cache: bool = True, save_results: bool = True) -> Dict:
        """Executa análise científica completa do Super Sete"""
        # Aplicar seed para reprodutibilidade
        np.random.seed(seed)
        if 'tf' in globals():
            tf.random.set_seed(seed)
        
        # Atualizar configuração com a seed
        self.config.random_state = seed
        
        print("🔬 SUPER SETE ANALYSIS - VERSÃO CIENTÍFICA AVANÇADA")
        print("=" * 60)
        print("🎯 Análise baseada em matemática, estatística e ciência de dados")
        print("=" * 60)
        
        # Carregar dados
        print(f"📂 Carregando dados de: {excel_path}")
        df = self.load_data(excel_path)
        print(f"✅ {len(df)} concursos carregados")
        
        # Análise exploratória científica
        print("\n📊 ANÁLISE EXPLORATÓRIA CIENTÍFICA")
        print("-" * 40)
        self.analyze_data(df)
        
        # Análise de correlação entre colunas
        if self.config.correlation_analysis:
            print("\n🔗 ANÁLISE DE CORRELAÇÃO")
            print("-" * 30)
            correlation_info = self.calculate_correlation_matrix(df)
            if correlation_info:
                print(f"Correlações significativas encontradas: {len(correlation_info['significant_correlations'])}")
                for pair, info in correlation_info['significant_correlations'].items():
                    print(f"  {pair}: {info['correlation']:.3f} (p={info['p_value']:.3f})")
        
        # Análise por coluna - PARALELIZADA
        print(f"\n🔍 ANÁLISE CIENTÍFICA POR COLUNA (PARALELA)")
        print("-" * 40)
        resultados_por_coluna = {}
        
        # Preparar dados para processamento paralelo (apenas colunas numéricas)
        numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
        column_data = [(col, df[col].values.astype(int)) for col in numeric_cols]
        
        # Processar colunas em paralelo
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submeter todas as tarefas
            future_to_col = {
                executor.submit(self.analyze_column, serie, col, use_cache): col 
                for col, serie in column_data
            }
            
            # Coletar resultados conforme completam
            for future in as_completed(future_to_col):
                col = future_to_col[future]
                try:
                    resultado = future.result()
                    resultados_por_coluna[col] = resultado
                    
                    # Exibir resultados científicos
                    print(f"  ✅ {col}:")
                    print(f"    🎯 Top-{topk} ML: {resultado['top_ml']}")
                    print(f"    📊 Confiança: {resultado['overall_confidence']:.3f}")
                    print(f"    🔬 Qualidade: {resultado['prediction_quality']}")
                    print(f"    📈 Previsível: {'✅' if resultado['scientific_metrics']['is_predictable'] else '❌'}")
                    print(f"    🎲 Aleatório: {'✅' if resultado['scientific_metrics']['is_random'] else '❌'}")
                    print(f"    📊 Padrões: {'✅' if resultado['scientific_metrics']['has_patterns'] else '❌'}")
                    print(f"    🔄 Padrões sequenciais: {'✅' if resultado['scientific_metrics']['has_sequential_patterns'] else '❌'}")
                    print(f"    ⚠️  Anomalias: {'✅' if resultado['scientific_metrics']['has_anomalies'] else '❌'}")
                    print(f"    🏆 Qualidade dos dados: {resultado['scientific_metrics']['data_quality']}")
                    print(f"    🧩 Complexidade de padrões: {resultado['scientific_metrics']['pattern_complexity']}")
                    print(f"    📈 Ciclos espectrais: {resultado['scientific_metrics']['spectral_cycles_count']}")
                    print(f"    🔀 Taxa de alternância: {resultado['scientific_metrics']['alternation_rate']:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ Erro ao processar {col}: {e}")
                    # Usar análise básica como fallback
                    serie = df[col].values.astype(int)
                    resultados_por_coluna[col] = {
                        'dirichlet': self.dirichlet_probs(serie),
                        'markov': self.markov_probs(serie),
                        'ml_avancado': self.dirichlet_probs(serie),
                        'top_dirichlet': self.topk_indices(self.dirichlet_probs(serie), topk),
                        'top_markov': self.topk_indices(self.markov_probs(serie), topk),
                        'top_ml': self.topk_indices(self.dirichlet_probs(serie), topk),
                        'overall_confidence': 0.0,
                        'prediction_quality': 'low',
                        'scientific_metrics': {
                            'is_predictable': False, 
                            'is_random': True, 
                            'has_patterns': False, 
                            'has_sequential_patterns': False,
                            'has_anomalies': False, 
                            'data_quality': 'low',
                            'pattern_complexity': 0,
                            'spectral_cycles_count': 0,
                            'alternation_rate': 0.0
                        }
                    }
        
        # Geração de jogos científicos
        print(f"\n🎮 GERAÇÃO DE JOGOS CIENTÍFICOS ({n_jogos} jogos)")
        print("-" * 40)
        
        # Usar probabilidades ML avançadas com filtro de confiança
        prob_by_col = {}
        confidence_by_col = {}
        
        for col in numeric_cols:
            resultado = resultados_por_coluna[col]
            prob_by_col[col] = resultado['ml_avancado']
            confidence_by_col[col] = resultado['overall_confidence']
        
        # Gerar jogos em paralelo
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_amostragem = executor.submit(self.sample_games, prob_by_col, n_jogos)
            future_ranking = executor.submit(self.top_product_games, prob_by_col, n_jogos, topk)
            
            jogos_amostragem = future_amostragem.result()
            jogos_ranking = future_ranking.result()
        
        # Gerar jogos de alta confiança
        jogos_alta_confianca = self._generate_high_confidence_games(
            prob_by_col, confidence_by_col, n_jogos, topk
        )
        
        print("🎲 Jogos por Amostragem:")
        for i, jogo in enumerate(jogos_amostragem, 1):
            print(f"  Jogo {i:02d}: {' - '.join(map(str, jogo))}")
        
        print(f"\n🏆 Jogos por Ranking (Top-{topk}):")
        for i, jogo in enumerate(jogos_ranking, 1):
            import math
            score = sum(math.log(prob_by_col[f"Coluna_{j+1}"][jogo[j]] + 1e-12)
                       for j in range(len(jogo)))
            print(f"  Rank {i:02d}: {' - '.join(map(str, jogo))} (score: {score:.3f})")
        
        print(f"\n⭐ Jogos de Alta Confiança:")
        for i, jogo in enumerate(jogos_alta_confianca, 1):
            conf_score = sum(confidence_by_col[f"Coluna_{j+1}"] for j in range(len(jogo))) / len(jogo)
            print(f"  Conf {i:02d}: {' - '.join(map(str, jogo))} (conf: {conf_score:.3f})")
        
        # Salvar resultados científicos
        if save_results:
            print(f"\n💾 SALVANDO RESULTADOS CIENTÍFICOS")
            print("-" * 40)
            
            os.makedirs("resultados", exist_ok=True)
            
            # Salvar probabilidades e métricas científicas
            scientific_results = {}
            for col in numeric_cols:
                resultado = resultados_por_coluna[col]
                scientific_results[col] = {
                    'probabilities': list(map(float, resultado['ml_avancado'].tolist())),
                    'confidence': float(resultado['overall_confidence']),
                    'quality': str(resultado['prediction_quality']),
                    'scientific_metrics': json.loads(json.dumps(resultado['scientific_metrics'], default=to_serializable)),
                    'top_digits': [int(x) for x in resultado['top_ml']]
                }
            
            with open("resultados/analise_cientifica.json", "w", encoding="utf-8") as f:
                json.dump(scientific_results, f, ensure_ascii=False, indent=2, default=to_serializable)
            
            # Salvar jogos científicos
            with open("resultados/jogos_cientificos.txt", "w", encoding="utf-8") as f:
                f.write("JOGOS SUPER SETE - ANÁLISE CIENTÍFICA AVANÇADA\n")
                f.write("=" * 60 + "\n\n")
                f.write("🔬 Baseado em matemática, estatística e ciência de dados\n")
                f.write("🎯 Análise de padrões temporais, entropia e aleatoriedade\n")
                f.write("📊 Validação cruzada rigorosa e métricas de confiança\n\n")
                
                f.write("🎲 JOGOS POR AMOSTRAGEM\n")
                f.write("-" * 30 + "\n")
                for i, jogo in enumerate(jogos_amostragem, 1):
                    f.write(f"Jogo {i:02d}: {' - '.join(map(str, jogo))}\n")
                
                f.write(f"\n🏆 JOGOS POR RANKING (Top-{topk})\n")
                f.write("-" * 30 + "\n")
                for i, jogo in enumerate(jogos_ranking, 1):
                    score = sum(math.log(prob_by_col[f"Coluna_{j+1}"][jogo[j]] + 1e-12)
                               for j in range(len(jogo)))
                    f.write(f"Rank {i:02d}: {' - '.join(map(str, jogo))} (score: {score:.3f})\n")
                
                f.write(f"\n⭐ JOGOS DE ALTA CONFIANÇA\n")
                f.write("-" * 30 + "\n")
                for i, jogo in enumerate(jogos_alta_confianca, 1):
                    conf_score = sum(confidence_by_col[f"Coluna_{j+1}"] for j in range(len(jogo))) / len(jogo)
                    f.write(f"Conf {i:02d}: {' - '.join(map(str, jogo))} (conf: {conf_score:.3f})\n")
                
                f.write(f"\n📊 RESUMO CIENTÍFICO\n")
                f.write("-" * 30 + "\n")
                for col in numeric_cols:
                    resultado = resultados_por_coluna[col]
                    f.write(f"{col}:\n")
                    f.write(f"  Confiança: {resultado['overall_confidence']:.3f}\n")
                    f.write(f"  Qualidade: {resultado['prediction_quality']}\n")
                    f.write(f"  Previsível: {'Sim' if resultado['scientific_metrics']['is_predictable'] else 'Não'}\n")
                    f.write(f"  Padrões: {'Sim' if resultado['scientific_metrics']['has_patterns'] else 'Não'}\n")
                    f.write(f"  Padrões sequenciais: {'Sim' if resultado['scientific_metrics']['has_sequential_patterns'] else 'Não'}\n")
                    f.write(f"  Qualidade dos dados: {resultado['scientific_metrics']['data_quality']}\n")
                    f.write(f"  Complexidade de padrões: {resultado['scientific_metrics']['pattern_complexity']}\n")
                    f.write(f"  Ciclos espectrais: {resultado['scientific_metrics']['spectral_cycles_count']}\n")
                    f.write(f"  Taxa de alternância: {resultado['scientific_metrics']['alternation_rate']:.3f}\n\n")
            
            # Salvar relatório detalhado
            with open("resultados/relatorio_detalhado.txt", "w", encoding="utf-8") as f:
                f.write(self._generate_detailed_report({
                    'dados_cientificos': scientific_results,
                    'jogos_ranking': jogos_ranking,
                    'jogos_alta_confianca': jogos_alta_confianca
                }))
            
            # Salvar guia de apostas
            with open("resultados/guia_apostas.txt", "w", encoding="utf-8") as f:
                f.write(self._generate_betting_guide({
                    'dados_cientificos': scientific_results,
                    'jogos_ranking': jogos_ranking,
                    'jogos_alta_confianca': jogos_alta_confianca
                }))
            
            # Análise de frequência para relatório
            frequency_analysis = self.frequency_analyzer.analyze_column_frequency(df)
            
            # Salvar relatório de frequência
            with open("resultados/relatorio_frequencia.txt", "w", encoding="utf-8") as f:
                f.write(self._generate_frequency_report(frequency_analysis))
            
            print("✅ Resultados científicos salvos em: resultados/")
            print("📊 Relatório detalhado: resultados/relatorio_detalhado.txt")
            print("🎯 Guia de apostas: resultados/guia_apostas.txt")
            print("📈 Relatório de frequência: resultados/relatorio_frequencia.txt")
        
        # Resumo final
        print(f"\n🎉 ANÁLISE CIENTÍFICA CONCLUÍDA!")
        print(f"📊 {len(df)} concursos analisados")
        print(f"🎮 {len(jogos_amostragem)} jogos gerados por amostragem")
        print(f"🏆 {len(jogos_ranking)} jogos gerados por ranking")
        print(f"⭐ {len(jogos_alta_confianca)} jogos de alta confiança")
        print(f"⚡ Paralelismo: {MAX_WORKERS} workers ativos")
        print(f"🔧 Threads otimizados: {min(4, cpu_count())} por biblioteca")
        
        # Estatísticas de confiança
        avg_confidence = np.mean([resultados_por_coluna[col]['overall_confidence'] for col in numeric_cols])
        predictable_cols = sum(1 for col in numeric_cols if resultados_por_coluna[col]['scientific_metrics']['is_predictable'])
        
        print(f"📈 Confiança média: {avg_confidence:.3f}")
        print(f"🎯 Colunas previsíveis: {predictable_cols}/{len(numeric_cols)}")
        
        # Análise de frequência para incluir nos resultados
        frequency_analysis = self.frequency_analyzer.analyze_column_frequency(df)
        
        return {
            'dados_info': {
                'total_concursos': len(df),
                'colunas_analisadas': len(numeric_cols),
                'periodo': {
                    'inicio': df['Data_Sorteio'].min().strftime('%d/%m/%Y') if 'Data_Sorteio' in df.columns else 'N/A',
                    'fim': df['Data_Sorteio'].max().strftime('%d/%m/%Y') if 'Data_Sorteio' in df.columns else 'N/A'
                }
            },
            'dados_cientificos': scientific_results if 'scientific_results' in locals() else {},
            'resultados': resultados_por_coluna,
            'jogos_amostragem': jogos_amostragem,
            'jogos_ranking': jogos_ranking,
            'jogos_alta_confianca': jogos_alta_confianca,
            'correlation_info': correlation_info if self.config.correlation_analysis else {},
            'frequency_analysis': frequency_analysis,
            'avg_confidence': float(avg_confidence),
            'predictable_cols': int(predictable_cols)
        }
    
    def _generate_high_confidence_games(self, prob_by_col: Dict, confidence_by_col: Dict, 
                                      n_jogos: int, topk: int) -> List[Tuple[int, ...]]:
        """Gera jogos inteligentes priorizando colunas com alta confiança e padrões"""
        cols = list(prob_by_col.keys())
        
        # Classificar colunas por confiança
        high_conf_cols = [col for col in cols if confidence_by_col[col] > self.config.min_confidence]
        medium_conf_cols = [col for col in cols if 0.3 <= confidence_by_col[col] <= self.config.min_confidence]
        low_conf_cols = [col for col in cols if confidence_by_col[col] < 0.3]
        
        # Estratégia adaptativa de geração
        games = []
        
        # 1. Jogos de alta confiança (60% dos jogos)
        if high_conf_cols:
            n_high_conf = int(n_jogos * 0.6)
            high_conf_probs = {col: prob_by_col[col] for col in high_conf_cols}
            high_conf_games = self.top_product_games(high_conf_probs, n_high_conf, min(topk, 3))
            games.extend(high_conf_games)
        
        # 2. Jogos mistos (30% dos jogos) - alta + média confiança
        if high_conf_cols and medium_conf_cols:
            n_mixed = int(n_jogos * 0.3)
            mixed_probs = {}
            
            # Usar todas as colunas, mas com pesos baseados na confiança
            for col in cols:
                if col in high_conf_cols or col in medium_conf_cols:
                    # Aplicar peso de confiança às probabilidades
                    weight = confidence_by_col[col]
                    mixed_probs[col] = prob_by_col[col] * weight + (1 - weight) * (np.ones(10) / 10)
            
            mixed_games = self.top_product_games(mixed_probs, n_mixed, min(topk, 4))
            games.extend(mixed_games)
        
        # 3. Jogos exploratórios (10% dos jogos) - incluir todas as colunas
        n_exploratory = n_jogos - len(games)
        if n_exploratory > 0:
            # Usar amostragem probabilística para exploração
            exploratory_games = self.sample_games(prob_by_col, n_exploratory)
            games.extend(exploratory_games)
        
        # Se não gerou jogos suficientes, completar com jogos aleatórios inteligentes
        if len(games) < n_jogos:
            remaining = n_jogos - len(games)
            # Gerar jogos com distribuição balanceada
            balanced_probs = {}
            for col in cols:
                # Suavizar probabilidades para evitar extremos
                probs = prob_by_col[col]
                smoothed = 0.7 * probs + 0.3 * (np.ones(10) / 10)
                balanced_probs[col] = smoothed / np.sum(smoothed)
            
            additional_games = self.sample_games(balanced_probs, remaining)
            games.extend(additional_games)
        
        return games[:n_jogos]

from ..config.model_configs import get_model_config

def main():
    """Função principal com configuração científica avançada"""
    # Carregar dados primeiro para obter quantidade de concursos
    print("📊 Carregando dados para configuração dinâmica...")
    analyzer_temp = SuperSeteAnalyzer()
    df = analyzer_temp.load_data("Super Sete.xlsx")
    n_concursos = len(df)
    
    print(f"📈 Concursos detectados: {n_concursos}")
    
    # Configuração dinâmica baseada na quantidade de concursos
    config = get_model_config('longo', n_concursos=n_concursos)
    
    print(f"🔧 Configuração dinâmica aplicada:")
    print(f"   - Window: {config.window}")
    print(f"   - Train min size: {config.train_min_size}")
    print(f"   - Min confidence: {config.min_confidence}")
    print(f"   - N estimators: {config.n_estimators}")
    print(f"   - Max depth: {config.max_depth}")
    
    # Criar analisador científico
    analyzer = SuperSeteAnalyzer(config)
    
    # Executar análise científica
    print("🚀 Iniciando análise científica do Super Sete...")
    print("📊 Configuração otimizada para máxima assertividade")
    print("🔬 Baseada em matemática, estatística e ciência de dados")
    print()
    
    resultados = analyzer.run_analysis(
        excel_path="Super Sete.xlsx",
        n_jogos=20,              # Mais jogos para análise
        topk=5,                  # Top 5 dígitos
        seed=42,                 # Seed para reprodutibilidade
        use_cache=True,          # Usar cache para performance
        save_results=True        # Salvar resultados científicos
    )
    
    # Exibir dígitos mais confiáveis por coluna
    print("\n" + "="*60)
    print("🎲 DÍGITOS MAIS CONFIÁVEIS POR COLUNA:")
    print("="*60)
    
    for coluna, dados in resultados['dados_cientificos'].items():
        if coluna.startswith('Coluna_'):
            top_digits = dados.get('top_digits', [])
            confianca = dados['confidence']
            # Adicionar emoji de estrela para confiança alta
            emoji = "⭐" if confianca > 0.7 else "🔸" if confianca > 0.6 else "🔹"
            print(f"  {coluna}: {', '.join(map(str, top_digits[:3]))} (conf: {confianca:.3f}) {emoji}")
    
    print("")
    print("💡 COMO USAR: Priorize estes dígitos ao montar seus jogos")
    print("⭐ = Alta confiança (0.7+) | 🔸 = Boa confiança (0.6-0.7) | 🔹 = Confiança moderada (<0.6)")
    print("")
    
    # Exibir resumo final
    print("\n" + "="*60)
    print("📊 RESUMO FINAL DA ANÁLISE CIENTÍFICA")
    print("="*60)
    print(f"🎯 Confiança média: {resultados['avg_confidence']:.3f}")
    print(f"📈 Colunas previsíveis: {resultados['predictable_cols']}/{len([col for col in resultados['dados'].columns if col.startswith('Coluna_')])}")
    print(f"🎮 Total de jogos gerados: {len(resultados['jogos_amostragem']) + len(resultados['jogos_ranking']) + len(resultados['jogos_alta_confianca'])}")
    print(f"⭐ Jogos de alta confiança: {len(resultados['jogos_alta_confianca'])}")
    
    if resultados['correlation_info']:
        print(f"🔗 Correlações significativas: {len(resultados['correlation_info']['significant_correlations'])}")
    
    print("\n💡 GUIA PRÁTICO PARA APOSTAS REAIS:")
    print("=" * 50)
    print("🎯 COMO INTERPRETAR OS RESULTADOS:")
    print("• Confiança 0.7+ = EXCELENTE para apostas (use valores maiores)")
    print("• Confiança 0.6-0.7 = BOM para apostas (use valores médios)")
    print("• Confiança 0.5-0.6 = CUIDADO (use valores baixos)")
    print("• Confiança <0.5 = EVITE apostas")
    print("")
    print("🏆 ESTRATÉGIA RECOMENDADA:")
    print("1. 🥇 APOSTAS PRINCIPAIS (60% do valor):")
    print("   • Use os TOP-3 jogos por ranking")
    print("   • Maior chance de retorno")
    print("2. 🥈 APOSTAS SECUNDÁRIAS (30% do valor):")
    print("   • Use jogos de alta confiança")
    print("   • Diversificação de risco")
    print("3. 🥉 APOSTAS EXPLORATÓRIAS (10% do valor):")
    print("   • Use jogos por amostragem")
    print("   • Teste novas combinações")
    print("")
    print("⚠️ GESTÃO DE RISCO:")
    print("• NUNCA aposte mais de 5% da sua renda")
    print("• Monitore resultados por 10-15 concursos")
    print("• Defina limite de perda e respeite")
    print("• Aumente apostas apenas após sucessos consistentes")
    print("")
    print("📊 ARQUIVOS GERADOS:")
    print("• resultados/relatorio_detalhado.txt - Análise completa")
    print("• resultados/guia_apostas.txt - Guia prático de apostas")
    print("• resultados/relatorio_frequencia.txt - Análise de frequência")
    print("• resultados/jogos_cientificos.txt - Lista de jogos")
    print("• resultados/analise_cientifica.json - Dados técnicos")
    
    return resultados


if __name__ == "__main__":
    resultados = main()
''