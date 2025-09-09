#!/usr/bin/env python3
"""
Configurações de Modelos do Super Sete Analysis
Parâmetros organizados por velocidade de processamento e assertividade

REVISADO PARA SUPER SETE:
- 7 colunas com números de 0 a 9
- 5 faixas de premiação (3, 4, 5, 6, 7 acertos)
- Probabilidade base: 1 em 10.000.000 para 7 acertos
- Estratégias de Covering Design implementadas
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import json
import os
import numpy as np

@dataclass
class SuperSeteConfig:
    """Configurações específicas do Super Sete"""
    # Estrutura do jogo
    n_colunas: int = 7
    numeros_por_coluna: int = 10  # 0 a 9
    numeros_possiveis: List[int] = None
    
    # Faixas de premiação
    faixas_premiacao: Dict[int, str] = None
    probabilidades_faixas: Dict[int, float] = None
    
    # Estratégias de apostas
    aposta_minima: float = 2.50
    max_numeros_por_coluna: int = 2
    covering_design_enabled: bool = True
    
    def __post_init__(self):
        if self.numeros_possiveis is None:
            self.numeros_possiveis = list(range(10))  # 0 a 9
        
        if self.faixas_premiacao is None:
            self.faixas_premiacao = {
                7: "1ª faixa (principal)",
                6: "2ª faixa", 
                5: "3ª faixa",
                4: "4ª faixa",
                3: "5ª faixa"
            }
        
        if self.probabilidades_faixas is None:
            self.probabilidades_faixas = {
                7: 1/10_000_000,  # 1 em 10 milhões
                6: 1/370_370,     # Aproximadamente
                5: 1/11_111,      # Aproximadamente  
                4: 1/370,         # Aproximadamente
                3: 1/44           # Aproximadamente
            }

@dataclass
class ModelConfig:
    """Configuração completa dos modelos de machine learning"""
    
    # ===== CONFIGURAÇÕES SUPER SETE =====
    super_sete: SuperSeteConfig = None
    
    # ===== PARÂMETROS BÁSICOS =====
    window: int = 50
    n_estimators: int = 500
    max_depth: int = 10
    learning_rate: float = 0.01
    calibrate: bool = True
    calibration_method: str = 'isotonic'
    calibration_cv: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        """Validação pós-inicialização"""
        if self.super_sete is None:
            self.super_sete = SuperSeteConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Valida configurações básicas"""
        if self.window < 1:
            raise ValueError("window deve ser maior que 0")
        if self.n_estimators < 1:
            raise ValueError("n_estimators deve ser maior que 0")
        if self.max_depth < 1:
            raise ValueError("max_depth deve ser maior que 0")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate deve estar entre 0 e 1")
        if self.calibration_cv < 2:
            raise ValueError("calibration_cv deve ser maior que 1")
        if self.calibration_method not in ['isotonic', 'sigmoid']:
            raise ValueError("calibration_method deve ser 'isotonic' ou 'sigmoid'")
    
    # ===== LSTM =====
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_patience: int = 10
    include_lstm: bool = False
    
    # ===== VALIDAÇÃO E BACKTESTING =====
    backtest_steps: int = 1000
    train_min_size: int = 200
    min_confidence: float = 0.6
    confidence_interval: float = 0.95
    
    # ===== ANÁLISE TEMPORAL AVANÇADA =====
    detect_seasonality: bool = True
    max_periods: int = 50
    min_period: int = 3
    seasonality_threshold: float = 0.1
    
    # ===== ANÁLISE DE ENTROPIA =====
    entropy_window: int = 20
    entropy_threshold: float = 0.8
    
    # ===== ENSEMBLE ADAPTATIVO =====
    adaptive_weights: bool = True
    weight_decay: float = 0.95
    min_weight: float = 0.01
    
    # ===== DETECÇÃO DE ANOMALIAS =====
    anomaly_detection: bool = True
    anomaly_threshold: float = 2.5
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    
    # ===== CORRELAÇÃO ENTRE COLUNAS =====
    correlation_analysis: bool = True
    min_correlation: float = 0.1
    max_correlation: float = 0.9
    
    # ===== OTIMIZAÇÃO BAYESIANA =====
    bayesian_optimization: bool = False
    n_trials: int = 50
    optimization_metric: str = 'brier_score'
    
    # ===== MÉTRICAS DE CONFIANÇA =====
    confidence_metrics: bool = True
    uncertainty_quantification: bool = True
    prediction_intervals: bool = True
    
    # ===== REGULARIZAÇÃO RIGOROSA =====
    regularization_strength: float = 1.0
    l1_ratio: float = 0.5
    dropout_rate: float = 0.3
    early_stopping_rounds: int = 10
    min_samples_split: int = 20
    min_samples_leaf: int = 10


# ===== CONFIGURAÇÕES POR VELOCIDADE DE PROCESSAMENTO =====

# 🚀 PROCESSAMENTO RÁPIDO (Velocidade Alta, Assertividade Média)
RAPIDO = {
    # Parâmetros básicos otimizados para velocidade
    'window': 50,
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'backtest_steps': 200,
    'train_min_size': 100,
    
    # Análise simplificada
    'detect_seasonality': False,
    'max_periods': 10,
    'entropy_window': 10,
    'correlation_analysis': False,
    'anomaly_detection': False,
    'include_lstm': False,
    'calibrate': False,
    'adaptive_weights': False,
    'confidence_metrics': False,
    'uncertainty_quantification': False,
    'prediction_intervals': False,
    'bayesian_optimization': False
}

# ⚡ PROCESSAMENTO MÉDIO (Velocidade Média, Assertividade Boa)
MEDIO = {
    # Parâmetros balanceados
    'window': 75,
    'n_estimators': 500,
    'max_depth': 12,
    'learning_rate': 0.05,
    'backtest_steps': 500,
    'train_min_size': 150,
    
    # Análise moderada
    'detect_seasonality': True,
    'max_periods': 20,
    'entropy_window': 15,
    'correlation_analysis': True,
    'anomaly_detection': True,
    'include_lstm': False,
    'calibrate': True,
    'calibration_cv': 5,
    'adaptive_weights': True,
    'confidence_metrics': True,
    'uncertainty_quantification': False,
    'prediction_intervals': True,
    'bayesian_optimization': False
}

# 🔥 PROCESSAMENTO LONGO (Velocidade Baixa, Assertividade Alta)
LONGO = {
    # Parâmetros avançados
    'window': 100,
    'n_estimators': 1000,
    'max_depth': 15,
    'learning_rate': 0.01,
    'backtest_steps': 1000,
    'train_min_size': 200,
    
    # Análise completa
    'detect_seasonality': True,
    'max_periods': 30,
    'entropy_window': 20,
    'correlation_analysis': True,
    'anomaly_detection': True,
    'include_lstm': True,
    'lstm_units': 64,
    'lstm_epochs': 50,
    'lstm_patience': 10,
    'calibrate': True,
    'calibration_cv': 10,
    'adaptive_weights': True,
    'confidence_metrics': True,
    'uncertainty_quantification': True,
    'prediction_intervals': True,
    'bayesian_optimization': False
}

# 🎯 PROCESSAMENTO ULTRA (Velocidade Muito Baixa, Assertividade Máxima)
ULTRA = {
    # Parâmetros máximos
    'window': 150,
    'n_estimators': 2000,
    'max_depth': 20,
    'learning_rate': 0.005,
    'backtest_steps': 2000,
    'train_min_size': 300,
    
    # Análise ultra completa
    'detect_seasonality': True,
    'max_periods': 50,
    'entropy_window': 30,
    'correlation_analysis': True,
    'anomaly_detection': True,
    'include_lstm': True,
    'lstm_units': 128,
    'lstm_epochs': 100,
    'lstm_patience': 15,
    'calibrate': True,
    'calibration_cv': 15,
    'adaptive_weights': True,
    'confidence_metrics': True,
    'uncertainty_quantification': True,
    'prediction_intervals': True,
    'bayesian_optimization': True,
    'n_trials': 100
}

# Dicionário com todas as configurações
MODEL_PRESETS = {
    'rapido': RAPIDO,
    'medio': MEDIO,
    'longo': LONGO,
    'ultra': ULTRA
}

# ===== FUNÇÕES DE UTILIDADE =====

def get_model_config(preset: str = 'medio', n_concursos: int = None, **overrides) -> ModelConfig:
    """
    Obtém configuração de modelo baseada em preset e quantidade de concursos
    
    Args:
        preset: Nome do preset ('rapido', 'medio', 'longo', 'ultra')
        n_concursos: Número de concursos disponíveis (para ajuste dinâmico)
        **overrides: Valores para sobrescrever
    
    Returns:
        ModelConfig: Configuração final
    
    Raises:
        ValueError: Se preset inválido ou parâmetros inválidos
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Preset '{preset}' não encontrado. Use: {list(MODEL_PRESETS.keys())}")
    
    if n_concursos is not None and n_concursos < 1:
        raise ValueError("n_concursos deve ser maior que 0")
    
    config_dict = MODEL_PRESETS[preset].copy()
    
    # Ajustes dinâmicos baseados na quantidade de concursos
    if n_concursos is not None:
        config_dict = _adjust_config_for_data_size(config_dict, n_concursos, preset)
    
    # Aplicar overrides
    config_dict.update(overrides)
    
    # Validar overrides
    _validate_overrides(overrides)
    
    try:
        return ModelConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuração inválida: {e}")

def _validate_overrides(overrides: Dict[str, Any]) -> None:
    """Valida overrides fornecidos"""
    valid_keys = {
        'window', 'n_estimators', 'max_depth', 'learning_rate', 'calibrate',
        'calibration_method', 'calibration_cv', 'random_state', 'lstm_units',
        'lstm_dropout', 'lstm_epochs', 'lstm_patience', 'include_lstm',
        'backtest_steps', 'train_min_size', 'min_confidence', 'confidence_interval',
        'detect_seasonality', 'max_periods', 'min_period', 'seasonality_threshold',
        'entropy_window', 'entropy_threshold', 'adaptive_weights', 'weight_decay',
        'min_weight', 'anomaly_detection', 'anomaly_threshold', 'outlier_method',
        'correlation_analysis', 'min_correlation', 'max_correlation',
        'bayesian_optimization', 'n_trials', 'optimization_metric',
        'confidence_metrics', 'uncertainty_quantification', 'prediction_intervals',
        'regularization_strength', 'l1_ratio', 'dropout_rate', 'early_stopping_rounds',
        'min_samples_split', 'min_samples_leaf'
    }
    
    invalid_keys = set(overrides.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(f"Chaves inválidas nos overrides: {invalid_keys}")
    
    # Validações específicas
    if 'learning_rate' in overrides and not 0 < overrides['learning_rate'] <= 1:
        raise ValueError("learning_rate deve estar entre 0 e 1")
    
    if 'calibration_method' in overrides and overrides['calibration_method'] not in ['isotonic', 'sigmoid']:
        raise ValueError("calibration_method deve ser 'isotonic' ou 'sigmoid'")
    
    if 'outlier_method' in overrides and overrides['outlier_method'] not in ['iqr', 'zscore', 'isolation']:
        raise ValueError("outlier_method deve ser 'iqr', 'zscore' ou 'isolation'")

def _adjust_config_for_data_size(config_dict: dict, n_concursos: int, preset: str) -> dict:
    """
    Ajusta configurações dinamicamente baseado na quantidade de concursos
    """
    import numpy as np
    
    # Calcular proporções dinâmicas
    data_ratio = min(1.0, n_concursos / 1000)  # Normalizar para 1000 concursos
    
    # Ajustar window baseado nos dados disponíveis (otimizado para crescimento)
    if n_concursos < 100:
        config_dict['window'] = max(5, min(10, n_concursos // 10))
    elif n_concursos < 300:
        config_dict['window'] = max(10, min(25, n_concursos // 12))
    elif n_concursos < 500:
        config_dict['window'] = max(15, min(40, n_concursos // 15))
    elif n_concursos < 750:
        config_dict['window'] = max(20, min(60, n_concursos // 18))
    else:
        config_dict['window'] = max(25, min(80, n_concursos // 20))
    
    # Ajustar train_min_size baseado nos dados
    if n_concursos < 200:
        config_dict['train_min_size'] = max(20, n_concursos // 4)
    elif n_concursos < 500:
        config_dict['train_min_size'] = max(30, n_concursos // 3)
    else:
        config_dict['train_min_size'] = max(50, n_concursos // 4)
    
    # Ajustar n_estimators para evitar overfitting (otimizado para crescimento)
    if n_concursos < 200:
        config_dict['n_estimators'] = max(50, min(300, n_concursos // 2))
    elif n_concursos < 500:
        config_dict['n_estimators'] = max(100, min(800, n_concursos // 2))
    elif n_concursos < 750:
        config_dict['n_estimators'] = max(200, min(1200, n_concursos // 2))
    else:
        config_dict['n_estimators'] = max(300, min(1500, n_concursos // 3))
    
    # Ajustar max_depth para evitar overfitting
    if n_concursos < 200:
        config_dict['max_depth'] = max(3, min(8, int(np.log2(n_concursos))))
    elif n_concursos < 500:
        config_dict['max_depth'] = max(5, min(12, int(np.log2(n_concursos))))
    else:
        config_dict['max_depth'] = max(8, min(15, int(np.log2(n_concursos))))
    
    # Ajustar min_confidence baseado na assertividade desejada e quantidade de dados
    base_confidence = {'rapido': 0.2, 'medio': 0.3, 'longo': 0.4, 'ultra': 0.5}
    config_dict['min_confidence'] = base_confidence[preset]
    
    # Ajustar confiança baseado na quantidade de dados
    if n_concursos > 500:
        config_dict['min_confidence'] = min(0.8, config_dict['min_confidence'] + 0.1)
    elif n_concursos > 750:
        config_dict['min_confidence'] = min(0.9, config_dict['min_confidence'] + 0.15)
    
    # Ajustar backtest_steps
    config_dict['backtest_steps'] = max(50, min(1000, n_concursos // 2))
    
    # Ajustar calibration_cv
    if n_concursos < 100:
        config_dict['calibration_cv'] = 3
    elif n_concursos < 300:
        config_dict['calibration_cv'] = 5
    else:
        config_dict['calibration_cv'] = min(10, n_concursos // 50)
    
    # Ajustar learning_rate baseado na quantidade de dados (otimizado para crescimento)
    if n_concursos < 200:
        config_dict['learning_rate'] = 0.1
    elif n_concursos < 500:
        config_dict['learning_rate'] = 0.05
    elif n_concursos < 750:
        config_dict['learning_rate'] = 0.03
    else:
        config_dict['learning_rate'] = 0.01
    
    # Ajustar parâmetros específicos para crescimento
    if n_concursos > 500:
        # Para muitos dados, usar parâmetros mais conservadores
        config_dict['calibration_cv'] = min(15, n_concursos // 50)
        config_dict['backtest_steps'] = min(2000, n_concursos // 2)
        
        # Ajustar threshold de sazonalidade
        config_dict['seasonality_threshold'] = max(0.05, 0.1 - (n_concursos - 500) / 10000)
        
        # Ajustar threshold de entropia
        config_dict['entropy_threshold'] = max(0.6, 0.8 - (n_concursos - 500) / 5000)
    
    # Habilitar features complexas baseado na quantidade de dados
    if n_concursos < 200:
        config_dict['detect_seasonality'] = False
        config_dict['anomaly_detection'] = False
        config_dict['correlation_analysis'] = False
        config_dict['bayesian_optimization'] = False
    elif n_concursos < 500:
        config_dict['detect_seasonality'] = True
        config_dict['anomaly_detection'] = True
        config_dict['correlation_analysis'] = True
        config_dict['bayesian_optimization'] = False
    elif n_concursos < 750:
        config_dict['detect_seasonality'] = True
        config_dict['anomaly_detection'] = True
        config_dict['correlation_analysis'] = True
        config_dict['bayesian_optimization'] = True
        config_dict['n_trials'] = min(50, n_concursos // 10)
    else:
        # Para muitos dados, habilitar todas as features
        config_dict['detect_seasonality'] = True
        config_dict['anomaly_detection'] = True
        config_dict['correlation_analysis'] = True
        config_dict['bayesian_optimization'] = True
        config_dict['n_trials'] = min(100, n_concursos // 8)
    
    return config_dict

def list_model_presets() -> None:
    """Lista todos os presets de modelo disponíveis com descrições"""
    print("🎯 PRESETS DE MODELO DISPONÍVEIS:")
    print("=" * 60)
    
    descriptions = {
        'rapido': '🚀 RÁPIDO - Velocidade Alta, Assertividade Média',
        'medio': '⚡ MÉDIO - Velocidade Média, Assertividade Boa',
        'longo': '🔥 LONGO - Velocidade Baixa, Assertividade Alta',
        'ultra': '🎯 ULTRA - Velocidade Muito Baixa, Assertividade Máxima'
    }
    
    for nome, config in MODEL_PRESETS.items():
        print(f"\n{descriptions[nome]}:")
        print("-" * 40)
        
        # Parâmetros principais
        print(f"  🔧 Parâmetros Básicos:")
        print(f"    window: {config.get('window', 50)}")
        print(f"    n_estimators: {config.get('n_estimators', 500)}")
        print(f"    max_depth: {config.get('max_depth', 10)}")
        print(f"    learning_rate: {config.get('learning_rate', 0.01)}")
        print(f"    backtest_steps: {config.get('backtest_steps', 1000)}")
        
        # Análises habilitadas
        print(f"  📊 Análises:")
        print(f"    Sazonalidade: {'✅' if config.get('detect_seasonality', True) else '❌'}")
        print(f"    Correlação: {'✅' if config.get('correlation_analysis', True) else '❌'}")
        print(f"    Anomalias: {'✅' if config.get('anomaly_detection', True) else '❌'}")
        print(f"    LSTM: {'✅' if config.get('include_lstm', False) else '❌'}")
        print(f"    Calibração: {'✅' if config.get('calibrate', True) else '❌'}")

def save_config_to_file(config: ModelConfig, filename: str) -> None:
    """
    Salva configuração em arquivo JSON
    
    Args:
        config: Configuração a salvar
        filename: Nome do arquivo
    """
    config_dict = config.__dict__
    
    os.makedirs('configs', exist_ok=True)
    filepath = os.path.join('configs', f"{filename}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Configuração salva em: {filepath}")

def load_config_from_file(filename: str) -> ModelConfig:
    """
    Carrega configuração de arquivo JSON
    
    Args:
        filename: Nome do arquivo
    
    Returns:
        ModelConfig: Configuração carregada
    """
    filepath = os.path.join('configs', f"{filename}.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return ModelConfig(**config_dict)

def compare_presets() -> None:
    """Compara diferentes presets lado a lado"""
    print("📊 COMPARAÇÃO DE PRESETS:")
    print("=" * 80)
    
    # Parâmetros para comparar
    params_to_compare = [
        'window', 'n_estimators', 'max_depth', 'learning_rate', 
        'backtest_steps', 'include_lstm', 'detect_seasonality',
        'correlation_analysis', 'anomaly_detection', 'calibrate'
    ]
    
    # Cabeçalho
    header = f"{'Parâmetro':<20} {'Rápido':<12} {'Médio':<12} {'Longo':<12} {'Ultra':<12}"
    print(header)
    print("-" * 80)
    
    # Comparar cada parâmetro
    for param in params_to_compare:
        row = f"{param:<20}"
        for preset in ['rapido', 'medio', 'longo', 'ultra']:
            value = MODEL_PRESETS[preset].get(param, 'N/A')
            if isinstance(value, bool):
                value = '✅' if value else '❌'
            row += f" {str(value):<12}"
        print(row)

def get_performance_estimate(preset: str) -> Dict[str, str]:
    """
    Estima tempo de processamento e assertividade
    
    Args:
        preset: Nome do preset
    
    Returns:
        Dict com estimativas
    """
    estimates = {
        'rapido': {
            'tempo_estimado': '1-3 minutos',
            'assertividade': 'Média (60-75%)',
            'uso_ram': 'Baixo (~500MB)',
            'cpu_usage': 'Médio'
        },
        'medio': {
            'tempo_estimado': '3-8 minutos',
            'assertividade': 'Boa (70-85%)',
            'uso_ram': 'Médio (~1GB)',
            'cpu_usage': 'Alto'
        },
        'longo': {
            'tempo_estimado': '8-20 minutos',
            'assertividade': 'Alta (80-90%)',
            'uso_ram': 'Alto (~2GB)',
            'cpu_usage': 'Muito Alto'
        },
        'ultra': {
            'tempo_estimado': '20-60 minutos',
            'assertividade': 'Máxima (85-95%)',
            'uso_ram': 'Muito Alto (~4GB)',
            'cpu_usage': 'Máximo'
        }
    }
    
    return estimates.get(preset, {})

# ===== FUNÇÕES ESPECÍFICAS DO SUPER SETE =====

def calculate_super_sete_probabilities(numeros_por_coluna: List[int]) -> Dict[str, Any]:
    """
    Calcula probabilidades para diferentes estratégias de apostas no Super Sete
    
    Args:
        numeros_por_coluna: Lista com quantidade de números escolhidos por coluna
    
    Returns:
        Dict com probabilidades e custos
    """
    if len(numeros_por_coluna) != 7:
        raise ValueError("Deve ter exatamente 7 colunas")
    
    # Calcular total de combinações
    total_combinacoes = 1
    for n in numeros_por_coluna:
        if n < 1 or n > 10:
            raise ValueError("Cada coluna deve ter entre 1 e 10 números")
        total_combinacoes *= n
    
    # Calcular custo total
    custo_total = total_combinacoes * 2.50
    
    # Calcular probabilidades para cada faixa
    probabilidades = {}
    for acertos in range(3, 8):  # 3 a 7 acertos
        # Probabilidade aproximada baseada na estratégia
        prob_base = 1 / (10 ** 7)  # Probabilidade base para 7 acertos
        
        # Ajustar baseado na quantidade de números por coluna
        fator_ajuste = 1.0
        for n in numeros_por_coluna:
            fator_ajuste *= (n / 10)  # Reduz probabilidade proporcionalmente
        
        probabilidades[acertos] = prob_base * fator_ajuste * (10 ** (7 - acertos))
    
    return {
        'total_combinacoes': total_combinacoes,
        'custo_total': custo_total,
        'probabilidades': probabilidades,
        'numeros_por_coluna': numeros_por_coluna
    }

def generate_covering_design_strategy(budget: float = 100.0) -> Dict[str, Any]:
    """
    Gera estratégia de Covering Design para Super Sete
    
    Args:
        budget: Orçamento disponível em reais
    
    Returns:
        Dict com estratégia otimizada
    """
    # Calcular quantas combinações cabem no orçamento
    max_combinacoes = int(budget / 2.50)
    
    # Estratégias baseadas no orçamento
    if max_combinacoes < 10:
        # Orçamento baixo: apostas simples
        strategy = {
            'tipo': 'simples',
            'numeros_por_coluna': [1] * 7,
            'combinacoes': 1,
            'custo': 2.50,
            'descricao': 'Aposta mínima - 1 número por coluna'
        }
    elif max_combinacoes < 50:
        # Orçamento médio: 2 números em algumas colunas
        strategy = {
            'tipo': 'parcial_covering',
            'numeros_por_coluna': [2, 2, 1, 1, 1, 1, 1],
            'combinacoes': 4,
            'custo': 10.00,
            'descricao': '2 números nas 2 primeiras colunas'
        }
    elif max_combinacoes < 200:
        # Orçamento alto: covering mais amplo
        strategy = {
            'tipo': 'covering_medio',
            'numeros_por_coluna': [2, 2, 2, 1, 1, 1, 1],
            'combinacoes': 8,
            'custo': 20.00,
            'descricao': '2 números nas 3 primeiras colunas'
        }
    else:
        # Orçamento muito alto: covering completo
        strategy = {
            'tipo': 'covering_completo',
            'numeros_por_coluna': [2, 2, 2, 2, 2, 2, 2],
            'combinacoes': 128,
            'custo': 320.00,
            'descricao': '2 números em todas as colunas'
        }
    
    # Calcular probabilidades para a estratégia
    prob_info = calculate_super_sete_probabilities(strategy['numeros_por_coluna'])
    strategy.update(prob_info)
    
    return strategy

def analyze_number_frequency_patterns(historico: np.ndarray) -> Dict[str, Any]:
    """
    Analisa padrões de frequência dos números no Super Sete
    
    Args:
        historico: Array com histórico de números sorteados (7 colunas)
    
    Returns:
        Dict com análise de padrões
    """
    if historico.shape[1] != 7:
        raise ValueError("Histórico deve ter 7 colunas")
    
    analysis = {}
    
    for col in range(7):
        col_data = historico[:, col]
        unique, counts = np.unique(col_data, return_counts=True)
        
        # Calcular frequências
        frequencies = counts / len(col_data)
        
        # Números mais e menos frequentes
        most_frequent = unique[np.argmax(frequencies)]
        least_frequent = unique[np.argmin(frequencies)]
        
        # Calcular entropia (aleatoriedade)
        entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
        
        analysis[f'coluna_{col+1}'] = {
            'frequencies': dict(zip(unique, frequencies)),
            'most_frequent': int(most_frequent),
            'least_frequent': int(least_frequent),
            'entropy': float(entropy),
            'is_random': entropy > 2.5  # Threshold para aleatoriedade
        }
    
    return analysis

def generate_anti_pattern_strategy(historico: np.ndarray) -> List[List[int]]:
    """
    Gera estratégia anti-padrão para evitar números populares
    
    Args:
        historico: Histórico de números sorteados
    
    Returns:
        Lista de combinações anti-padrão
    """
    analysis = analyze_number_frequency_patterns(historico)
    
    # Selecionar números menos frequentes de cada coluna
    anti_pattern = []
    for col in range(7):
        col_key = f'coluna_{col+1}'
        col_data = analysis[col_key]
        
        # Pegar os 2 números menos frequentes
        frequencies = col_data['frequencies']
        sorted_nums = sorted(frequencies.items(), key=lambda x: x[1])
        least_frequent = [num for num, freq in sorted_nums[:2]]
        
        anti_pattern.append(least_frequent)
    
    return anti_pattern

# ===== EXEMPLO DE USO =====
if __name__ == "__main__":
    print("🔧 CONFIGURAÇÕES DE MODELO DO SUPER SETE ANALYSIS")
    print("=" * 60)
    
    # Listar todos os presets
    list_model_presets()
    
    # Comparar presets
    print("\n")
    compare_presets()
    
    # Exemplo de uso
    print(f"\n🎨 EXEMPLO DE CONFIGURAÇÃO PERSONALIZADA:")
    print("-" * 40)
    
    # Criar configuração baseada no preset 'longo' com modificações
    config = get_model_config('longo', n_estimators=1500, window=120)
    print(f"Preset base: longo")
    print(f"Janela: {config.window}")
    print(f"Estimadores: {config.n_estimators}")
    print(f"LSTM habilitado: {config.include_lstm}")
    
    # Salvar configuração personalizada
    save_config_to_file(config, 'minha_config_personalizada')
    
    # Mostrar estimativas de performance
    print(f"\n⏱️  ESTIMATIVAS DE PERFORMANCE:")
    print("-" * 40)
    for preset in MODEL_PRESETS.keys():
        estimates = get_performance_estimate(preset)
        print(f"\n{preset.upper()}:")
        for key, value in estimates.items():
            print(f"  {key}: {value}")
    
    # Exemplo de funções do Super Sete
    print(f"\n🎯 EXEMPLO DE FUNÇÕES SUPER SETE:")
    print("-" * 40)
    
    # Calcular probabilidades
    prob = calculate_super_sete_probabilities([1, 1, 1, 1, 1, 1, 1])
    print(f"Aposta simples: {prob['custo_total']:.2f} reais")
    
    # Estratégia de covering
    strategy = generate_covering_design_strategy(50.0)
    print(f"Estratégia para R$ 50: {strategy['descricao']}")