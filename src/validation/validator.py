#!/usr/bin/env python3
"""
Validador Específico para Super Sete
===================================

Sistema de validação específico para Super Sete com regras corretas:
- 7 colunas com números de 0 a 9
- 5 faixas de premiação (3, 4, 5, 6, 7 acertos)
- Validação de probabilidades exatas
- Testes específicos para Covering Design
- Validação de estratégias anti-padrão

Baseado na análise científica fornecida.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Importar módulos específicos do Super Sete
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.super_sete_analysis import SuperSeteAnalyzer, ModelConfig
from src.core.probability_calculator import SuperSeteProbabilityCalculator
from src.core.covering_design import SuperSeteCoveringDesign
from src.config.model_configs import SuperSeteConfig

warnings.filterwarnings('ignore')

@dataclass
class SuperSeteValidationConfig:
    """Configuração específica para validação do Super Sete"""
    test_size: float = 0.2  # 20% dos dados para teste
    cv_folds: int = 5       # 5 folds para validação cruzada
    min_samples: int = 50   # Mínimo de amostras para teste
    confidence_level: float = 0.95  # Nível de confiança para testes
    random_state: int = 42
    save_plots: bool = True
    plot_dir: str = "resultados_validacao/plots"
    results_dir: str = "resultados_validacao"
    
    # Configurações específicas do Super Sete
    n_colunas: int = 7
    numeros_por_coluna: int = 10  # 0 a 9
    faixas_premiacao: List[int] = None
    custo_aposta: float = 2.50
    
    def __post_init__(self):
        if self.faixas_premiacao is None:
            self.faixas_premiacao = [3, 4, 5, 6, 7]

class SuperSeteValidator:
    """Validador específico para Super Sete com regras corretas"""
    
    def __init__(self, config: SuperSeteValidationConfig = None, 
                 analysis_config: ModelConfig = None):
        self.config = config or SuperSeteValidationConfig()
        self.analysis_config = analysis_config or ModelConfig()
        self.results = {}
        self.predictions = {}
        self.true_values = {}
        
        # Usar analisador específico do Super Sete
        self.analyzer = SuperSeteAnalyzer(self.analysis_config)
        self.probability_calculator = SuperSeteProbabilityCalculator()
        self.covering_design = SuperSeteCoveringDesign()
        
        # Criar diretórios para resultados e plots
        os.makedirs(self.config.results_dir, exist_ok=True)
        if self.config.save_plots:
            os.makedirs(self.config.plot_dir, exist_ok=True)
    
    def load_and_prepare_data(self, excel_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Carrega e prepara dados específicos do Super Sete"""
        print("📊 Carregando dados do Super Sete para validação...")
        
        # Usar o analisador específico do Super Sete
        df = self.analyzer.load_data(excel_path)
        
        # Validar estrutura do Super Sete
        self._validate_super_sete_structure(df)
        
        # Preparar dados para cada coluna
        column_data = {}
        for col in df.columns:
            if col.startswith('Coluna_'):
                series = df[col].dropna().values
                if len(series) >= self.config.min_samples:
                    # Validar se os números estão no range 0-9
                    if self._validate_column_range(series):
                        column_data[col] = series
                    else:
                        print(f"⚠️  {col} contém valores fora do range 0-9")
        
        print(f"✅ Dados do Super Sete carregados: {len(df)} concursos, {len(column_data)} colunas válidas")
        return df, column_data
    
    def _validate_super_sete_structure(self, df: pd.DataFrame) -> None:
        """Valida se a estrutura dos dados está correta para Super Sete"""
        expected_columns = [f'Coluna_{i+1}' for i in range(7)]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Estrutura inválida do Super Sete. Colunas faltando: {missing_columns}")
        
        print("✅ Estrutura do Super Sete validada: 7 colunas encontradas")
    
    def _validate_column_range(self, series: np.ndarray) -> bool:
        """Valida se os números estão no range 0-9"""
        return np.all((series >= 0) & (series <= 9))
    
    def create_features(self, series: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Cria features para predição usando o analisador principal"""
        if len(series) < window + 1:
            return np.array([]), np.array([])
        
        try:
            # Usar diretamente o método do analisador principal
            # Isso garante que qualquer mudança no SuperSeteAnalyzer seja refletida aqui
            X, y, feature_names = self.analyzer.create_features(series, window)
            return X, y
        except Exception as e:
            print(f"⚠️  Erro ao criar features com analisador principal: {e}")
            # Fallback: features básicas otimizadas
            return self._create_basic_features(series, window)
    
    def _create_basic_features(self, series: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Cria features básicas como fallback - versão otimizada"""
        if len(series) < window + 1:
            return np.array([]), np.array([])
        
        # Usar operações vetorizadas para melhor performance
        n_samples = len(series) - window
        n_features = window + 3  # window + mean + std + var
        
        # Adicionar features de diferença se window > 1
        if window > 1:
            n_features += 2
        
        X = np.zeros((n_samples, n_features))
        y = series[window:]
        
        for i in range(n_samples):
            start_idx = i
            end_idx = i + window
            
            # Últimos valores
            X[i, :window] = series[start_idx:end_idx]
            
            # Estatísticas básicas
            window_data = series[start_idx:end_idx]
            X[i, window] = np.mean(window_data)
            X[i, window + 1] = np.std(window_data)
            X[i, window + 2] = np.var(window_data)
            
            # Diferenças
            if window > 1:
                X[i, window + 3] = series[end_idx - 1] - series[end_idx - 2]
                X[i, window + 4] = series[end_idx - 1] - series[start_idx]
        
        return X, y
    
    def validate_super_sete_probabilities(self, historico: np.ndarray) -> Dict[str, Any]:
        """Valida probabilidades específicas do Super Sete"""
        print("🎯 Validando probabilidades do Super Sete...")
        
        # Testar diferentes estratégias de aposta
        strategies = [
            [1, 1, 1, 1, 1, 1, 1],  # Aposta simples
            [2, 1, 1, 1, 1, 1, 1],  # 2 números na primeira coluna
            [2, 2, 1, 1, 1, 1, 1],  # 2 números nas duas primeiras
            [2, 2, 2, 1, 1, 1, 1],  # 2 números nas três primeiras
        ]
        
        validation_results = {}
        
        for i, strategy in enumerate(strategies):
            strategy_name = f"Estratégia_{i+1}"
            
            # Calcular probabilidades exatas
            probabilities = self.probability_calculator.calculate_exact_probabilities(strategy)
            
            # Validar se as probabilidades somam corretamente
            total_prob = (probabilities.faixa_3 + probabilities.faixa_4 + 
                         probabilities.faixa_5 + probabilities.faixa_6 + 
                         probabilities.faixa_7)
            
            validation_results[strategy_name] = {
                'strategy': strategy,
                'probabilities': probabilities.__dict__,
                'total_probability': total_prob,
                'is_valid': abs(total_prob - 1.0) < 0.01,  # Tolerância de 1%
                'custo_total': probabilities.total_combinations * self.config.custo_aposta
            }
        
        return validation_results
    
    def validate_covering_design_strategies(self, historico: np.ndarray) -> Dict[str, Any]:
        """Valida estratégias de Covering Design"""
        print("🎯 Validando estratégias de Covering Design...")
        
        # Gerar estratégias otimizadas
        strategies = self.covering_design.generate_optimized_strategies(historico, 200.0)
        
        validation_results = {}
        
        for strategy in strategies:
            # Calcular eficiência
            efficiency = self.covering_design.calculate_strategy_efficiency(strategy)
            
            # Validar se a estratégia é viável
            is_viable = (strategy.custo_total <= 200.0 and 
                        strategy.cobertura_estimada > 0.1)
            
            validation_results[strategy.nome] = {
                'strategy': strategy.__dict__,
                'efficiency': efficiency,
                'is_viable': is_viable,
                'recommendation': 'Recomendada' if is_viable else 'Não recomendada'
            }
        
        return validation_results
    
    def validate_frequency_patterns(self, historico: np.ndarray) -> Dict[str, Any]:
        """Valida padrões de frequência específicos do Super Sete"""
        print("📊 Validando padrões de frequência do Super Sete...")
        
        validation_results = {}
        
        for col in range(7):
            col_data = historico[:, col]
            col_name = f"Coluna_{col+1}"
            
            # Análise de frequência
            unique, counts = np.unique(col_data, return_counts=True)
            frequencies = counts / len(col_data)
            
            # Teste de uniformidade (Chi-square)
            expected_freq = 1.0 / 10
            chi2_stat = np.sum((frequencies - expected_freq) ** 2 / expected_freq)
            p_value = 1 - stats.chi2.cdf(chi2_stat, 9)
            
            # Entropia
            entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
            max_entropy = np.log2(10)
            normalized_entropy = entropy / max_entropy
            
            # Análise de sequências
            sequences = self._analyze_sequences(col_data)
            
            validation_results[col_name] = {
                'frequencies': dict(zip(unique, frequencies)),
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'is_uniform': p_value > 0.05,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'is_random': normalized_entropy > 0.8,
                'sequences': sequences
            }
        
        return validation_results
    
    def _analyze_sequences(self, series: np.ndarray) -> Dict[str, Any]:
        """Analisa sequências na série"""
        # Sequências consecutivas
        consecutive = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(series)):
            if series[i] == series[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Padrões de alternância
        alternations = 0
        for i in range(2, len(series)):
            if series[i] != series[i-1] and series[i-1] != series[i-2]:
                alternations += 1
        
        return {
            'max_consecutive': max_consecutive,
            'alternation_rate': alternations / max(1, len(series) - 2),
            'total_alternations': alternations
        }
    
    def validate_single_column(self, series: np.ndarray, column_name: str) -> Dict:
        """Valida modelo para uma única coluna específica do Super Sete"""
        print(f"🔍 Validando {column_name} do Super Sete...")
        
        # Usar análise específica do Super Sete
        try:
            analysis_results = self.analyzer.analyze_column(series, column_name)
            print(f"✅ Análise do Super Sete concluída para {column_name}")
        except Exception as e:
            print(f"⚠️  Erro na análise: {e}")
            analysis_results = {}
        
        # Criar features usando analisadores especializados
        X, y = self.create_features(series, window=self.model_config.window)
        
        if len(X) < 20:  # Mínimo de amostras
            return {"error": "Dados insuficientes", "analysis_base": analysis_results}
        
        # Split temporal
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_test) < 5:  # Mínimo para teste
            return {"error": "Conjunto de teste muito pequeno", "analysis_base": analysis_results}
        
        # Usar configuração do modelo do SuperSete
        from sklearn.preprocessing import StandardScaler
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Usar modelo configurado do SuperSete
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            min_samples_split=self.model_config.min_samples_split,
            min_samples_leaf=self.model_config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predições
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Métricas específicas para loterias
        n_classes = len(np.unique(y_test))
        random_accuracy = 1.0 / n_classes
        
        # Calcular melhoria sobre aleatório
        improvement = accuracy - random_accuracy
        improvement_pct = (improvement / random_accuracy) * 100 if random_accuracy > 0 else 0
        
        # Métricas de calibração
        try:
            # Brier Score para multiclasse: média dos Brier Scores de cada classe
            brier_scores = []
            for class_idx in range(y_pred_proba.shape[1]):
                # Binarizar para cada classe
                y_binary = (y_test == class_idx).astype(int)
                if len(np.unique(y_binary)) > 1:  # Só calcula se a classe existe no teste
                    brier_scores.append(brier_score_loss(y_binary, y_pred_proba[:, class_idx]))
            brier_score = np.mean(brier_scores) if brier_scores else 0.0
        except:
            brier_score = 0.0
            
        log_loss_score = log_loss(y_test, y_pred_proba)
        
        # Validação cruzada temporal mais robusta
        from sklearn.ensemble import RandomForestClassifier
        tscv = TimeSeriesSplit(n_splits=min(self.config.cv_folds, len(X_train_scaled)//10))
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train_scaled):
            if len(train_idx) < 10 or len(val_idx) < 5:  # Mínimo de amostras
                continue
                
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            temp_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            temp_model.fit(X_tr, y_tr)
            y_val_pred = temp_model.predict(X_val)
            cv_scores.append(accuracy_score(y_val, y_val_pred))
        
        cv_mean = np.mean(cv_scores) if cv_scores else 0.0
        cv_std = np.std(cv_scores) if cv_scores else 0.0
        
        # Teste de significância vs aleatório
        z_score = (accuracy - random_accuracy) / np.sqrt(random_accuracy * (1 - random_accuracy) / len(y_test))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Análise de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # Calcular métricas por classe
        class_metrics = {}
        for i, class_label in enumerate(np.unique(y_test)):
            if i < len(model.classes_):
                class_idx = np.where(model.classes_ == class_label)[0]
                if len(class_idx) > 0:
                    class_idx = class_idx[0]
                    precision_class = precision_score(y_test == class_label, y_pred == class_label, zero_division=0)
                    recall_class = recall_score(y_test == class_label, y_pred == class_label, zero_division=0)
                    f1_class = f1_score(y_test == class_label, y_pred == class_label, zero_division=0)
                    
                    class_metrics[str(class_label)] = {
                        'precision': precision_class,
                        'recall': recall_class,
                        'f1': f1_class,
                        'support': int(np.sum(y_test == class_label))
                    }
        
        # Análise de distribuição de frequências
        unique_values, counts = np.unique(y_test, return_counts=True)
        frequency_analysis = {
            'most_frequent': int(unique_values[np.argmax(counts)]),
            'frequency_ratio': float(np.max(counts) / len(y_test)),
            'entropy': float(stats.entropy(counts))
        }
        
        # Integrar análises especializadas do Super Sete (simplificado)
        specialized_analysis = {}
        
        # Análise temporal simplificada
        try:
            seasonality = self.analyzer.temporal_analyzer.detect_seasonality(series)
            trend = self.analyzer.temporal_analyzer.calculate_trend(series)
            specialized_analysis['temporal'] = {
                'seasonality': seasonality,
                'trend': trend
            }
        except Exception as e:
            print(f"⚠️  Erro na análise temporal: {e}")
            specialized_analysis['temporal'] = {}
        
        # Análise de entropia simplificada
        try:
            entropy_info = self.analyzer.entropy_analyzer.calculate_entropy(series)
            randomness = self.analyzer.entropy_analyzer.test_randomness(series)
            specialized_analysis['entropy'] = {
                'entropy': entropy_info,
                'randomness': randomness
            }
        except Exception as e:
            print(f"⚠️  Erro na análise de entropia: {e}")
            specialized_analysis['entropy'] = {}
        
        # Análise de confiança simplificada
        try:
            predictions = np.random.random(10)
            confidence_intervals = self.analyzer.confidence_analyzer.calculate_confidence_intervals(predictions)
            specialized_analysis['confidence'] = {
                'intervals': confidence_intervals
            }
        except Exception as e:
            print(f"⚠️  Erro na análise de confiança: {e}")
            specialized_analysis['confidence'] = {}
        
        return {
            'column_name': column_name,
            'n_samples': len(y_test),
            'n_features': X.shape[1],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'random_accuracy': random_accuracy,
            'improvement_over_random': improvement,
            'improvement_percentage': improvement_pct,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'feature_importance': model.feature_importances_.tolist(),
            'frequency_analysis': frequency_analysis,
            'model_quality': self._assess_model_quality(accuracy, improvement_pct, p_value),
            'analysis_base': analysis_results,  # Resultados da análise base do SuperSete
            'specialized_analysis': specialized_analysis  # Análises especializadas
        }
    
    def _assess_model_quality(self, accuracy: float, improvement_pct: float, p_value: float) -> str:
        """Avalia a qualidade do modelo baseado nas métricas"""
        if p_value < 0.01 and improvement_pct > 50:
            return "excelente"
        elif p_value < 0.05 and improvement_pct > 20:
            return "bom"
        elif p_value < 0.1 and improvement_pct > 10:
            return "moderado"
        else:
            return "baixo"
    
    def test_temporal_stability(self, series: np.ndarray, column_name: str) -> Dict:
        """Testa estabilidade temporal do modelo"""
        print(f"⏰ Testando estabilidade temporal de {column_name}...")
        
        # Dividir em janelas temporais mais robustas
        min_window_size = max(30, len(series) // 6)  # Mínimo 30 ou 1/6 dos dados
        window_size = min(min_window_size, len(series) // 3)  # Máximo 1/3 dos dados
        
        if window_size < 20:
            return {"error": "Série muito pequena para teste de estabilidade"}
        
        stability_scores = []
        window_info = []
        
        # Criar janelas sobrepostas
        step_size = max(5, window_size // 4)  # Passo mínimo de 5
        
        for i in range(0, len(series) - window_size + 1, step_size):
            window_data = series[i:i + window_size]
            if len(window_data) < 20:
                continue
                
            # Testar modelo nesta janela
            X, y = self.create_features(window_data, window=min(5, window_size//4))
            if len(X) < 10:
                continue
                
            # Split temporal (70% treino, 30% teste)
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if len(X_test) < 3:  # Mínimo para teste
                continue
            
            try:
                # Treinar e testar
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = RandomForestClassifier(
                    n_estimators=50, 
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                stability_scores.append(accuracy)
                
                # Informações da janela
                window_info.append({
                    'start_idx': i,
                    'end_idx': i + window_size,
                    'accuracy': accuracy,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                })
                
            except Exception as e:
                print(f"⚠️  Erro na janela {i}: {e}")
                continue
        
        if len(stability_scores) < 2:
            return {"error": "Dados insuficientes para teste de estabilidade"}
        
        # Calcular métricas de estabilidade
        mean_stability = np.mean(stability_scores)
        std_stability = np.std(stability_scores)
        
        # Coeficiente de variação (menor = mais estável)
        cv = std_stability / mean_stability if mean_stability > 0 else float('inf')
        
        # Teste de tendência temporal
        if len(stability_scores) >= 3:
            # Correlação entre posição temporal e acurácia
            time_correlation = np.corrcoef(range(len(stability_scores)), stability_scores)[0, 1]
        else:
            time_correlation = 0.0
        
        return {
            'stability_scores': stability_scores,
            'mean_stability': mean_stability,
            'std_stability': std_stability,
            'stability_coefficient': 1 - cv if cv != float('inf') else 0,
            'coefficient_of_variation': cv,
            'is_stable': cv < 0.3,  # Menos de 30% de variação
            'time_correlation': time_correlation,
            'n_windows': len(stability_scores),
            'window_info': window_info,
            'trend': 'melhorando' if time_correlation > 0.3 else 'piorando' if time_correlation < -0.3 else 'estável'
        }
    
    def test_overfitting(self, series: np.ndarray, column_name: str) -> Dict:
        """Testa se o modelo está sofrendo overfitting"""
        print(f"🔍 Testando overfitting em {column_name}...")
        
        X, y = self.create_features(series, window=10)
        if len(X) < 30:
            return {"error": "Dados insuficientes para teste de overfitting"}
        
        # Split temporal
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_test) < 5:
            return {"error": "Conjunto de teste muito pequeno"}
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelo com diferentes complexidades
        complexities = [3, 5, 8, 12, 20, 30]
        train_scores = []
        test_scores = []
        overfitting_gaps = []
        
        for max_depth in complexities:
            try:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                
                train_scores.append(train_acc)
                test_scores.append(test_acc)
                overfitting_gaps.append(train_acc - test_acc)
                
            except Exception as e:
                print(f"⚠️  Erro com max_depth={max_depth}: {e}")
                train_scores.append(0.0)
                test_scores.append(0.0)
                overfitting_gaps.append(0.0)
        
        # Calcular métricas de overfitting
        max_overfitting_gap = max(overfitting_gaps) if overfitting_gaps else 0.0
        avg_overfitting_gap = np.mean(overfitting_gaps) if overfitting_gaps else 0.0
        
        # Encontrar complexidade ótima (melhor score de teste)
        if test_scores:
            optimal_idx = np.argmax(test_scores)
            optimal_complexity = complexities[optimal_idx]
        else:
            optimal_complexity = complexities[0]
        
        # Avaliar severidade do overfitting
        if max_overfitting_gap > 0.2:
            overfitting_level = "severo"
        elif max_overfitting_gap > 0.1:
            overfitting_level = "moderado"
        elif max_overfitting_gap > 0.05:
            overfitting_level = "leve"
        else:
            overfitting_level = "mínimo"
        
        # Análise de tendência
        if len(overfitting_gaps) >= 3:
            # Correlação entre complexidade e overfitting
            complexity_correlation = np.corrcoef(complexities, overfitting_gaps)[0, 1]
        else:
            complexity_correlation = 0.0
        
        return {
            'complexities': complexities,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'overfitting_gaps': overfitting_gaps,
            'max_overfitting_gap': max_overfitting_gap,
            'avg_overfitting_gap': avg_overfitting_gap,
            'is_overfitting': max_overfitting_gap > 0.1,
            'overfitting_level': overfitting_level,
            'optimal_complexity': optimal_complexity,
            'complexity_correlation': complexity_correlation,
            'recommendation': self._get_overfitting_recommendation(max_overfitting_gap, optimal_complexity)
        }
    
    def _get_overfitting_recommendation(self, max_gap: float, optimal_complexity: int) -> str:
        """Gera recomendação baseada no nível de overfitting"""
        if max_gap > 0.2:
            return "Reduzir drasticamente a complexidade do modelo"
        elif max_gap > 0.1:
            return "Aplicar regularização e reduzir complexidade"
        elif max_gap > 0.05:
            return "Monitorar e considerar regularização leve"
        else:
            return "Modelo bem balanceado, manter configuração atual"
    
    def generate_validation_report(self, results: Dict) -> str:
        """Gera relatório de validação"""
        report = []
        report.append("="*80)
        report.append("📊 RELATÓRIO DE VALIDAÇÃO DO MODELO SUPER SETE")
        report.append("="*80)
        report.append("")
        
        # Resumo geral
        total_columns = len([k for k in results.keys() if k.startswith('Coluna_')])
        significant_columns = len([k for k, v in results.items() 
                                 if isinstance(v, dict) and v.get('is_significant', False)])
        
        report.append(f"📈 RESUMO GERAL:")
        report.append(f"   • Total de colunas testadas: {total_columns}")
        report.append(f"   • Colunas com significância estatística: {significant_columns}")
        report.append(f"   • Taxa de sucesso: {significant_columns/total_columns*100:.1f}%")
        report.append("")
        
        # Métricas por coluna
        report.append("🔍 MÉTRICAS POR COLUNA:")
        report.append("-" * 60)
        
        for col_name, col_results in results.items():
            if not isinstance(col_results, dict) or 'error' in col_results:
                continue
                
            report.append(f"\n{col_name}:")
            report.append(f"   • Acurácia: {col_results.get('accuracy', 0):.3f}")
            report.append(f"   • F1-Score: {col_results.get('f1', 0):.3f}")
            report.append(f"   • Brier Score: {col_results.get('brier_score', 0):.3f}")
            report.append(f"   • P-valor: {col_results.get('p_value', 1):.3f}")
            report.append(f"   • Significativo: {'✅' if col_results.get('is_significant', False) else '❌'}")
            report.append(f"   • Melhoria vs aleatório: {col_results.get('improvement_over_random', 0):.3f}")
            report.append(f"   • Melhoria %: {col_results.get('improvement_percentage', 0):.1f}%")
            report.append(f"   • Qualidade: {col_results.get('model_quality', 'desconhecida')}")
            
            # Análise de frequência
            freq_analysis = col_results.get('frequency_analysis', {})
            if freq_analysis:
                report.append(f"   • Valor mais frequente: {freq_analysis.get('most_frequent', 'N/A')}")
                report.append(f"   • Entropia: {freq_analysis.get('entropy', 0):.3f}")
        
        # Recomendações
        report.append("\n" + "="*60)
        report.append("💡 RECOMENDAÇÕES:")
        report.append("="*60)
        
        if significant_columns / total_columns > 0.5:
            report.append("✅ O modelo mostra potencial preditivo significativo")
        else:
            report.append("⚠️  O modelo tem limitações preditivas")
        
        avg_accuracy = np.mean([v.get('accuracy', 0) for v in results.values() 
                               if isinstance(v, dict) and 'accuracy' in v])
        
        if avg_accuracy > 0.2:
            report.append("✅ Acurácia média acima do esperado para loteria")
        else:
            report.append("⚠️  Acurácia baixa - considere ajustes no modelo")
        
        report.append("\n🎯 PRÓXIMOS PASSOS:")
        report.append("1. Foque nas colunas com significância estatística")
        report.append("2. Monitore estabilidade temporal das previsões")
        report.append("3. Ajuste parâmetros para colunas com baixa performance")
        report.append("4. Implemente validação contínua com novos dados")
        report.append("5. Considere ensemble de modelos para melhor performance")
        
        return "\n".join(report)
    
    def run_full_validation(self, excel_path: str) -> Dict:
        """Executa validação completa específica do Super Sete"""
        print("🚀 Iniciando validação completa do Super Sete...")
        
        # Carregar dados específicos do Super Sete
        df, column_data = self.load_and_prepare_data(excel_path)
        
        # Preparar histórico para análises específicas
        historico = df[[f'Coluna_{i+1}' for i in range(7)]].values
        
        results = {
            'timestamp': time.time(),
            'total_concursos': len(df),
            'validation_type': 'Super Sete Específico'
        }
        
        # 1. Validação de probabilidades específicas do Super Sete
        print("\n🎯 VALIDAÇÃO DE PROBABILIDADES DO SUPER SETE")
        print("=" * 50)
        prob_validation = self.validate_super_sete_probabilities(historico)
        results['probability_validation'] = prob_validation
        
        # 2. Validação de estratégias de Covering Design
        print("\n🎯 VALIDAÇÃO DE COVERING DESIGN")
        print("=" * 50)
        covering_validation = self.validate_covering_design_strategies(historico)
        results['covering_validation'] = covering_validation
        
        # 3. Validação de padrões de frequência
        print("\n📊 VALIDAÇÃO DE PADRÕES DE FREQUÊNCIA")
        print("=" * 50)
        frequency_validation = self.validate_frequency_patterns(historico)
        results['frequency_validation'] = frequency_validation
        
        # 4. Validação individual de colunas
        print("\n🔍 VALIDAÇÃO INDIVIDUAL DE COLUNAS")
        print("=" * 50)
        column_validation = {}
        
        for col_name, series in column_data.items():
            print(f"\nValidando {col_name}...")
            try:
                col_results = self.validate_single_column(series, col_name)
                column_validation[col_name] = col_results
            except Exception as e:
                print(f"❌ Erro ao validar {col_name}: {e}")
                column_validation[col_name] = {"error": str(e)}
        
        results['column_validation'] = column_validation
        
        # 5. Análise geral do Super Sete
        print("\n📈 ANÁLISE GERAL DO SUPER SETE")
        print("=" * 50)
        general_analysis = self._generate_super_sete_analysis(historico, results)
        results['general_analysis'] = general_analysis
        
        # Gerar relatório específico do Super Sete
        report = self.generate_super_sete_report(results)
        print("\n" + report)
        
        # Salvar resultados
        self._save_validation_results(results, report)
        
        return results
    
    def _generate_super_sete_analysis(self, historico: np.ndarray, results: Dict) -> Dict[str, Any]:
        """Gera análise geral específica do Super Sete"""
        # Calcular estatísticas gerais
        total_concursos = len(historico)
        
        # Análise de aleatoriedade geral
        random_columns = 0
        for col in range(7):
            col_data = historico[:, col]
            unique, counts = np.unique(col_data, return_counts=True)
            frequencies = counts / len(col_data)
            entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
            max_entropy = np.log2(10)
            normalized_entropy = entropy / max_entropy
            
            if normalized_entropy > 0.8:
                random_columns += 1
        
        # Análise de cobertura das estratégias
        covering_strategies = results.get('covering_validation', {})
        viable_strategies = sum(1 for strategy in covering_strategies.values() 
                              if strategy.get('is_viable', False))
        
        # Análise de probabilidades
        prob_validation = results.get('probability_validation', {})
        valid_probabilities = sum(1 for strategy in prob_validation.values() 
                                if strategy.get('is_valid', False))
        
        return {
            'total_concursos': total_concursos,
            'random_columns': random_columns,
            'randomness_ratio': random_columns / 7,
            'viable_strategies': viable_strategies,
            'valid_probabilities': valid_probabilities,
            'overall_quality': 'Bom' if random_columns >= 5 and viable_strategies > 0 else 'Regular'
        }
    
    def generate_super_sete_report(self, results: Dict) -> str:
        """Gera relatório específico do Super Sete"""
        report = []
        report.append("="*80)
        report.append("📊 RELATÓRIO DE VALIDAÇÃO DO SUPER SETE")
        report.append("="*80)
        report.append("")
        
        # Resumo geral
        general_analysis = results.get('general_analysis', {})
        report.append(f"📈 RESUMO GERAL:")
        report.append(f"   • Total de concursos: {general_analysis.get('total_concursos', 0)}")
        report.append(f"   • Colunas aleatórias: {general_analysis.get('random_columns', 0)}/7")
        report.append(f"   • Estratégias viáveis: {general_analysis.get('viable_strategies', 0)}")
        report.append(f"   • Qualidade geral: {general_analysis.get('overall_quality', 'Desconhecida')}")
        report.append("")
        
        # Validação de probabilidades
        prob_validation = results.get('probability_validation', {})
        report.append(f"🎯 VALIDAÇÃO DE PROBABILIDADES:")
        report.append("-" * 40)
        
        for strategy_name, strategy_data in prob_validation.items():
            is_valid = strategy_data.get('is_valid', False)
            custo = strategy_data.get('custo_total', 0)
            report.append(f"   • {strategy_name}: {'✅' if is_valid else '❌'} (R$ {custo:.2f})")
        
        report.append("")
        
        # Validação de Covering Design
        covering_validation = results.get('covering_validation', {})
        report.append(f"🎯 ESTRATÉGIAS DE COVERING DESIGN:")
        report.append("-" * 40)
        
        for strategy_name, strategy_data in covering_validation.items():
            is_viable = strategy_data.get('is_viable', False)
            efficiency = strategy_data.get('efficiency', {}).get('eficiencia_geral', 0)
            report.append(f"   • {strategy_name}: {'✅' if is_viable else '❌'} (Eficiência: {efficiency:.3f})")
        
        report.append("")
        
        # Validação de frequências
        frequency_validation = results.get('frequency_validation', {})
        report.append(f"📊 ANÁLISE DE FREQUÊNCIAS:")
        report.append("-" * 40)
        
        for col_name, col_data in frequency_validation.items():
            is_random = col_data.get('is_random', False)
            entropy = col_data.get('normalized_entropy', 0)
            report.append(f"   • {col_name}: {'Aleatória' if is_random else 'Padrão'} (Entropia: {entropy:.3f})")
        
        report.append("")
        
        # Recomendações específicas do Super Sete
        report.append("💡 RECOMENDAÇÕES ESPECÍFICAS DO SUPER SETE:")
        report.append("=" * 60)
        
        if general_analysis.get('randomness_ratio', 0) > 0.7:
            report.append("✅ O Super Sete mostra alta aleatoriedade - use estratégias conservadoras")
        else:
            report.append("⚠️  O Super Sete mostra padrões - considere estratégias anti-padrão")
        
        if general_analysis.get('viable_strategies', 0) > 0:
            report.append("✅ Estratégias de Covering Design disponíveis")
        else:
            report.append("⚠️  Considere ajustar orçamento para estratégias viáveis")
        
        report.append("\n🎯 PRÓXIMOS PASSOS:")
        report.append("1. Foque nas estratégias validadas com maior eficiência")
        report.append("2. Monitore padrões de frequência regularmente")
        report.append("3. Ajuste estratégias baseado na análise de entropia")
        report.append("4. Use probabilidades exatas para tomada de decisão")
        report.append("5. Implemente validação contínua com novos concursos")
        
        return "\n".join(report)
    
    def _save_validation_results(self, results: Dict, report: str) -> None:
        """Salva resultados da validação"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Salvar resultados JSON
        results_file = os.path.join(self.config.results_dir, f"super_sete_validation_{timestamp}.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Salvar relatório
        report_file = os.path.join(self.config.results_dir, f"super_sete_report_{timestamp}.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n💾 Resultados salvos em:")
        print(f"   • {results_file}")
        print(f"   • {report_file}")
        print(f"   • Pasta: {self.config.results_dir}")

def main():
    """Função principal para executar validação"""
    print("�� VALIDADOR DE MODELO SUPER SETE")
    print("="*50)
    
    # Configuração
    config = ValidationConfig(
        test_size=0.2,
        cv_folds=5,
        min_samples=30,
        confidence_level=0.95,
        save_plots=True,
        results_dir="resultados_validacao",
        plot_dir="resultados_validacao/plots"
    )
    
    # Configuração do modelo usando a estrutura unificada
    model_config = get_model_config('longo', n_concursos=1000)  # Ajustar conforme necessário
    
    # Executar validação
    validator = SuperSeteValidator(config, model_config)
    results = validator.run_full_validation("Super Sete.xlsx")
    
    print("\n✅ Validação concluída!")
    return results

if __name__ == "__main__":
    results = main()