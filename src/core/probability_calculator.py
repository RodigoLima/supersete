#!/usr/bin/env python3
"""
Calculadora de Probabilidades do Super Sete
==========================================

Módulo especializado para cálculos de probabilidades e análise estatística
específica do Super Sete, incluindo todas as 5 faixas de premiação.

Baseado na análise científica fornecida:
- 7 colunas com números de 0 a 9
- 5 faixas de premiação (3, 4, 5, 6, 7 acertos)
- Probabilidades exatas calculadas matematicamente
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy.special import comb
from scipy.stats import binom
import math

@dataclass
class SuperSeteProbabilities:
    """Estrutura para armazenar probabilidades do Super Sete"""
    faixa_7: float  # 1ª faixa (principal)
    faixa_6: float  # 2ª faixa
    faixa_5: float  # 3ª faixa
    faixa_4: float  # 4ª faixa
    faixa_3: float  # 5ª faixa
    total_combinations: int
    cost_per_combination: float = 2.50

class SuperSeteProbabilityCalculator:
    """Calculadora de probabilidades para Super Sete"""
    
    def __init__(self):
        self.n_colunas = 7
        self.numeros_por_coluna = 10  # 0 a 9
        self.custo_aposta = 2.50
        
    def calculate_exact_probabilities(self, numeros_por_coluna: List[int]) -> SuperSeteProbabilities:
        """
        Calcula probabilidades exatas para uma estratégia de aposta
        
        Args:
            numeros_por_coluna: Lista com quantidade de números escolhidos por coluna
            
        Returns:
            SuperSeteProbabilities: Probabilidades calculadas
        """
        if len(numeros_por_coluna) != 7:
            raise ValueError("Deve ter exatamente 7 colunas")
        
        # Calcular total de combinações
        total_combinations = 1
        for n in numeros_por_coluna:
            if n < 1 or n > 10:
                raise ValueError("Cada coluna deve ter entre 1 e 10 números")
            total_combinations *= n
        
        # Calcular probabilidades para cada faixa
        probabilities = {}
        
        for acertos in range(3, 8):  # 3 a 7 acertos
            prob = self._calculate_faixa_probability(numeros_por_coluna, acertos)
            probabilities[f'faixa_{acertos}'] = prob
        
        return SuperSeteProbabilities(
            faixa_7=probabilities['faixa_7'],
            faixa_6=probabilities['faixa_6'],
            faixa_5=probabilities['faixa_5'],
            faixa_4=probabilities['faixa_4'],
            faixa_3=probabilities['faixa_3'],
            total_combinations=total_combinations
        )
    
    def _calculate_faixa_probability(self, numeros_por_coluna: List[int], acertos: int) -> float:
        """
        Calcula probabilidade para uma faixa específica
        
        Args:
            numeros_por_coluna: Números escolhidos por coluna
            acertos: Número de acertos desejado (3-7)
            
        Returns:
            float: Probabilidade de acertar exatamente 'acertos' números
        """
        if acertos < 3 or acertos > 7:
            return 0.0
        
        # Para cada combinação possível de acertos
        total_prob = 0.0
        
        # Gerar todas as combinações de colunas que podem ser acertadas
        from itertools import combinations
        
        for colunas_acertadas in combinations(range(7), acertos):
            # Calcular probabilidade desta combinação específica
            prob_esta_combinacao = 1.0
            
            for col in range(7):
                if col in colunas_acertadas:
                    # Esta coluna foi acertada
                    # Probabilidade = (números escolhidos) / 10
                    prob_esta_combinacao *= (numeros_por_coluna[col] / 10)
                else:
                    # Esta coluna não foi acertada
                    # Probabilidade = (10 - números escolhidos) / 10
                    prob_esta_combinacao *= ((10 - numeros_por_coluna[col]) / 10)
            
            total_prob += prob_esta_combinacao
        
        return total_prob
    
    def calculate_expected_value(self, probabilities: SuperSeteProbabilities, 
                               premios: Dict[int, float]) -> Dict[str, float]:
        """
        Calcula valor esperado para cada faixa
        
        Args:
            probabilities: Probabilidades calculadas
            premios: Valores dos prêmios por faixa
            
        Returns:
            Dict com valores esperados
        """
        expected_values = {}
        
        faixas = {
            7: probabilities.faixa_7,
            6: probabilities.faixa_6,
            5: probabilities.faixa_5,
            4: probabilities.faixa_4,
            3: probabilities.faixa_3
        }
        
        for faixa, prob in faixas.items():
            premio = premios.get(faixa, 0.0)
            expected_values[f'faixa_{faixa}'] = prob * premio
        
        # Valor esperado total
        expected_values['total'] = sum(expected_values.values())
        
        # ROI (Return on Investment)
        custo_total = probabilities.total_combinations * self.custo_aposta
        expected_values['roi'] = expected_values['total'] / custo_total if custo_total > 0 else 0
        
        return expected_values
    
    def analyze_frequency_patterns(self, historico: np.ndarray) -> Dict[str, Dict]:
        """
        Analisa padrões de frequência nos dados históricos
        
        Args:
            historico: Array com histórico de sorteios (n_sorteios x 7)
            
        Returns:
            Dict com análise de frequências por coluna
        """
        if historico.shape[1] != 7:
            raise ValueError("Histórico deve ter 7 colunas")
        
        analysis = {}
        
        for col in range(7):
            col_data = historico[:, col]
            
            # Frequências absolutas
            unique, counts = np.unique(col_data, return_counts=True)
            frequencies = counts / len(col_data)
            
            # Estatísticas
            most_frequent = unique[np.argmax(frequencies)]
            least_frequent = unique[np.argmin(frequencies)]
            
            # Entropia (medida de aleatoriedade)
            entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
            
            # Teste de uniformidade (Chi-square)
            expected_freq = 1.0 / 10  # Frequência esperada se uniforme
            chi2_stat = np.sum((frequencies - expected_freq) ** 2 / expected_freq)
            p_value = 1 - binom.cdf(chi2_stat, 9, 1)  # Aproximação
            
            # Análise de sequências
            sequences = self._analyze_sequences(col_data)
            
            analysis[f'coluna_{col+1}'] = {
                'frequencies': dict(zip(unique, frequencies)),
                'most_frequent': int(most_frequent),
                'least_frequent': int(least_frequent),
                'entropy': float(entropy),
                'is_random': entropy > 2.5 and p_value > 0.05,
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'sequences': sequences
            }
        
        return analysis
    
    def _analyze_sequences(self, data: np.ndarray) -> Dict[str, Any]:
        """Analisa sequências e padrões nos dados"""
        # Sequências consecutivas
        consecutive = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Padrões de alternância
        alternations = 0
        for i in range(2, len(data)):
            if data[i] != data[i-1] and data[i-1] != data[i-2]:
                alternations += 1
        
        return {
            'max_consecutive': max_consecutive,
            'alternation_rate': alternations / max(1, len(data) - 2),
            'total_alternations': alternations
        }
    
    def generate_optimal_strategy(self, historico: np.ndarray, 
                                budget: float = 100.0) -> Dict[str, Any]:
        """
        Gera estratégia ótima baseada no histórico e orçamento
        
        Args:
            historico: Dados históricos
            budget: Orçamento disponível
            
        Returns:
            Dict com estratégia recomendada
        """
        # Analisar padrões históricos
        frequency_analysis = self.analyze_frequency_patterns(historico)
        
        # Identificar colunas com maior aleatoriedade
        random_columns = []
        for col in range(7):
            col_key = f'coluna_{col+1}'
            if frequency_analysis[col_key]['is_random']:
                random_columns.append(col)
        
        # Calcular quantas combinações cabem no orçamento
        max_combinations = int(budget / self.custo_aposta)
        
        # Estratégia baseada no orçamento e análise
        if max_combinations < 10:
            # Orçamento baixo: apostas simples
            strategy = {
                'tipo': 'simples',
                'numeros_por_coluna': [1] * 7,
                'combinacoes': 1,
                'custo': 2.50,
                'descricao': 'Aposta mínima - 1 número por coluna'
            }
        elif max_combinations < 50:
            # Orçamento médio: focar em colunas menos aleatórias
            numeros_por_coluna = [1] * 7
            for col in random_columns[:2]:  # 2 colunas menos aleatórias
                numeros_por_coluna[col] = 2
            
            strategy = {
                'tipo': 'seletivo',
                'numeros_por_coluna': numeros_por_coluna,
                'combinacoes': 4,
                'custo': 10.00,
                'descricao': '2 números nas colunas menos aleatórias'
            }
        else:
            # Orçamento alto: covering mais amplo
            numeros_por_coluna = [1] * 7
            for col in range(min(3, len(random_columns))):
                numeros_por_coluna[random_columns[col]] = 2
            
            strategy = {
                'tipo': 'covering_otimizado',
                'numeros_por_coluna': numeros_por_coluna,
                'combinacoes': 8,
                'custo': 20.00,
                'descricao': '2 números nas 3 colunas menos aleatórias'
            }
        
        # Calcular probabilidades para a estratégia
        probabilities = self.calculate_exact_probabilities(strategy['numeros_por_coluna'])
        strategy['probabilities'] = probabilities
        
        return strategy
    
    def calculate_covering_design_efficiency(self, numeros_por_coluna: List[int]) -> Dict[str, float]:
        """
        Calcula eficiência da estratégia de Covering Design
        
        Args:
            numeros_por_coluna: Estratégia de aposta
            
        Returns:
            Dict com métricas de eficiência
        """
        probabilities = self.calculate_exact_probabilities(numeros_por_coluna)
        
        # Calcular cobertura para cada faixa
        cobertura = {}
        for acertos in range(3, 8):
            prob = getattr(probabilities, f'faixa_{acertos}')
            cobertura[f'faixa_{acertos}'] = prob
        
        # Eficiência geral (média ponderada das probabilidades)
        pesos = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5}  # Peso maior para faixas mais altas
        eficiencia = sum(cobertura[f'faixa_{acertos}'] * pesos[acertos] 
                        for acertos in range(3, 8)) / sum(pesos.values())
        
        # Custo-benefício
        custo_total = probabilities.total_combinations * self.custo_aposta
        custo_beneficio = eficiencia / custo_total if custo_total > 0 else 0
        
        return {
            'eficiencia_geral': eficiencia,
            'custo_beneficio': custo_beneficio,
            'cobertura_por_faixa': cobertura,
            'custo_total': custo_total
        }
    
    def generate_anti_pattern_combinations(self, historico: np.ndarray, 
                                         n_combinations: int = 10) -> List[List[int]]:
        """
        Gera combinações anti-padrão baseadas no histórico
        
        Args:
            historico: Dados históricos
            n_combinations: Número de combinações a gerar
            
        Returns:
            Lista de combinações anti-padrão
        """
        frequency_analysis = self.analyze_frequency_patterns(historico)
        
        combinations = []
        
        for _ in range(n_combinations):
            combination = []
            
            for col in range(7):
                col_key = f'coluna_{col+1}'
                frequencies = frequency_analysis[col_key]['frequencies']
                
                # Escolher números menos frequentes
                sorted_nums = sorted(frequencies.items(), key=lambda x: x[1])
                least_frequent = [num for num, freq in sorted_nums[:2]]
                
                # Escolher aleatoriamente entre os menos frequentes
                chosen = np.random.choice(least_frequent)
                combination.append(chosen)
            
            combinations.append(combination)
        
        return combinations

def main():
    """Exemplo de uso da calculadora de probabilidades"""
    print("🎯 CALCULADORA DE PROBABILIDADES DO SUPER SETE")
    print("=" * 60)
    
    calculator = SuperSeteProbabilityCalculator()
    
    # Exemplo 1: Aposta simples
    print("\n📊 APOSTA SIMPLES (1 número por coluna):")
    print("-" * 40)
    
    prob_simples = calculator.calculate_exact_probabilities([1] * 7)
    print(f"Total de combinações: {prob_simples.total_combinations}")
    print(f"Custo total: R$ {prob_simples.total_combinations * 2.50:.2f}")
    print(f"Probabilidade 7 acertos: {prob_simples.faixa_7:.2e}")
    print(f"Probabilidade 6 acertos: {prob_simples.faixa_6:.2e}")
    print(f"Probabilidade 5 acertos: {prob_simples.faixa_5:.2e}")
    print(f"Probabilidade 4 acertos: {prob_simples.faixa_4:.2e}")
    print(f"Probabilidade 3 acertos: {prob_simples.faixa_3:.2e}")
    
    # Exemplo 2: Estratégia de covering
    print("\n📊 ESTRATÉGIA COVERING (2 números em 3 colunas):")
    print("-" * 40)
    
    prob_covering = calculator.calculate_exact_probabilities([2, 2, 2, 1, 1, 1, 1])
    print(f"Total de combinações: {prob_covering.total_combinations}")
    print(f"Custo total: R$ {prob_covering.total_combinations * 2.50:.2f}")
    print(f"Probabilidade 7 acertos: {prob_covering.faixa_7:.2e}")
    print(f"Probabilidade 6 acertos: {prob_covering.faixa_6:.2e}")
    print(f"Probabilidade 5 acertos: {prob_covering.faixa_5:.2e}")
    print(f"Probabilidade 4 acertos: {prob_covering.faixa_4:.2e}")
    print(f"Probabilidade 3 acertos: {prob_covering.faixa_3:.2e}")
    
    # Exemplo 3: Análise de eficiência
    print("\n📊 ANÁLISE DE EFICIÊNCIA:")
    print("-" * 40)
    
    eficiencia = calculator.calculate_covering_design_efficiency([2, 2, 2, 1, 1, 1, 1])
    print(f"Eficiência geral: {eficiencia['eficiencia_geral']:.4f}")
    print(f"Custo-benefício: {eficiencia['custo_beneficio']:.6f}")
    print(f"Custo total: R$ {eficiencia['custo_total']:.2f}")

if __name__ == "__main__":
    main()
