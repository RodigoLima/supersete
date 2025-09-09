#!/usr/bin/env python3
"""
Estratégias de Covering Design para Super Sete
=============================================

Implementa algoritmos de Covering Design para otimizar apostas no Super Sete,
baseado na teoria matemática de conjuntos cobertos.

Referência: Covering Design Theory aplicada a loterias
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations, product
import math
from collections import defaultdict

@dataclass
class CoveringDesignConfig:
    """Configuração para Covering Design"""
    n_colunas: int = 7
    numeros_por_coluna: int = 10  # 0 a 9
    max_numeros_por_coluna: int = 2
    budget_max: float = 1000.0
    custo_aposta: float = 2.50
    min_coverage: float = 0.8  # Cobertura mínima desejada

@dataclass
class CoveringStrategy:
    """Estratégia de covering gerada"""
    nome: str
    numeros_por_coluna: List[int]
    total_combinacoes: int
    custo_total: float
    cobertura_estimada: float
    descricao: str
    probabilidades: Dict[int, float]

class SuperSeteCoveringDesign:
    """Implementa estratégias de Covering Design para Super Sete"""
    
    def __init__(self, config: CoveringDesignConfig = None):
        self.config = config or CoveringDesignConfig()
        self.numeros_possiveis = list(range(10))  # 0 a 9
        
    def generate_basic_strategies(self) -> List[CoveringStrategy]:
        """Gera estratégias básicas de covering"""
        strategies = []
        
        # Estratégia 1: Aposta mínima
        strategies.append(self._create_strategy(
            nome="Aposta Mínima",
            numeros_por_coluna=[1] * 7,
            descricao="1 número por coluna - aposta básica"
        ))
        
        # Estratégia 2: Duas colunas com 2 números
        strategies.append(self._create_strategy(
            nome="Duas Colunas Duplas",
            numeros_por_coluna=[2, 2, 1, 1, 1, 1, 1],
            descricao="2 números nas 2 primeiras colunas"
        ))
        
        # Estratégia 3: Três colunas com 2 números
        strategies.append(self._create_strategy(
            nome="Três Colunas Duplas",
            numeros_por_coluna=[2, 2, 2, 1, 1, 1, 1],
            descricao="2 números nas 3 primeiras colunas"
        ))
        
        # Estratégia 4: Quatro colunas com 2 números
        strategies.append(self._create_strategy(
            nome="Quatro Colunas Duplas",
            numeros_por_coluna=[2, 2, 2, 2, 1, 1, 1],
            descricao="2 números nas 4 primeiras colunas"
        ))
        
        # Estratégia 5: Todas as colunas com 2 números
        strategies.append(self._create_strategy(
            nome="Todas Colunas Duplas",
            numeros_por_coluna=[2] * 7,
            descricao="2 números em todas as colunas"
        ))
        
        return strategies
    
    def generate_optimized_strategies(self, historico: np.ndarray, 
                                    budget: float = 100.0) -> List[CoveringStrategy]:
        """
        Gera estratégias otimizadas baseadas no histórico
        
        Args:
            historico: Dados históricos de sorteios
            budget: Orçamento disponível
            
        Returns:
            Lista de estratégias otimizadas
        """
        # Analisar frequências históricas
        frequency_analysis = self._analyze_historical_frequencies(historico)
        
        # Identificar colunas com menor aleatoriedade (mais previsíveis)
        predictable_columns = self._identify_predictable_columns(frequency_analysis)
        
        strategies = []
        
        # Estratégia 1: Foco em colunas previsíveis
        if len(predictable_columns) >= 2:
            numeros_por_coluna = [1] * 7
            for col in predictable_columns[:2]:
                numeros_por_coluna[col] = 2
            
            strategies.append(self._create_strategy(
                nome="Foco Previsível",
                numeros_por_coluna=numeros_por_coluna,
                descricao="2 números nas colunas mais previsíveis"
            ))
        
        # Estratégia 2: Cobertura balanceada
        numeros_por_coluna = [1] * 7
        for i, col in enumerate(predictable_columns[:3]):
            if i < 3:  # Máximo 3 colunas com 2 números
                numeros_por_coluna[col] = 2
        
        strategies.append(self._create_strategy(
            nome="Cobertura Balanceada",
            numeros_por_coluna=numeros_por_coluna,
            descricao="2 números nas 3 colunas mais previsíveis"
        ))
        
        # Estratégia 3: Anti-padrão (números menos frequentes)
        anti_pattern = self._generate_anti_pattern_strategy(frequency_analysis)
        strategies.append(self._create_strategy(
            nome="Anti-Padrão",
            numeros_por_coluna=anti_pattern,
            descricao="2 números nas colunas com números menos frequentes"
        ))
        
        # Filtrar por orçamento
        max_combinations = int(budget / self.config.custo_aposta)
        strategies = [s for s in strategies if s.total_combinacoes <= max_combinations]
        
        return strategies
    
    def _create_strategy(self, nome: str, numeros_por_coluna: List[int], 
                        descricao: str) -> CoveringStrategy:
        """Cria uma estratégia de covering"""
        total_combinacoes = 1
        for n in numeros_por_coluna:
            total_combinacoes *= n
        
        custo_total = total_combinacoes * self.config.custo_aposta
        
        # Calcular cobertura estimada
        cobertura_estimada = self._calculate_coverage_estimate(numeros_por_coluna)
        
        # Calcular probabilidades
        probabilidades = self._calculate_probabilities(numeros_por_coluna)
        
        return CoveringStrategy(
            nome=nome,
            numeros_por_coluna=numeros_por_coluna,
            total_combinacoes=total_combinacoes,
            custo_total=custo_total,
            cobertura_estimada=cobertura_estimada,
            descricao=descricao,
            probabilidades=probabilidades
        )
    
    def _analyze_historical_frequencies(self, historico: np.ndarray) -> Dict[int, Dict[int, float]]:
        """Analisa frequências históricas por coluna"""
        frequency_analysis = {}
        
        for col in range(7):
            col_data = historico[:, col]
            unique, counts = np.unique(col_data, return_counts=True)
            frequencies = counts / len(col_data)
            
            frequency_analysis[col] = dict(zip(unique, frequencies))
        
        return frequency_analysis
    
    def _identify_predictable_columns(self, frequency_analysis: Dict[int, Dict[int, float]]) -> List[int]:
        """Identifica colunas mais previsíveis baseado na entropia"""
        column_entropies = []
        
        for col in range(7):
            frequencies = list(frequency_analysis[col].values())
            entropy = -sum(f * math.log2(f + 1e-10) for f in frequencies)
            column_entropies.append((col, entropy))
        
        # Ordenar por entropia (menor = mais previsível)
        column_entropies.sort(key=lambda x: x[1])
        
        return [col for col, _ in column_entropies]
    
    def _generate_anti_pattern_strategy(self, frequency_analysis: Dict[int, Dict[int, float]]) -> List[int]:
        """Gera estratégia anti-padrão baseada em números menos frequentes"""
        numeros_por_coluna = [1] * 7
        
        for col in range(7):
            frequencies = frequency_analysis[col]
            # Encontrar números menos frequentes
            sorted_nums = sorted(frequencies.items(), key=lambda x: x[1])
            
            # Se há números claramente menos frequentes, usar 2 números
            if len(sorted_nums) >= 2:
                least_frequent = [num for num, freq in sorted_nums[:2]]
                # Verificar se a diferença é significativa
                if sorted_nums[1][1] < sorted_nums[-1][1] * 0.7:  # 30% menos frequente
                    numeros_por_coluna[col] = 2
        
        return numeros_por_coluna
    
    def _calculate_coverage_estimate(self, numeros_por_coluna: List[int]) -> float:
        """Calcula estimativa de cobertura da estratégia"""
        # Cobertura baseada na probabilidade de acertar pelo menos 3 números
        prob_3_ou_mais = 0.0
        
        for acertos in range(3, 8):
            prob = self._calculate_exact_probability(numeros_por_coluna, acertos)
            prob_3_ou_mais += prob
        
        return prob_3_ou_mais
    
    def _calculate_exact_probability(self, numeros_por_coluna: List[int], acertos: int) -> float:
        """Calcula probabilidade exata de acertar 'acertos' números"""
        if acertos < 3 or acertos > 7:
            return 0.0
        
        total_prob = 0.0
        
        # Gerar todas as combinações de colunas que podem ser acertadas
        for colunas_acertadas in combinations(range(7), acertos):
            prob_esta_combinacao = 1.0
            
            for col in range(7):
                if col in colunas_acertadas:
                    # Esta coluna foi acertada
                    prob_esta_combinacao *= (numeros_por_coluna[col] / 10)
                else:
                    # Esta coluna não foi acertada
                    prob_esta_combinacao *= ((10 - numeros_por_coluna[col]) / 10)
            
            total_prob += prob_esta_combinacao
        
        return total_prob
    
    def _calculate_probabilities(self, numeros_por_coluna: List[int]) -> Dict[int, float]:
        """Calcula probabilidades para todas as faixas"""
        probabilidades = {}
        
        for acertos in range(3, 8):
            prob = self._calculate_exact_probability(numeros_por_coluna, acertos)
            probabilidades[acertos] = prob
        
        return probabilidades
    
    def optimize_for_budget(self, strategies: List[CoveringStrategy], 
                          budget: float) -> List[CoveringStrategy]:
        """Otimiza estratégias para um orçamento específico"""
        # Filtrar estratégias que cabem no orçamento
        affordable_strategies = [s for s in strategies if s.custo_total <= budget]
        
        if not affordable_strategies:
            return []
        
        # Ordenar por eficiência (cobertura / custo)
        affordable_strategies.sort(key=lambda s: s.cobertura_estimada / s.custo_total, reverse=True)
        
        return affordable_strategies
    
    def generate_covering_matrix(self, strategy: CoveringStrategy) -> np.ndarray:
        """
        Gera matriz de cobertura para uma estratégia
        
        Args:
            strategy: Estratégia de covering
            
        Returns:
            Matriz onde cada linha é uma combinação possível
        """
        # Gerar todas as combinações possíveis
        combinations = []
        
        for col in range(7):
            col_combinations = []
            for num in range(10):
                if num < strategy.numeros_por_coluna[col]:
                    col_combinations.append(num)
            
            combinations.append(col_combinations)
        
        # Gerar produto cartesiano
        all_combinations = list(product(*combinations))
        
        return np.array(all_combinations)
    
    def calculate_strategy_efficiency(self, strategy: CoveringStrategy) -> Dict[str, float]:
        """Calcula métricas de eficiência para uma estratégia"""
        # Eficiência baseada na cobertura por real investido
        cobertura_por_real = strategy.cobertura_estimada / strategy.custo_total
        
        # Eficiência baseada na probabilidade de ganhar algo
        prob_ganhar_algo = sum(strategy.probabilidades.values())
        
        # Eficiência baseada na probabilidade de faixas altas
        prob_faixas_altas = (strategy.probabilidades[6] + strategy.probabilidades[7])
        
        return {
            'cobertura_por_real': cobertura_por_real,
            'probabilidade_ganhar_algo': prob_ganhar_algo,
            'probabilidade_faixas_altas': prob_faixas_altas,
            'eficiencia_geral': (cobertura_por_real + prob_ganhar_algo + prob_faixas_altas) / 3
        }
    
    def recommend_strategy(self, historico: np.ndarray, budget: float = 100.0) -> CoveringStrategy:
        """
        Recomenda a melhor estratégia baseada no histórico e orçamento
        
        Args:
            historico: Dados históricos
            budget: Orçamento disponível
            
        Returns:
            Estratégia recomendada
        """
        # Gerar estratégias otimizadas
        strategies = self.generate_optimized_strategies(historico, budget)
        
        if not strategies:
            # Fallback para estratégias básicas
            strategies = self.generate_basic_strategies()
            strategies = self.optimize_for_budget(strategies, budget)
        
        if not strategies:
            # Último recurso: aposta mínima
            return self._create_strategy(
                nome="Aposta Mínima",
                numeros_por_coluna=[1] * 7,
                descricao="1 número por coluna - aposta básica"
            )
        
        # Calcular eficiência para cada estratégia
        best_strategy = None
        best_efficiency = -1
        
        for strategy in strategies:
            efficiency = self.calculate_strategy_efficiency(strategy)
            if efficiency['eficiencia_geral'] > best_efficiency:
                best_efficiency = efficiency['eficiencia_geral']
                best_strategy = strategy
        
        return best_strategy

def main():
    """Exemplo de uso do Covering Design"""
    print("🎯 COVERING DESIGN PARA SUPER SETE")
    print("=" * 50)
    
    # Configuração
    config = CoveringDesignConfig(budget_max=200.0)
    covering = SuperSeteCoveringDesign(config)
    
    # Gerar estratégias básicas
    print("\n📊 ESTRATÉGIAS BÁSICAS:")
    print("-" * 30)
    
    basic_strategies = covering.generate_basic_strategies()
    for strategy in basic_strategies:
        print(f"\n{strategy.nome}:")
        print(f"  Descrição: {strategy.descricao}")
        print(f"  Combinações: {strategy.total_combinacoes}")
        print(f"  Custo: R$ {strategy.custo_total:.2f}")
        print(f"  Cobertura: {strategy.cobertura_estimada:.4f}")
        
        # Mostrar probabilidades
        print("  Probabilidades:")
        for acertos in range(3, 8):
            prob = strategy.probabilidades[acertos]
            print(f"    {acertos} acertos: {prob:.2e}")
    
    # Exemplo com dados simulados
    print("\n📊 ESTRATÉGIAS OTIMIZADAS (dados simulados):")
    print("-" * 40)
    
    # Simular dados históricos
    np.random.seed(42)
    historico_simulado = np.random.randint(0, 10, (100, 7))
    
    optimized_strategies = covering.generate_optimized_strategies(historico_simulado, 50.0)
    
    for strategy in optimized_strategies:
        print(f"\n{strategy.nome}:")
        print(f"  Descrição: {strategy.descricao}")
        print(f"  Combinações: {strategy.total_combinacoes}")
        print(f"  Custo: R$ {strategy.custo_total:.2f}")
        print(f"  Cobertura: {strategy.cobertura_estimada:.4f}")
        
        # Calcular eficiência
        efficiency = covering.calculate_strategy_efficiency(strategy)
        print(f"  Eficiência: {efficiency['eficiencia_geral']:.4f}")
    
    # Recomendar melhor estratégia
    print("\n🎯 RECOMENDAÇÃO:")
    print("-" * 20)
    
    recommended = covering.recommend_strategy(historico_simulado, 50.0)
    print(f"Estratégia recomendada: {recommended.nome}")
    print(f"Descrição: {recommended.descricao}")
    print(f"Custo: R$ {recommended.custo_total:.2f}")
    print(f"Cobertura: {recommended.cobertura_estimada:.4f}")

if __name__ == "__main__":
    main()
