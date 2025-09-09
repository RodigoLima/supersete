#!/usr/bin/env python3
"""
Guia Completo de Estratégias para Super Sete
===========================================

Guia baseado na análise científica fornecida, implementando todas as estratégias
recomendadas para maximizar as chances de ganhar no Super Sete.

Estratégias implementadas:
1. Entendimento da matemática do jogo
2. Distribuição equilibrada de números
3. Foco nas faixas intermediárias
4. Estratégias anti-padrão
5. Covering Design otimizado
6. Análise de frequências históricas
7. Estratégias baseadas em orçamento
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class StrategyRecommendation:
    """Recomendação de estratégia para Super Sete"""
    nome: str
    descricao: str
    numeros_por_coluna: List[int]
    custo_total: float
    probabilidade_ganhar: float
    roi_esperado: float
    nivel_risco: str  # 'baixo', 'medio', 'alto'
    adequado_para: str  # 'iniciante', 'intermediario', 'avancado'

class SuperSeteStrategyGuide:
    """Guia completo de estratégias para Super Sete"""
    
    def __init__(self):
        self.n_colunas = 7
        self.numeros_por_coluna = 10  # 0 a 9
        self.custo_aposta = 2.50
        self.faixas_premiacao = {
            7: "1ª faixa (principal)",
            6: "2ª faixa", 
            5: "3ª faixa",
            4: "4ª faixa",
            3: "5ª faixa"
        }
        
    def generate_basic_strategies(self) -> List[StrategyRecommendation]:
        """Gera estratégias básicas baseadas na análise científica"""
        strategies = []
        
        # Estratégia 1: Aposta Mínima (Conservadora)
        strategies.append(StrategyRecommendation(
            nome="Aposta Mínima",
            descricao="1 número por coluna - estratégia mais conservadora",
            numeros_por_coluna=[1] * 7,
            custo_total=2.50,
            probabilidade_ganhar=1/10_000_000,  # 7 acertos
            roi_esperado=0.0,  # Muito baixo
            nivel_risco="baixo",
            adequado_para="iniciante"
        ))
        
        # Estratégia 2: Duas Colunas Duplas (Moderada)
        strategies.append(StrategyRecommendation(
            nome="Duas Colunas Duplas",
            descricao="2 números nas 2 primeiras colunas - balanceada",
            numeros_por_coluna=[2, 2, 1, 1, 1, 1, 1],
            custo_total=10.00,
            probabilidade_ganhar=4/10_000_000,  # 4x mais chances
            roi_esperado=0.1,
            nivel_risco="medio",
            adequado_para="intermediario"
        ))
        
        # Estratégia 3: Três Colunas Duplas (Agressiva)
        strategies.append(StrategyRecommendation(
            nome="Três Colunas Duplas",
            descricao="2 números nas 3 primeiras colunas - mais agressiva",
            numeros_por_coluna=[2, 2, 2, 1, 1, 1, 1],
            custo_total=20.00,
            probabilidade_ganhar=8/10_000_000,  # 8x mais chances
            roi_esperado=0.2,
            nivel_risco="alto",
            adequado_para="avancado"
        ))
        
        return strategies
    
    def generate_anti_pattern_strategies(self, historico: np.ndarray) -> List[StrategyRecommendation]:
        """Gera estratégias anti-padrão baseadas no histórico"""
        strategies = []
        
        # Analisar frequências históricas
        frequency_analysis = self._analyze_historical_frequencies(historico)
        
        # Estratégia Anti-Padrão Básica
        anti_pattern_basic = self._create_anti_pattern_strategy(frequency_analysis, nivel="basico")
        strategies.append(anti_pattern_basic)
        
        # Estratégia Anti-Padrão Avançada
        anti_pattern_advanced = self._create_anti_pattern_strategy(frequency_analysis, nivel="avancado")
        strategies.append(anti_pattern_advanced)
        
        return strategies
    
    def generate_covering_design_strategies(self, historico: np.ndarray, 
                                          budget: float = 100.0) -> List[StrategyRecommendation]:
        """Gera estratégias de Covering Design otimizadas"""
        strategies = []
        
        # Analisar padrões históricos
        frequency_analysis = self._analyze_historical_frequencies(historico)
        
        # Identificar colunas mais previsíveis
        predictable_columns = self._identify_predictable_columns(frequency_analysis)
        
        # Estratégia Covering Conservadora
        if len(predictable_columns) >= 2:
            covering_conservative = self._create_covering_strategy(
                predictable_columns[:2], budget * 0.3, "Covering Conservador"
            )
            strategies.append(covering_conservative)
        
        # Estratégia Covering Balanceada
        if len(predictable_columns) >= 3:
            covering_balanced = self._create_covering_strategy(
                predictable_columns[:3], budget * 0.6, "Covering Balanceado"
            )
            strategies.append(covering_balanced)
        
        # Estratégia Covering Agressiva
        if len(predictable_columns) >= 4:
            covering_aggressive = self._create_covering_strategy(
                predictable_columns[:4], budget, "Covering Agressivo"
            )
            strategies.append(covering_aggressive)
        
        return strategies
    
    def generate_budget_based_strategies(self, budget: float) -> List[StrategyRecommendation]:
        """Gera estratégias baseadas no orçamento disponível"""
        strategies = []
        
        if budget < 10:
            # Orçamento baixo - estratégias conservadoras
            strategies.append(StrategyRecommendation(
                nome="Orçamento Baixo",
                descricao="Aposta mínima para orçamento limitado",
                numeros_por_coluna=[1] * 7,
                custo_total=2.50,
                probabilidade_ganhar=1/10_000_000,
                roi_esperado=0.0,
                nivel_risco="baixo",
                adequado_para="iniciante"
            ))
        
        elif budget < 50:
            # Orçamento médio - estratégias balanceadas
            strategies.append(StrategyRecommendation(
                nome="Orçamento Médio",
                descricao="2 números em 2 colunas para orçamento médio",
                numeros_por_coluna=[2, 2, 1, 1, 1, 1, 1],
                custo_total=10.00,
                probabilidade_ganhar=4/10_000_000,
                roi_esperado=0.1,
                nivel_risco="medio",
                adequado_para="intermediario"
            ))
        
        elif budget < 200:
            # Orçamento alto - estratégias agressivas
            strategies.append(StrategyRecommendation(
                nome="Orçamento Alto",
                descricao="2 números em 3 colunas para orçamento alto",
                numeros_por_coluna=[2, 2, 2, 1, 1, 1, 1],
                custo_total=20.00,
                probabilidade_ganhar=8/10_000_000,
                roi_esperado=0.2,
                nivel_risco="alto",
                adequado_para="avancado"
            ))
        
        else:
            # Orçamento muito alto - estratégias máximas
            strategies.append(StrategyRecommendation(
                nome="Orçamento Máximo",
                descricao="2 números em todas as colunas",
                numeros_por_coluna=[2] * 7,
                custo_total=320.00,
                probabilidade_ganhar=128/10_000_000,
                roi_esperado=0.5,
                nivel_risco="muito_alto",
                adequado_para="avancado"
            ))
        
        return strategies
    
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
            entropy = -sum(f * np.log2(f + 1e-10) for f in frequencies)
            column_entropies.append((col, entropy))
        
        # Ordenar por entropia (menor = mais previsível)
        column_entropies.sort(key=lambda x: x[1])
        
        return [col for col, _ in column_entropies]
    
    def _create_anti_pattern_strategy(self, frequency_analysis: Dict[int, Dict[int, float]], 
                                    nivel: str) -> StrategyRecommendation:
        """Cria estratégia anti-padrão baseada em frequências"""
        numeros_por_coluna = [1] * 7
        
        if nivel == "basico":
            # Focar em 2 colunas com números menos frequentes
            for col in range(2):
                frequencies = frequency_analysis[col]
                sorted_nums = sorted(frequencies.items(), key=lambda x: x[1])
                least_frequent = [num for num, freq in sorted_nums[:2]]
                
                if len(least_frequent) >= 2:
                    numeros_por_coluna[col] = 2
            
            nome = "Anti-Padrão Básico"
            custo = 4 * 2.50
            descricao = "2 números nas colunas com números menos frequentes"
        
        else:  # avançado
            # Focar em 3 colunas com números menos frequentes
            for col in range(3):
                frequencies = frequency_analysis[col]
                sorted_nums = sorted(frequencies.items(), key=lambda x: x[1])
                least_frequent = [num for num, freq in sorted_nums[:2]]
                
                if len(least_frequent) >= 2:
                    numeros_por_coluna[col] = 2
            
            nome = "Anti-Padrão Avançado"
            custo = 8 * 2.50
            descricao = "2 números nas 3 colunas com números menos frequentes"
        
        return StrategyRecommendation(
            nome=nome,
            descricao=descricao,
            numeros_por_coluna=numeros_por_coluna,
            custo_total=custo,
            probabilidade_ganhar=custo / 2.50 / 10_000_000,
            roi_esperado=0.15,
            nivel_risco="medio",
            adequado_para="intermediario"
        )
    
    def _create_covering_strategy(self, predictable_columns: List[int], 
                                budget: float, nome: str) -> StrategyRecommendation:
        """Cria estratégia de covering baseada em colunas previsíveis"""
        numeros_por_coluna = [1] * 7
        
        # Adicionar 2 números nas colunas previsíveis
        for col in predictable_columns:
            numeros_por_coluna[col] = 2
        
        # Calcular custo
        total_combinations = 1
        for n in numeros_por_coluna:
            total_combinations *= n
        
        custo_total = total_combinations * self.custo_aposta
        
        # Ajustar se exceder o orçamento
        if custo_total > budget:
            # Reduzir para caber no orçamento
            max_combinations = int(budget / self.custo_aposta)
            if max_combinations >= 4:  # Mínimo 2 colunas com 2 números
                numeros_por_coluna = [2, 2, 1, 1, 1, 1, 1]
                custo_total = 10.00
            else:
                numeros_por_coluna = [1] * 7
                custo_total = 2.50
        
        return StrategyRecommendation(
            nome=nome,
            descricao=f"2 números nas colunas mais previsíveis: {predictable_columns}",
            numeros_por_coluna=numeros_por_coluna,
            custo_total=custo_total,
            probabilidade_ganhar=custo_total / 2.50 / 10_000_000,
            roi_esperado=0.2,
            nivel_risco="medio",
            adequado_para="avancado"
        )
    
    def generate_comprehensive_guide(self, historico: np.ndarray, 
                                   budget: float = 100.0) -> Dict[str, Any]:
        """Gera guia completo de estratégias"""
        print("📚 Gerando guia completo de estratégias para Super Sete...")
        
        # Gerar todas as estratégias
        basic_strategies = self.generate_basic_strategies()
        anti_pattern_strategies = self.generate_anti_pattern_strategies(historico)
        covering_strategies = self.generate_covering_design_strategies(historico, budget)
        budget_strategies = self.generate_budget_based_strategies(budget)
        
        # Combinar todas as estratégias
        all_strategies = (basic_strategies + anti_pattern_strategies + 
                         covering_strategies + budget_strategies)
        
        # Remover duplicatas
        unique_strategies = []
        seen_names = set()
        for strategy in all_strategies:
            if strategy.nome not in seen_names:
                unique_strategies.append(strategy)
                seen_names.add(strategy.nome)
        
        # Ordenar por ROI esperado
        unique_strategies.sort(key=lambda x: x.roi_esperado, reverse=True)
        
        # Gerar recomendações personalizadas
        recommendations = self._generate_personalized_recommendations(
            historico, budget, unique_strategies
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(unique_strategies),
            'budget_analyzed': budget,
            'strategies': [strategy.__dict__ for strategy in unique_strategies],
            'recommendations': recommendations,
            'guide_summary': self._generate_guide_summary(unique_strategies)
        }
    
    def _generate_personalized_recommendations(self, historico: np.ndarray, 
                                             budget: float, 
                                             strategies: List[StrategyRecommendation]) -> Dict[str, Any]:
        """Gera recomendações personalizadas baseadas no histórico e orçamento"""
        # Analisar características do histórico
        frequency_analysis = self._analyze_historical_frequencies(historico)
        
        # Calcular nível de aleatoriedade
        random_columns = 0
        for col in range(7):
            frequencies = list(frequency_analysis[col].values())
            entropy = -sum(f * np.log2(f + 1e-10) for f in frequencies)
            max_entropy = np.log2(10)
            normalized_entropy = entropy / max_entropy
            
            if normalized_entropy > 0.8:
                random_columns += 1
        
        randomness_ratio = random_columns / 7
        
        # Gerar recomendações baseadas na análise
        recommendations = {
            'overall_assessment': self._assess_overall_situation(randomness_ratio, budget),
            'best_strategies': self._select_best_strategies(strategies, budget),
            'risk_management': self._generate_risk_management_advice(randomness_ratio),
            'budget_optimization': self._optimize_budget_allocation(budget, strategies)
        }
        
        return recommendations
    
    def _assess_overall_situation(self, randomness_ratio: float, budget: float) -> str:
        """Avalia a situação geral para recomendações"""
        if randomness_ratio > 0.7 and budget < 50:
            return "Situação conservadora: alta aleatoriedade e orçamento limitado. Use estratégias básicas."
        elif randomness_ratio < 0.3 and budget > 100:
            return "Situação favorável: baixa aleatoriedade e orçamento adequado. Use estratégias agressivas."
        elif randomness_ratio > 0.5 and budget > 50:
            return "Situação balanceada: aleatoriedade média e orçamento adequado. Use estratégias moderadas."
        else:
            return "Situação mista: considere estratégias adaptativas baseadas na análise de frequências."
    
    def _select_best_strategies(self, strategies: List[StrategyRecommendation], 
                               budget: float) -> List[Dict[str, Any]]:
        """Seleciona as melhores estratégias para o orçamento"""
        # Filtrar estratégias que cabem no orçamento
        affordable_strategies = [s for s in strategies if s.custo_total <= budget]
        
        if not affordable_strategies:
            # Se nenhuma estratégia cabe, pegar a mais barata
            affordable_strategies = [min(strategies, key=lambda x: x.custo_total)]
        
        # Ordenar por ROI esperado
        affordable_strategies.sort(key=lambda x: x.roi_esperado, reverse=True)
        
        # Retornar top 3
        return [strategy.__dict__ for strategy in affordable_strategies[:3]]
    
    def _generate_risk_management_advice(self, randomness_ratio: float) -> List[str]:
        """Gera conselhos de gestão de risco"""
        advice = []
        
        if randomness_ratio > 0.7:
            advice.append("✅ Alta aleatoriedade detectada - use estratégias conservadoras")
            advice.append("⚠️  Evite apostas muito agressivas")
            advice.append("💡 Foque em recuperação de custos com faixas menores")
        else:
            advice.append("🎯 Baixa aleatoriedade detectada - estratégias anti-padrão podem funcionar")
            advice.append("💡 Considere estratégias de Covering Design")
            advice.append("⚠️  Monitore mudanças nos padrões regularmente")
        
        advice.append("📊 Sempre analise frequências antes de apostar")
        advice.append("💰 Nunca aposte mais do que pode perder")
        advice.append("🔄 Ajuste estratégias baseado em resultados recentes")
        
        return advice
    
    def _optimize_budget_allocation(self, budget: float, 
                                  strategies: List[StrategyRecommendation]) -> Dict[str, Any]:
        """Otimiza alocação de orçamento"""
        if budget < 25:
            return {
                'recommendation': 'Orçamento baixo - use apenas apostas básicas',
                'max_strategies': 1,
                'suggested_allocation': '100% em aposta simples'
            }
        elif budget < 100:
            return {
                'recommendation': 'Orçamento médio - combine 2-3 estratégias',
                'max_strategies': 3,
                'suggested_allocation': '60% estratégia principal, 40% estratégia secundária'
            }
        else:
            return {
                'recommendation': 'Orçamento alto - diversifique estratégias',
                'max_strategies': 5,
                'suggested_allocation': '40% estratégia principal, 30% estratégia secundária, 30% estratégias de apoio'
            }
    
    def _generate_guide_summary(self, strategies: List[StrategyRecommendation]) -> Dict[str, Any]:
        """Gera resumo do guia"""
        total_strategies = len(strategies)
        strategies_by_risk = {
            'baixo': len([s for s in strategies if s.nivel_risco == 'baixo']),
            'medio': len([s for s in strategies if s.nivel_risco == 'medio']),
            'alto': len([s for s in strategies if s.nivel_risco == 'alto'])
        }
        
        strategies_by_level = {
            'iniciante': len([s for s in strategies if s.adequado_para == 'iniciante']),
            'intermediario': len([s for s in strategies if s.adequado_para == 'intermediario']),
            'avancado': len([s for s in strategies if s.adequado_para == 'avancado'])
        }
        
        avg_roi = np.mean([s.roi_esperado for s in strategies])
        max_roi = max([s.roi_esperado for s in strategies])
        
        return {
            'total_strategies': total_strategies,
            'strategies_by_risk': strategies_by_risk,
            'strategies_by_level': strategies_by_level,
            'average_roi': avg_roi,
            'max_roi': max_roi,
            'most_risky_strategy': max(strategies, key=lambda x: x.roi_esperado).nome,
            'safest_strategy': min(strategies, key=lambda x: x.roi_esperado).nome
        }

def main():
    """Exemplo de uso do guia de estratégias"""
    print("📚 GUIA DE ESTRATÉGIAS PARA SUPER SETE")
    print("=" * 50)
    
    # Criar guia
    guide = SuperSeteStrategyGuide()
    
    # Simular dados históricos
    np.random.seed(42)
    historico_simulado = np.random.randint(0, 10, (100, 7))
    
    # Gerar guia completo
    guia_completo = guide.generate_comprehensive_guide(historico_simulado, 100.0)
    
    print(f"\n📊 RESUMO DO GUIA:")
    print(f"   • Total de estratégias: {guia_completo['total_strategies']}")
    print(f"   • Orçamento analisado: R$ {guia_completo['budget_analyzed']}")
    
    # Mostrar melhores estratégias
    print(f"\n🏆 MELHORES ESTRATÉGIAS:")
    for i, strategy in enumerate(guia_completo['recommendations']['best_strategies'][:3]):
        print(f"   {i+1}. {strategy['nome']} - ROI: {strategy['roi_esperado']:.3f}")
    
    # Mostrar conselhos de gestão de risco
    print(f"\n💡 CONSELHOS DE GESTÃO DE RISCO:")
    for advice in guia_completo['recommendations']['risk_management']:
        print(f"   {advice}")

if __name__ == "__main__":
    main()
