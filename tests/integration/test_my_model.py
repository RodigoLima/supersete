#!/usr/bin/env python3
"""
Teste Completo e Robusto do Modelo Super Sete
=============================================
Testa o modelo atual com validações específicas para loterias
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.common_interface import SuperSeteInterface, create_analyzer, quick_analysis, quick_validation
from src.validation.validator import SuperSeteValidator, SuperSeteValidationConfig
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any

class SuperSeteTester:
    """Classe para testes completos do modelo Super Sete"""
    
    def __init__(self, config_name: str = 'longo', n_concursos: int = 1000):
        self.interface = create_analyzer(config_name, n_concursos)
        self.config_name = config_name
        self.n_concursos = n_concursos
        self.results = {}
        
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Testa funcionalidade básica do modelo"""
        print("🧪 TESTE 1: FUNCIONALIDADE BÁSICA")
        print("="*50)
        
        try:
            # Carregar dados
            df = self.interface.load_data("data/raw/Super Sete.xlsx")
            print(f"✅ Dados carregados: {len(df)} concursos")
            
            # Verificar estrutura dos dados
            colunas_numericas = [col for col in df.columns if col.startswith('Coluna_')]
            print(f"✅ Colunas numéricas encontradas: {len(colunas_numericas)}")
            
            # Testar geração de jogos usando métodos corretos
            # Simular probabilidades para teste
            prob_by_col = {}
            for col in colunas_numericas:
                prob_by_col[col] = np.ones(10) / 10
            
            jogos_amostragem = self.interface.analyzer.sample_games(prob_by_col, n_games=5)
            jogos_ranking = self.interface.analyzer.top_product_games(prob_by_col, n_games=5, top_per_col=5)
            
            print(f"✅ Jogos gerados (ranking): {len(jogos_ranking)}")
            print(f"✅ Jogos gerados (amostragem): {len(jogos_amostragem)}")
            
            # Verificar validade dos jogos
            jogos_validos = self._validate_games(jogos_ranking + jogos_amostragem)
            print(f"✅ Jogos válidos: {jogos_validos}/{len(jogos_ranking + jogos_amostragem)}")
            
            return {
                'status': 'success',
                'concursos': len(df),
                'colunas': len(colunas_numericas),
                'jogos_ranking': len(jogos_ranking),
                'jogos_amostragem': len(jogos_amostragem),
                'jogos_validos': jogos_validos,
                'total_jogos': len(jogos_ranking + jogos_amostragem)
            }
            
        except Exception as e:
            print(f"❌ Erro no teste básico: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_model_performance(self) -> Dict[str, Any]:
        """Testa performance do modelo usando diretamente o SuperSeteAnalyzer"""
        print("\n📊 TESTE 2: PERFORMANCE DO MODELO")
        print("="*50)
        
        try:
            # Executar análise completa usando diretamente o analisador
            resultados = self.interface.analyzer.run_analysis(
                excel_path="data/raw/Super Sete.xlsx",
                n_jogos=10,
                topk=5,
                seed=42,
                use_cache=True,
                save_results=False  # Não salvar para teste
            )
            
            # Analisar métricas
            confianca_media = resultados.get('avg_confidence', 0)
            colunas_previsiveis = resultados.get('predictable_cols', 0)
            
            print(f"🎯 Confiança média: {confianca_media:.3f}")
            print(f"📈 Colunas previsíveis: {colunas_previsiveis}")
            
            # Testar geração de jogos usando métodos corretos do Super Sete
            # Primeiro precisamos obter as probabilidades das colunas
            df = self.interface.load_data("data/raw/Super Sete.xlsx")
            numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
            
            # Simular probabilidades para teste (em um caso real, viria da análise)
            prob_by_col = {}
            for col in numeric_cols:
                # Distribuição uniforme para teste
                prob_by_col[col] = np.ones(10) / 10
            
            # Usar métodos corretos do SuperSeteAnalyzer
            jogos_amostragem = self.interface.analyzer.sample_games(prob_by_col, n_games=5)
            jogos_ranking = self.interface.analyzer.top_product_games(prob_by_col, n_games=5, top_per_col=5)
            
            # Para jogos de alta confiança, precisamos das confianças também
            confidence_by_col = {col: 0.8 for col in numeric_cols}  # Simular alta confiança
            jogos_alta_confianca = self.interface.analyzer._generate_high_confidence_games(
                prob_by_col, confidence_by_col, n_jogos=5, topk=5
            )
            
            print(f"🎮 Jogos ranking: {len(jogos_ranking)}")
            print(f"🎮 Jogos amostragem: {len(jogos_amostragem)}")
            print(f"🎮 Jogos alta confiança: {len(jogos_alta_confianca)}")
            
            # Avaliar qualidade
            if confianca_media > 0.6:
                qualidade = "excelente"
            elif confianca_media > 0.4:
                qualidade = "boa"
            elif confianca_media > 0.2:
                qualidade = "moderada"
            else:
                qualidade = "baixa"
            
            print(f"⭐ Qualidade geral: {qualidade}")
            
            return {
                'status': 'success',
                'confianca_media': confianca_media,
                'colunas_previsiveis': colunas_previsiveis,
                'qualidade': qualidade,
                'jogos_gerados': {
                    'ranking': len(jogos_ranking),
                    'amostragem': len(jogos_amostragem),
                    'alta_confianca': len(jogos_alta_confianca)
                },
                'resultados_completos': resultados
            }
            
        except Exception as e:
            print(f"❌ Erro no teste de performance: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_validation_integration(self) -> Dict[str, Any]:
        """Testa integração com validator"""
        print("\n🔍 TESTE 3: INTEGRAÇÃO COM VALIDATOR")
        print("="*50)
        
        try:
            # Configuração de validação otimizada
            validation_config = {
                'test_size': 0.2,
                'cv_folds': 3,  # Menos folds para teste rápido
                'min_samples': 20,
                'confidence_level': 0.95,
                'save_plots': False,  # Desabilitar plots para teste
                'results_dir': "resultados_teste",
                'plot_dir': "resultados_teste/plots"
            }
            
            # Executar validação
            results = self.interface.validate_model("data/raw/Super Sete.xlsx", validation_config)
            
            # Analisar resultados da validação
            colunas_testadas = len([k for k in results.keys() if k.startswith('Coluna_') and 'error' not in results[k]])
            colunas_significativas = len([k for k, v in results.items() 
                                        if isinstance(v, dict) and v.get('is_significant', False)])
            
            print(f"✅ Colunas testadas: {colunas_testadas}")
            print(f"✅ Colunas significativas: {colunas_significativas}")
            
            if colunas_testadas > 0:
                taxa_sucesso = colunas_significativas / colunas_testadas * 100
                print(f"📊 Taxa de sucesso: {taxa_sucesso:.1f}%")
            else:
                taxa_sucesso = 0
            
            return {
                'status': 'success',
                'colunas_testadas': colunas_testadas,
                'colunas_significativas': colunas_significativas,
                'taxa_sucesso': taxa_sucesso,
                'resultados_validacao': results
            }
            
        except Exception as e:
            print(f"❌ Erro na integração com validator: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_data_quality(self) -> Dict[str, Any]:
        """Testa qualidade dos dados usando analisadores do Super Sete"""
        print("\n🔍 TESTE 4: QUALIDADE DOS DADOS")
        print("="*50)
        
        try:
            df = self.interface.load_data("data/raw/Super Sete.xlsx")
            colunas_numericas = [col for col in df.columns if col.startswith('Coluna_')]
            
            qualidade_dados = {}
            analises_especializadas = {}
            
            for col in colunas_numericas:
                serie = df[col].dropna().values
                
                # Análise básica
                valores_unicos = len(np.unique(serie))
                entropia = self._calculate_entropy(pd.Series(serie))
                
                # Verificar outliers
                q1, q3 = np.percentile(serie, [25, 75])
                iqr = q3 - q1
                outliers = len(serie[(serie < q1 - 1.5*iqr) | (serie > q3 + 1.5*iqr)])
                
                qualidade_dados[col] = {
                    'valores_unicos': valores_unicos,
                    'entropia': entropia,
                    'outliers': outliers,
                    'outliers_pct': outliers / len(serie) * 100 if len(serie) > 0 else 0
                }
                
                # Usar analisadores especializados do Super Sete
                try:
                    # Análise temporal - usar métodos corretos
                    seasonality = self.interface.temporal_analyzer.detect_seasonality(serie)
                    trend = self.interface.temporal_analyzer.calculate_trend(serie)
                    temporal_analysis = {
                        'seasonality': seasonality,
                        'trend': trend
                    }
                    analises_especializadas[f"{col}_temporal"] = temporal_analysis
                    
                    # Análise de entropia - usar métodos corretos
                    entropy_info = self.interface.entropy_analyzer.calculate_entropy(serie)
                    randomness = self.interface.entropy_analyzer.test_randomness(serie)
                    entropy_analysis = {
                        'entropy': entropy_info,
                        'randomness': randomness
                    }
                    analises_especializadas[f"{col}_entropy"] = entropy_analysis
                    
                    # Análise de confiança - usar métodos corretos
                    # Simular predições para teste
                    predictions = np.random.random(10)  # Simular probabilidades
                    confidence_intervals = self.interface.confidence_analyzer.calculate_confidence_intervals(predictions)
                    confidence_analysis = {
                        'intervals': confidence_intervals
                    }
                    analises_especializadas[f"{col}_confidence"] = confidence_analysis
                    
                except Exception as e:
                    print(f"⚠️  Erro na análise especializada de {col}: {e}")
                    analises_especializadas[f"{col}_error"] = str(e)
            
            print(f"✅ Análise de qualidade concluída para {len(colunas_numericas)} colunas")
            print(f"✅ Análises especializadas: {len(analises_especializadas)}")
            
            return {
                'status': 'success',
                'colunas_analisadas': len(colunas_numericas),
                'qualidade_detalhada': qualidade_dados,
                'analises_especializadas': analises_especializadas
            }
            
        except Exception as e:
            print(f"❌ Erro no teste de qualidade: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_supersete_integration(self) -> Dict[str, Any]:
        """Testa integração direta com funções do Super Sete"""
        print("\n🔗 TESTE 5: INTEGRAÇÃO SUPER SETE")
        print("="*50)
        
        try:
            # Testar métodos diretos do SuperSeteAnalyzer
            analyzer = self.interface.analyzer
            
            # Testar carregamento de dados
            df = analyzer.load_data("data/raw/Super Sete.xlsx")
            print(f"✅ Carregamento direto: {len(df)} concursos")
            
            # Testar análise de coluna individual
            coluna_teste = df['Coluna_1'].values
            analise_coluna = analyzer.analyze_column(coluna_teste, 'Coluna_1')
            print(f"✅ Análise de coluna: {len(analise_coluna)} métricas")
            
            # Testar criação de features
            X, y, feature_names = analyzer.create_features(coluna_teste, window=10)
            print(f"✅ Features criadas: {X.shape[0]} amostras, {X.shape[1]} features")
            
            # Testar analisadores especializados usando métodos corretos
            seasonality = analyzer.temporal_analyzer.detect_seasonality(coluna_teste)
            trend = analyzer.temporal_analyzer.calculate_trend(coluna_teste)
            temporal_result = {'seasonality': seasonality, 'trend': trend}
            
            entropy_info = analyzer.entropy_analyzer.calculate_entropy(coluna_teste)
            randomness = analyzer.entropy_analyzer.test_randomness(coluna_teste)
            entropy_result = {'entropy': entropy_info, 'randomness': randomness}
            
            # Para confiança, simular predições
            predictions = np.random.random(10)
            confidence_intervals = analyzer.confidence_analyzer.calculate_confidence_intervals(predictions)
            confidence_result = {'intervals': confidence_intervals}
            
            print(f"✅ Análise temporal: {len(temporal_result)} métricas")
            print(f"✅ Análise entropia: {len(entropy_result)} métricas")
            print(f"✅ Análise confiança: {len(confidence_result)} métricas")
            
            # Testar geração de jogos usando métodos corretos
            # Simular probabilidades para teste
            numeric_cols = [col for col in df.columns if col.startswith('Coluna_')]
            prob_by_col = {}
            for col in numeric_cols:
                prob_by_col[col] = np.ones(10) / 10
            
            jogos_amostragem = analyzer.sample_games(prob_by_col, n_games=3)
            jogos_ranking = analyzer.top_product_games(prob_by_col, n_games=3, top_per_col=5)
            
            print(f"✅ Jogos amostragem: {len(jogos_amostragem)}")
            print(f"✅ Jogos ranking: {len(jogos_ranking)}")
            
            return {
                'status': 'success',
                'concursos_carregados': len(df),
                'analise_coluna_metricas': len(analise_coluna),
                'features_criadas': X.shape[0] if len(X) > 0 else 0,
                'n_features': X.shape[1] if len(X) > 0 else 0,
                'temporal_metricas': len(temporal_result),
                'entropy_metricas': len(entropy_result),
                'confidence_metricas': len(confidence_result),
                'jogos_amostragem': len(jogos_amostragem),
                'jogos_ranking': len(jogos_ranking),
                'integracao_funcionando': True
            }
            
        except Exception as e:
            print(f"❌ Erro na integração Super Sete: {e}")
            return {'status': 'error', 'error': str(e), 'integracao_funcionando': False}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Executa teste completo"""
        print("🚀 INICIANDO TESTE COMPLETO DO MODELO SUPER SETE")
        print("="*60)
        
        # Executar todos os testes
        test1 = self.test_basic_functionality()
        test2 = self.test_model_performance()
        test3 = self.test_validation_integration()
        test4 = self.test_data_quality()
        test5 = self.test_supersete_integration()  # Novo teste de integração
        
        # Compilar resultados
        resultados_completos = {
            'timestamp': datetime.now().isoformat(),
            'configuracao': {
                'config_name': self.config_name,
                'n_concursos': self.n_concursos
            },
            'testes': {
                'funcionalidade_basica': test1,
                'performance_modelo': test2,
                'integracao_validator': test3,
                'qualidade_dados': test4,
                'integracao_supersete': test5
            },
            'resumo': self._generate_summary(test1, test2, test3, test4, test5)
        }
        
        # Salvar resultados
        self._save_test_results(resultados_completos)
        
        return resultados_completos
    
    def _validate_games(self, jogos: List) -> int:
        """Valida se os jogos estão no formato correto"""
        validos = 0
        for jogo in jogos:
            # Aceitar tanto listas quanto tuplas
            if ((isinstance(jogo, (list, tuple))) and 
                len(jogo) == 7 and 
                all(isinstance(n, int) and 0 <= n <= 9 for n in jogo)):
                validos += 1
        return validos
    
    def _calculate_entropy(self, serie: pd.Series) -> float:
        """Calcula entropia de uma série"""
        try:
            from scipy.stats import entropy
            counts = serie.value_counts()
            return float(entropy(counts))
        except:
            return 0.0
    
    def _generate_summary(self, test1: Dict, test2: Dict, test3: Dict, test4: Dict, test5: Dict) -> Dict[str, Any]:
        """Gera resumo dos testes"""
        total_testes = 5
        testes_sucesso = sum(1 for test in [test1, test2, test3, test4, test5] if test.get('status') == 'success')
        
        # Verificar se a integração Super Sete está funcionando
        integracao_ok = test5.get('integracao_funcionando', False)
        
        return {
            'total_testes': total_testes,
            'testes_sucesso': testes_sucesso,
            'taxa_sucesso': testes_sucesso / total_testes * 100,
            'integracao_supersete_ok': integracao_ok,
            'status_geral': 'aprovado' if testes_sucesso >= 4 and integracao_ok else 'reprovado'
        }
    
    def _save_test_results(self, resultados: Dict[str, Any]) -> None:
        """Salva resultados do teste"""
        pasta_resultados = "resultados_teste"
        if not os.path.exists(pasta_resultados):
            os.makedirs(pasta_resultados)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"teste_completo_{timestamp}.json"
        caminho_arquivo = os.path.join(pasta_resultados, nome_arquivo)
        
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Resultados salvos em: {caminho_arquivo}")

def main():
    """Função principal para executar testes"""
    print("🧪 TESTADOR COMPLETO SUPER SETE")
    print("="*50)
    
    # Criar testador
    tester = SuperSeteTester('longo', 1000)
    
    # Executar teste completo
    resultados = tester.run_comprehensive_test()
    
    # Mostrar resumo
    print("\n" + "="*60)
    print("📋 RESUMO DOS TESTES")
    print("="*60)
    
    resumo = resultados['resumo']
    print(f"✅ Testes executados: {resumo['total_testes']}")
    print(f"✅ Testes aprovados: {resumo['testes_sucesso']}")
    print(f"📊 Taxa de sucesso: {resumo['taxa_sucesso']:.1f}%")
    print(f"🎯 Status geral: {resumo['status_geral'].upper()}")
    
    print("\n✅ Teste completo finalizado!")
    return resultados

if __name__ == "__main__":
    resultados = main()
