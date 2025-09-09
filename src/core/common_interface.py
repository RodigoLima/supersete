#!/usr/bin/env python3
"""
Interface Comum para Análise Super Sete
=======================================

Este módulo fornece uma interface unificada para todos os scripts
de análise, validação e teste do Super Sete.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .super_sete_analysis import SuperSeteAnalyzer, ModelConfig, TemporalAnalyzer, EntropyAnalyzer, ConfidenceAnalyzer
from ..config.model_configs import get_model_config

class SuperSeteInterface:
    """Interface unificada para análise Super Sete"""
    
    def __init__(self, config_name: str = 'longo', n_concursos: int = 1000):
        """
        Inicializa a interface com configuração unificada
        
        Args:
            config_name: Nome da configuração ('curto', 'medio', 'longo')
            n_concursos: Número de concursos para análise
            
        Raises:
            ValueError: Se config_name ou n_concursos forem inválidos
        """
        if config_name not in ['rapido', 'medio', 'longo', 'ultra']:
            raise ValueError(f"config_name deve ser 'rapido', 'medio', 'longo' ou 'ultra', recebido: {config_name}")
        
        if n_concursos < 1:
            raise ValueError(f"n_concursos deve ser maior que 0, recebido: {n_concursos}")
        
        self.config_name = config_name
        self.n_concursos = n_concursos
        
        try:
            # Configuração do modelo
            self.model_config = get_model_config(config_name, n_concursos)
            
            # Analisador principal
            self.analyzer = SuperSeteAnalyzer(self.model_config)
            
            # Analisadores especializados
            self.temporal_analyzer = self.analyzer.temporal_analyzer
            self.entropy_analyzer = self.analyzer.entropy_analyzer
            self.confidence_analyzer = self.analyzer.confidence_analyzer
            
            print(f"🔧 Interface inicializada com configuração: {config_name}")
            
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar interface: {e}")
    
    def load_data(self, excel_path: str) -> pd.DataFrame:
        """
        Carrega dados do Excel
        
        Args:
            excel_path: Caminho para o arquivo Excel
            
        Returns:
            pd.DataFrame: Dados carregados
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o arquivo não contiver dados válidos
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {excel_path}")
        
        try:
            df = self.analyzer.load_data(excel_path)
            if df.empty:
                raise ValueError("Arquivo Excel está vazio ou não contém dados válidos")
            return df
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados: {e}")
    
    def run_analysis(self, excel_path: str, **kwargs) -> Dict[str, Any]:
        """
        Executa análise completa
        
        Args:
            excel_path: Caminho para o arquivo Excel
            **kwargs: Argumentos adicionais para a análise
            
        Returns:
            Dict[str, Any]: Resultados da análise
        """
        try:
            return self.analyzer.run_analysis(excel_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Erro na análise: {e}")
    
    def analyze_column(self, series: np.ndarray, column_name: str) -> Dict[str, Any]:
        """
        Analisa uma coluna específica
        
        Args:
            series: Série de dados para análise
            column_name: Nome da coluna
            
        Returns:
            Dict[str, Any]: Resultados da análise da coluna
        """
        if not isinstance(series, np.ndarray):
            raise TypeError("series deve ser um numpy.ndarray")
        
        if len(series) == 0:
            raise ValueError("series não pode estar vazia")
        
        try:
            return self.analyzer.analyze_column(series, column_name)
        except Exception as e:
            raise RuntimeError(f"Erro na análise da coluna {column_name}: {e}")
    
    def generate_games(self, n_jogos: int = 10, method: str = 'ranking') -> List[List[int]]:
        """
        Gera jogos usando o método especificado
        
        Args:
            n_jogos: Número de jogos a gerar
            method: Método de geração ('ranking', 'sampling', 'high_confidence')
            
        Returns:
            List[List[int]]: Lista de jogos gerados
        """
        if n_jogos < 1:
            raise ValueError("n_jogos deve ser maior que 0")
        
        if method not in ['ranking', 'sampling', 'high_confidence']:
            raise ValueError("method deve ser 'ranking', 'sampling' ou 'high_confidence'")
        
        try:
            return self.analyzer.generate_games(n_jogos, method)
        except Exception as e:
            raise RuntimeError(f"Erro na geração de jogos: {e}")
    
    def validate_model(self, excel_path: str, validation_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Executa validação do modelo
        
        Args:
            excel_path: Caminho para o arquivo Excel
            validation_config: Configuração de validação (opcional)
            
        Returns:
            Dict[str, Any]: Resultados da validação
        """
        from ..validation.validator import SuperSeteValidator, SuperSeteValidationConfig
        
        # Validação do arquivo
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {excel_path}")
        
        # Configuração padrão de validação
        if validation_config is None:
            validation_config = {
                'test_size': 0.2,
                'cv_folds': 5,
                'min_samples': 30,
                'confidence_level': 0.95,
                'save_plots': True,
                'results_dir': "resultados_validacao",
                'plot_dir': "resultados_validacao/plots"
            }
        
        try:
            config = SuperSeteValidationConfig(**validation_config)
            validator = SuperSeteValidator(config, self.model_config)
            return validator.run_full_validation(excel_path)
        except Exception as e:
            raise RuntimeError(f"Erro na validação: {e}")
    
    def test_model(self, excel_path: str, n_jogos: int = 10) -> Dict[str, Any]:
        """
        Testa o modelo atual
        
        Args:
            excel_path: Caminho para o arquivo Excel
            n_jogos: Número de jogos para teste
            
        Returns:
            Dict[str, Any]: Resultados do teste
        """
        if n_jogos < 1:
            raise ValueError("n_jogos deve ser maior que 0")
        
        print("🧪 TESTANDO MODELO ATUAL")
        print("="*50)
        
        try:
            # Carregar dados
            df = self.load_data(excel_path)
            print(f"📊 Dados carregados: {len(df)} concursos")
            
            # Executar análise
            resultados = self.run_analysis(
                excel_path=excel_path,
                n_jogos=n_jogos,
                topk=5,
                seed=42,
                use_cache=True,
                save_results=True
            )
        except Exception as e:
            raise RuntimeError(f"Erro no teste do modelo: {e}")
        
        # Analisar resultados
        print("\n📈 ANÁLISE DOS RESULTADOS:")
        print("-" * 40)
        
        # Verificar confiança média
        avg_confidence = resultados.get('avg_confidence', 0)
        print(f"🎯 Confiança média: {avg_confidence:.3f}")
        
        if avg_confidence > 0.6:
            print("✅ Alta confiança - modelo promissor")
        elif avg_confidence > 0.4:
            print("⚠️  Confiança moderada - use com cautela")
        else:
            print("❌ Baixa confiança - modelo precisa de ajustes")
        
        # Verificar colunas previsíveis
        predictable_cols = resultados.get('predictable_cols', 0)
        total_cols = len([col for col in df.columns if col.startswith('Coluna_')])
        print(f"📊 Colunas previsíveis: {predictable_cols}/{total_cols}")
        
        if predictable_cols / total_cols > 0.5:
            print("✅ Boa parte das colunas é previsível")
        else:
            print("⚠️  Poucas colunas são previsíveis")
        
        # Salvar relatório
        self._save_test_report(resultados, df)
        
        return resultados
    
    def _save_test_report(self, resultados: Dict, df: pd.DataFrame) -> None:
        """Salva relatório de teste"""
        # Criar pasta para resultados se não existir
        pasta_resultados = "resultados_teste"
        if not os.path.exists(pasta_resultados):
            os.makedirs(pasta_resultados)
            print(f"📁 Pasta criada: {pasta_resultados}")
        
        # Gerar nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"teste_modelo_{timestamp}.json"
        caminho_arquivo = os.path.join(pasta_resultados, nome_arquivo)
        
        # Salvar relatório de teste
        relatorio = {
            'data_teste': str(pd.Timestamp.now()),
            'concursos_analisados': len(df),
            'confianca_media': resultados.get('avg_confidence', 0),
            'colunas_previsiveis': resultados.get('predictable_cols', 0),
            'total_colunas': len([col for col in df.columns if col.startswith('Coluna_')]),
            'taxa_previsibilidade': resultados.get('predictable_cols', 0) / len([col for col in df.columns if col.startswith('Coluna_')]),
            'jogos_gerados': {
                'amostragem': len(resultados.get('jogos_amostragem', [])),
                'ranking': len(resultados.get('jogos_ranking', [])),
                'alta_confianca': len(resultados.get('jogos_alta_confianca', []))
            },
            'qualidade_geral': 'boa' if resultados.get('avg_confidence', 0) > 0.6 else 'moderada' if resultados.get('avg_confidence', 0) > 0.4 else 'baixa'
        }
        
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            from .super_sete_analysis import to_serializable
            json.dump(relatorio, f, indent=2, ensure_ascii=False, default=to_serializable)
        
        print(f"\n💾 Relatório salvo em: {caminho_arquivo}")
    
    def get_analysis_summary(self, excel_path: str) -> Dict[str, Any]:
        """Obtém resumo da análise sem executar análise completa"""
        df = self.load_data(excel_path)
        
        summary = {
            'total_concursos': len(df),
            'total_colunas': len([col for col in df.columns if col.startswith('Coluna_')]),
            'configuracao': self.config_name,
            'n_concursos_config': self.n_concursos,
            'data_analise': str(pd.Timestamp.now())
        }
        
        return summary

# Funções de conveniência para uso direto
def create_analyzer(config_name: str = 'longo', n_concursos: int = 1000) -> SuperSeteInterface:
    """Cria uma instância da interface"""
    return SuperSeteInterface(config_name, n_concursos)

def quick_analysis(excel_path: str, config_name: str = 'longo', n_jogos: int = 10) -> Dict[str, Any]:
    """Executa análise rápida"""
    interface = SuperSeteInterface(config_name)
    return interface.test_model(excel_path, n_jogos)

def quick_validation(excel_path: str, config_name: str = 'longo') -> Dict[str, Any]:
    """Executa validação rápida"""
    interface = SuperSeteInterface(config_name)
    return interface.validate_model(excel_path)

if __name__ == "__main__":
    # Exemplo de uso
    print("🚀 INTERFACE COMUM SUPER SETE")
    print("="*50)
    
    # Criar interface
    interface = create_analyzer('longo', 1000)
    
    # Executar análise rápida
    resultados = quick_analysis("Super Sete.xlsx", 'longo', 5)
    
    print("\n✅ Análise concluída!")
