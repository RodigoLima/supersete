#!/usr/bin/env python3
"""
Script Principal do Super Sete Analysis
======================================

Script de entrada principal para executar análises do Super Sete.
"""

import sys
import os
import argparse
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.common_interface import create_analyzer, quick_analysis, quick_validation

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Super Sete Analysis - Análise Inteligente de Jogos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
 c --config longo --modo analise
  python main.py --config medio --concursos 500 --modo ambos
  python main.py --config curto --output resultados/
        """
    )
    
    parser.add_argument('--config', '-c', 
                       choices=['curto', 'medio', 'longo' ], 
                       default='longo',
                       help='Configuração de análise (padrão: longo)')
    
    parser.add_argument('--concursos', '-n', 
                       type=int, 
                       default=99999999,
                       help='Número de concursos para análise (padrão: todos)')
    
    parser.add_argument('--modo', '-m',
                       choices=['analise', 'validacao', 'ambos'],
                       default='analise',
                       help='Modo de execução (padrão: analise)')
    
    parser.add_argument('--output', '-o',
                       default='resultados',
                       help='Diretório de saída (padrão: resultados/)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Modo verboso para debug')
    
    parser.add_argument('--data-file', '-d',
                       default='data/raw/Super Sete.xlsx',
                       help='Caminho para o arquivo de dados (padrão: data/raw/Super Sete.xlsx)')
    
    args = parser.parse_args()
    
    # Validações
    if args.concursos < 1:
        print("❌ Erro: Número de concursos deve ser maior que 0")
        sys.exit(1)
    
    if not os.path.exists(args.data_file):
        print(f"❌ Erro: Arquivo de dados não encontrado: {args.data_file}")
        sys.exit(1)
    
    print("🎯 Super Sete Analysis - Iniciando...")
    print(f"📊 Configuração: {args.config}")
    print(f"🎲 Concursos: {args.concursos}")
    print(f"⚙️  Modo: {args.modo}")
    print(f"📁 Arquivo: {args.data_file}")
    print(f"📂 Saída: {args.output}")
    if args.verbose:
        print("🔍 Modo verboso ativado")
    print("=" * 50)
    
    try:
        # Criar analisador
        if args.verbose:
            print("🔧 Criando analisador...")
        analyzer = create_analyzer(args.config, args.concursos)
        
        # Criar diretório de saída
        os.makedirs(args.output, exist_ok=True)
        
        if args.modo in ['analise', 'ambos']:
            print("\n🔍 Executando análise...")
            if args.verbose:
                print(f"📊 Analisando arquivo: {args.data_file}")
            
            resultado_analise = analyzer.test_model(args.data_file)
            print("✅ Análise concluída!")
            
            # Salvar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analise_file = f"{args.output}/analise_{timestamp}.json"
            with open(analise_file, 'w', encoding='utf-8') as f:
                import json
                from src.core.super_sete_analysis import to_serializable
                json.dump(resultado_analise, f, indent=2, ensure_ascii=False, default=to_serializable)
            print(f"💾 Resultados salvos em: {analise_file}")
        
        if args.modo in ['validacao', 'ambos']:
            print("\n🧪 Executando validação...")
            if args.verbose:
                print(f"📊 Validando arquivo: {args.data_file}")
            
            resultado_validacao = analyzer.validate_model(args.data_file)
            print("✅ Validação concluída!")
            
            # Salvar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            validacao_file = f"{args.output}/validacao_{timestamp}.json"
            with open(validacao_file, 'w', encoding='utf-8') as f:
                import json
                from src.core.super_sete_analysis import to_serializable
                json.dump(resultado_validacao, f, indent=2, ensure_ascii=False, default=to_serializable)
            print(f"💾 Resultados salvos em: {validacao_file}")
        
        print("\n🎉 Processo concluído com sucesso!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Arquivo não encontrado: {str(e)}")
        print("💡 Verifique se o arquivo de dados existe e o caminho está correto")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Erro de importação: {str(e)}")
        print("💡 Verifique se todas as dependências estão instaladas: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")
        if args.verbose:
            import traceback
            print("\n🔍 Detalhes do erro:")
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
