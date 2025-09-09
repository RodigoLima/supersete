# Super Sete Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

Sistema de análise e previsão para jogos da loteria Super Sete, utilizando machine learning e análise estatística.

## Características Principais

- **Análise Científica**: Algoritmos de machine learning para análise de padrões
- **Múltiplas Configurações**: Presets otimizados para diferentes necessidades
- **Validação Robusta**: Testes estatísticos rigorosos dos modelos
- **Performance Otimizada**: Processamento paralelo e cache inteligente
- **Análises Especializadas**: Temporal, entropia e confiança
- **Geração de Jogos**: Múltiplos métodos de geração baseados em probabilidades

## Estrutura do Projeto

```
supersete/
├── src/                          # Código fonte principal
│   ├── core/                     # Módulos principais
│   │   ├── super_sete_analysis.py
│   │   ├── super_sete_analysis_revised.py
│   │   ├── common_interface.py
│   │   ├── covering_design.py
│   │   └── probability_calculator.py
│   ├── config/                   # Configurações
│   │   └── model_configs.py
│   ├── validation/               # Validação e testes
│   │   └── validator.py
│   └── guides/                   # Guias e estratégias
│       └── super_sete_strategy_guide.py
├── tests/                        # Testes automatizados
│   ├── integration/              # Testes de integração
│   │   └── test_my_model.py
│   └── __init__.py
├── data/                         # Dados do projeto
│   └── raw/                      # Dados brutos
│       └── Super Sete.xlsx
├── resultados/                   # Resultados das análises
│   ├── analise_*.json           # Resultados de análise
│   ├── guia_apostas.txt         # Guia de apostas gerado
│   ├── jogos_cientificos.txt    # Jogos recomendados
│   └── relatorio_detalhado.txt  # Relatórios detalhados
├── resultados_teste/             # Resultados de testes
│   └── teste_modelo_*.json      # Resultados de validação
├── model_cache/                  # Cache de modelos treinados
├── main.py                       # Script principal
├── pyproject.toml               # Configuração do projeto
├── requirements.txt             # Dependências básicas
└── README.md                    # Este arquivo
```

## Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação Básica

1. **Clone o repositório:**
```bash
git clone https://github.com/usuario/super-sete-analysis.git
cd super-sete-analysis
```

2. **Instale as dependências básicas:**
```bash
pip install -r requirements.txt
```

### Instalação com Funcionalidades Avançadas

Para funcionalidades de deep learning e otimização avançada:

```bash
# Instalação completa com todas as dependências
pip install -e ".[advanced,viz,optimization]"

# Ou instalação para desenvolvimento
pip install -e ".[dev,advanced,viz,optimization]"
```

### Instalação via pip (quando disponível)

```bash
pip install super-sete-analysis
```

### Dependências Opcionais

O projeto funciona com dependências mínimas, mas você pode instalar bibliotecas opcionais para funcionalidades avançadas:

```bash
# Para melhor performance em machine learning
pip install xgboost lightgbm

# Para deep learning (LSTM, redes neurais)
pip install tensorflow

# Para visualizações
pip install matplotlib seaborn plotly

# Para otimização de hiperparâmetros
pip install optuna hyperopt
```

## Uso Rápido

### Via Linha de Comando

```bash
# Análise básica
python main.py --config longo --modo analise

# Análise com validação
python main.py --config medio --modo ambos --concursos 500

# Análise rápida
python main.py --config curto --modo analise --verbose

# Análise com arquivo específico
python main.py --config longo --data-file "data/raw/Super Sete.xlsx" --output "meus_resultados"

# Validação apenas
python main.py --config medio --modo validacao --concursos 1000
```

### Via Python

```python
from src.core.common_interface import create_analyzer, quick_analysis

# Criar analisador
analyzer = create_analyzer('longo', n_concursos=1000)

# Análise rápida
resultado = quick_analysis("data/raw/Super Sete.xlsx", 'longo', n_jogos=10)
print(resultado)

# Análise completa
resultados = analyzer.run_analysis("data/raw/Super Sete.xlsx", n_jogos=20)
```

## Configurações

O projeto suporta diferentes configurações de análise:

| Configuração | Velocidade | Precisão | Uso Recomendado |
|-------------|------------|----------|-----------------|
| `curto` | Rápida | Média | Testes rápidos, prototipagem |
| `medio` | Moderada | Boa | Uso geral, análise balanceada |
| `longo` | Lenta | Alta | Análise completa, produção |

### Parâmetros de Configuração

- **`curto`**: Configuração rápida para testes e desenvolvimento
- **`medio`**: Configuração balanceada para uso geral
- **`longo`**: Configuração completa para análises detalhadas e produção

## Funcionalidades

### Análises Científicas
- **Análise de Probabilidades**: Cálculo de probabilidades estatísticas para cada número
- **Análise de Padrões**: Identificação de sequências e tendências nos dados históricos
- **Análise de Cobertura**: Algoritmos de covering design para otimização de jogos
- **Validação Estatística**: Testes rigorosos de confiabilidade dos modelos

### Machine Learning
- **Múltiplos Algoritmos**: Random Forest, XGBoost, LightGBM (quando disponível)
- **Ensemble Methods**: Combinação inteligente de diferentes modelos
- **Validação Cruzada**: Testes robustos com TimeSeriesSplit
- **Otimização de Hiperparâmetros**: Ajuste automático de parâmetros

### Geração de Jogos
- **Método de Ranking**: Baseado nas probabilidades mais altas
- **Amostragem Probabilística**: Seleção aleatória ponderada
- **Alta Confiança**: Jogos com maior probabilidade de acerto
- **Validação de Jogos**: Verificação automática de formato

### Módulos Especializados
- **`super_sete_analysis.py`**: Módulo principal de análise
- **`super_sete_analysis_revised.py`**: Versão revisada com melhorias
- **`covering_design.py`**: Algoritmos de covering design
- **`probability_calculator.py`**: Cálculos de probabilidades
- **`super_sete_strategy_guide.py`**: Guias de estratégia

### Performance
- **Cache Inteligente**: Armazenamento otimizado de modelos em `model_cache/`
- **Configurações Adaptativas**: Ajuste automático baseado nos dados
- **Otimização de Memória**: Uso eficiente de recursos
- **Processamento Eficiente**: Algoritmos otimizados para grandes volumes de dados

## Resultados Gerados

O projeto gera diversos tipos de resultados organizados em diretórios específicos:

### Resultados de Análise (`resultados/`)
- **`analise_*.json`**: Resultados detalhados das análises em formato JSON
- **`guia_apostas.txt`**: Guia de apostas com recomendações estratégicas
- **`jogos_cientificos.txt`**: Lista de jogos recomendados baseados na análise
- **`relatorio_detalhado.txt`**: Relatório completo com estatísticas e insights

### Resultados de Validação (`resultados_teste/`)
- **`teste_modelo_*.json`**: Resultados de validação dos modelos em formato JSON
- Métricas de performance e confiabilidade dos algoritmos

### Cache de Modelos (`model_cache/`)
- Modelos treinados salvos para reutilização
- Otimização de performance em execuções subsequentes

### Tipos de Análise Disponíveis
- **Análise de Frequência**: Estatísticas de aparição de cada número
- **Análise de Padrões**: Identificação de sequências e tendências
- **Análise de Probabilidades**: Cálculos estatísticos para previsões
- **Análise de Cobertura**: Otimização de jogos usando covering design

## Testes

### Executar Todos os Testes
```bash
python -m pytest tests/ -v
```

### Testes com Cobertura
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Testes Específicos
```bash
# Testes unitários
python -m pytest tests/unit/ -v

# Testes de integração
python -m pytest tests/integration/ -v

# Testes lentos (pular)
python -m pytest tests/ -m "not slow" -v
```

## Documentação

### Módulos Principais
- **`common_interface.py`**: Interface principal para criação de analisadores
- **`super_sete_analysis.py`**: Módulo principal de análise
- **`model_configs.py`**: Configurações dos modelos
- **`validator.py`**: Validação e testes dos modelos

### Guias Disponíveis
- **`super_sete_strategy_guide.py`**: Guia de estratégias para apostas
- Este README: Documentação completa do projeto

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Aviso Legal

Este software é destinado apenas para fins educacionais/experimento e de pesquisa. Não garante ganhos em jogos de loteria. Jogue com responsabilidade.
