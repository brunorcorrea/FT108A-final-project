# FT108A – Classificação Supervisionada de Doenças Cardiovasculares (NHANES 2021–2023)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brunorcorrea/FT108A-final-project/blob/main/nhanes-cvd-diagnosis.ipynb)

Projeto final da disciplina **FT108A – Introdução ao Aprendizado de Máquina (2s 2026)** da Faculdade de Tecnologia da UNICAMP.

## Integrantes do Grupo
- **Bruno Ricardo Corrêa** (RA: 260759)
- **Cristiane Brandão Cobo** (RA: 288831)
- **Gustavo Bilia Marques** (RA: 279808)
- **Julio Morino Anastácio** (RA: 173434)
- **Sabrina Tomaz Barbosa** (RA: 324065)

---

## Como Executar no Google Colab (Sem precisar configurar nada localmente)
1. Clique no botão **[Open in Colab](https://colab.research.google.com/github/brunorcorrea/FT108A-final-project/blob/main/nhanes-cvd-diagnosis.ipynb)** acima.
2. No menu superior do Colab, clique em **Ambiente de execução (Runtime) → Executar tudo (Run all)**.
3. O próprio notebook detecta automaticamente que está no Colab e baixa a base `data/nhanes_2021_2023_merged_data.csv` direto deste repositório do GitHub.

---

## Estrutura do Repositório
- `nhanes-cvd-diagnosis.ipynb`: Notebook principal contendo limpeza da coorte NHANES 2021–2023, seleção das 20 variáveis preditoras, remoção de ausentes (`dropna`), análise exploratória (EDA), otimização de hiperparâmetros com validação cruzada estratificada ($k=5$) e balanceamento (`SMOTE` vs. `RandomUnderSampler`) para 5 classificadores (**Regressão Logística, KNN, Árvore de Decisão, Random Forest e XGBoost**).
- `data/nhanes_2021_2023_merged_data.csv`: Base de dados consolidada da coorte NHANES 2021–2023.
- `figuras/`: Gráficos de alta resolução (300 DPI) gerados pelo notebook para o artigo no formato IEEE.
