# Checklist de Entregas — Frente 1: Base de Código/Experimentos + Seção 3 (Metodologia)

**Disciplina:** FT108A – Introdução ao Aprendizado de Máquina (2s 2026)  
**Responsável:** Bruno Ricardo Corrêa (RA: 260759)  
**Prazo Final (Artigo IEEE + Apresentação):** 24/11/2026  

---

## 1. Execução no Código (Fechar as decisões em aberto do Pré-Projeto)

- [x] **1.1. Base de Dados e Seleção de Atributos**
  - Carregar a coorte NHANES 2021–2023 com as **20 variáveis preditoras** previamente selecionadas (12 numéricas contínuas + 8 categóricas/binárias).
  - Documentar o número total de registros elegíveis após filtro de adultos ($\ge 20$ anos) e exclusão de gestantes ($N = 7.723$, sendo $12,72\%$ positivos) e o total de casos completos após remoção de valores ausentes via `dropna()` ($N = 3.947$, sendo $3.497$ negativos [$88,60\%$] e $450$ positivos [$11,40\%$]).
  - **Como finalizar a demanda:**
    - Código já executado nas Células 5 e 7 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb). Basta reportar $N = 3.947$ casos completos ($11,4\%$ positivos) na Seção 3.1.

- [x] **1.2. Divisão Estratificada em Treino (70%), Validação (15%) e Teste (15%)**
  - Separar de forma estratificada (`stratify=y, random_state=42`) os $3.947$ registros completos em três partições mutuamente exclusivas:
    - **Treinamento (70%):** **$2.762$ amostras** ($315$ positivas — $11,40\%$).
    - **Validação (15%):** **$592$ amostras** ($67$ positivas — $11,32\%$).
    - **Teste Final / Holdout (15%):** **$593$ amostras** ($68$ positivas — $11,47\%$), reservado exclusivamente para a avaliação final.
  - **Como finalizar a demanda:**
    - Já implementado na Célula 14 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb). Reportar a divisão $70/15/15$ na Seção 3.1 do artigo.

- [x] **1.3. Pipeline de Pré-processamento sem Vazamento (*Data Leakage*)**
  - Implementar remoção de ausentes (`dropna()`), padronização de variáveis numéricas contínuas (`StandardScaler`) e codificação *one-hot* de variáveis categóricas (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) via `ColumnTransformer` e `ImbPipeline`.
  - Garantir que o `.fit()` das transformações ocorra exclusivamente nos dados de treinamento.
  - **Como finalizar a demanda:**
    - Código finalizado na Célula 14 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb).

- [x] **1.4. Experimento Comparativo do Desbalanceamento (3 Abordagens $\times$ 5 Modelos)**
  - Avaliar e comparar sistematicamente os **3 cenários de balanceamento** dentro do `ImbPipeline` (aplicados somente nas partições de treino):
    1. **Sem Reamostragem (`'passthrough'`):** Distribuição original desbalanceada ($11,4\%$ positivos).
    2. **Com `SMOTE` (*Over-sampling*):** Geração sintética de exemplos da classe minoritária.
    3. **Com `RandomUnderSampler` (*Under-sampling*):** Subamostragem aleatória da classe majoritária.
  - **Como finalizar a demanda:**
    - Experimento já executado nas Células 16 a 21 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb).

- [x] **1.5. Otimização Multi-Métrica no Treino (`AUC-ROC`, `F1` e `F2` com `refit='F1'`) e Avaliação na Validação (15%)**
  - Monitorar simultaneamente **`AUC-ROC`**, **`F1-Score`** e **`F2-Score`** durante o `RandomizedSearchCV` ($k=5$, `n_iter=20`) e selecionar os hiperparâmetros finais por **`refit='F1'`**.
  - **Por que `refit='F1'` no treinamento é superior a `refit='F2'` quando há reamostragem:** Como o `SMOTE` e o `RandomUnderSampler` já equilibram as classes em 50%/50% nos *folds* de treino (elevando naturalmente o *Recall*), otimizar os hiperparâmetros por `F1` evita penalizar duplamente a precisão e previne *overfitting*, resultando em maiores valores médios de **F2-Score**, **F1-Score**, **Recall** e **AUC-ROC** nos conjuntos de Validação e Teste.
  - **Como finalizar a demanda:**
    - Execução concluída na Célula 16 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb).

- [ ] **1.6. Geração e Envio dos Insumos para a Seção 4 (Resultados e Discussão)**
  - Extrair e enviar para o integrante responsável pela **Seção 4**:
    - [x] Tabela de validação ($15\%$) e tabela completa no conjunto de teste ($15\%$) das **3 abordagens $\times$ 5 modelos** (**Acurácia, Precisão, Recall, F1-score, F2-score e AUC-ROC**).
    - [x] Figura comparativa de barras (**F2-Score, Recall e AUC-ROC** entre Sem Reamostragem, SMOTE e RandomUnderSampler): [`figuras/metricas-teste-holdout.png`](figuras/metricas-teste-holdout.png).
    - [x] Figura com os 3 painéis de **Curvas ROC sobrepostas**: [`figuras/auc-roc-baseline.png`](figuras/auc-roc-baseline.png).
    - [x] Figura com a grade $3 \times 5$ de **Matrizes de Confusão**: [`figuras/matrizes-confusao.png`](figuras/matrizes-confusao.png).
  - **Como finalizar a demanda:**
    - Enviar no grupo a tabela abaixo, as imagens da pasta [`figuras/`](figuras/) e o link direto do Google Colab (`https://colab.research.google.com/github/brunorcorrea/FT108A-final-project/blob/main/nhanes-cvd-diagnosis.ipynb`).

### Tabela Completa para Envio à Seção 4 (Validação 15% [$N=592$] e Teste Final Holdout 15% [$N=593$])

| Abordagem | Modelo | Val F1 | Val F2 | Val AUC | Accuracy (Teste) | Precision (Teste) | Recall (Teste) | F1-Score (Teste) | F2-Score (Teste) | AUC-ROC (Teste) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Sem Reamostragem** | Logistic Regression | 0.3488 | 0.2613 | 0.8331 | 0.8820 | 0.4500 | 0.1324 | 0.2045 | 0.1541 | 0.8182 |
| **1. Sem Reamostragem** | KNN | 0.2292 | 0.1852 | 0.6917 | 0.8583 | 0.2895 | 0.1618 | 0.2075 | 0.1774 | 0.6536 |
| **1. Sem Reamostragem** | Decision Tree | 0.2542 | 0.2351 | 0.6850 | 0.8381 | 0.2500 | 0.2059 | 0.2258 | 0.2134 | 0.6220 |
| **1. Sem Reamostragem** | Random Forest | 0.2250 | 0.1601 | 0.8472 | **0.8836** | 0.4444 | 0.0588 | 0.1039 | 0.0712 | 0.8030 |
| **1. Sem Reamostragem** | XGBoost | 0.3226 | 0.2551 | 0.8467 | 0.8820 | **0.4643** | 0.1912 | 0.2708 | 0.2167 | 0.8100 |
| **2. SMOTE (Over-sampling)** | **Logistic Regression** | 0.3953 | 0.5507 | 0.8303 | 0.7673 | 0.2917 | 0.7206 | **0.4153** | 0.5568 | 0.8239 |
| **2. SMOTE (Over-sampling)** | KNN | 0.3439 | 0.4502 | 0.7735 | 0.7656 | 0.2767 | 0.6471 | 0.3877 | 0.5104 | 0.7614 |
| **2. SMOTE (Over-sampling)** | Decision Tree | 0.4180 | 0.5730 | 0.8205 | 0.7673 | 0.2892 | 0.7059 | 0.4103 | 0.5479 | 0.7831 |
| **2. SMOTE (Over-sampling)** | Random Forest | 0.3860 | 0.5128 | 0.8226 | 0.7875 | 0.3014 | 0.6471 | 0.4112 | 0.5263 | 0.8275 |
| **2. SMOTE (Over-sampling)** | **XGBoost** | **0.4220** | 0.5489 | 0.8339 | 0.7875 | 0.2958 | 0.6176 | 0.4000 | 0.5072 | **0.8277** |
| **3. RandomUnderSampler** | **Logistic Regression** | 0.3507 | 0.5011 | 0.8256 | 0.7437 | 0.2789 | 0.7794 | 0.4109 | **0.5736** | 0.8212 |
| **3. RandomUnderSampler** | KNN | 0.4130 | 0.5692 | 0.8182 | 0.7352 | 0.2620 | 0.7206 | 0.3843 | 0.5338 | 0.7791 |
| **3. RandomUnderSampler** | **Decision Tree** | 0.3794 | **0.5762** | 0.8013 | 0.6914 | 0.2422 | **0.7941** | 0.3711 | 0.5455 | 0.7799 |
| **3. RandomUnderSampler** | Random Forest | 0.3816 | 0.5579 | 0.8415 | 0.7218 | 0.2563 | 0.7500 | 0.3820 | 0.5414 | 0.8151 |
| **3. RandomUnderSampler** | XGBoost | 0.4030 | 0.5757 | **0.8523** | 0.7352 | 0.2620 | 0.7206 | 0.3843 | 0.5338 | 0.8101 |

---

## 2. Escrita da Seção 3 — Metodologia (Artigo IEEE)

- [ ] **2.1. Preencher e revisar o texto definitivo da Seção 3 (Metodologia)**
  - Converter o texto da proposta para o tempo passado, preenchendo todas as lacunas `[...]` com a divisão **Treino (70%) / Validação (15%) / Teste (15%)**, as 3 abordagens de balanceamento e o monitoramento multi-métrica (`AUC-ROC`, `F1` e `F2` com `refit='F1'`).
  - **Como finalizar a demanda:**
    - Copiar o texto **já preenchido abaixo** e colar diretamente no documento compartilhado do grupo (Overleaf / Word / Google Docs no formato IEEE).

### Texto da Seção 3 Já Preenchido com os Dados do Código (Pronto para Colar no Artigo)

> ### 3. Metodologia
> 
> **3.1. Conjunto de Dados e Pré-processamento**  
> Este estudo utilizou dados da coorte NHANES (*National Health and Nutrition Examination Survey*) referente ao ciclo 2021–2023. Após a filtragem de participantes adultos ($\ge 20$ anos), exclusão de gestantes e definição do desfecho cardiovascular ($N = 7.723$ elegíveis, com $12,7\%$ de prevalência de DCV), foram selecionadas 20 variáveis preditoras demográficas, clínicas, comportamentais e laboratoriais (12 contínuas e 8 categóricas/binárias) e removidos os registros com valores ausentes (*complete-case analysis*), totalizando **3.947 registros completos**. A variável-alvo binária indica a presença ou o registro de doenças cardiovasculares (insuficiência cardíaca, doença coronariana, angina, infarto agudo do miocárdio ou acidente vascular cerebral), apresentando distribuição desbalanceada com **11,4% de casos pertencentes à classe positiva (450 registros)** e **88,6% à classe negativa (3.497 registros)**.
> 
> Os dados foram divididos de forma estratificada na proporção de **70% / 15% / 15%** entre os conjuntos de **treinamento** (**2.762** amostras, sendo 315 positivas), **validação** (**592** amostras, sendo 67 positivas) e **teste** (**593** amostras, sendo 68 positivas), permanecendo o conjunto de teste estritamente reservado à avaliação final dos modelos. Para evitar vazamento de informações (*data leakage*), todas as etapas de transformação — incluindo a codificação de variáveis categóricas via **`OneHotEncoder`** e a padronização de atributos numéricos contínuos via **`StandardScaler`** — foram encapsuladas por meio das estruturas `Pipeline` e `ColumnTransformer` das bibliotecas *scikit-learn* e *imbalanced-learn*, com ajuste exclusivo sobre os dados de treinamento.
> 
> **3.2. Algoritmos, Treinamento e Tratamento do Desbalanceamento**  
> Foram aplicados e comparados cinco algoritmos de aprendizado de máquina supervisionado de diferentes níveis de complexidade: Regressão Logística (RL), adotada como modelo linear de referência; *K-Nearest Neighbors* (KNN), baseado na proximidade entre observações; Árvore de Decisão (AD), fundamentada em regras de decisão individuais; *Random Forest* (RF), baseado na combinação de múltiplas árvores independentes; e XGBoost, que emprega árvores construídas sequencialmente via *gradient boosting*.
> 
> Para investigar o impacto do desbalanceamento de classes sobre a sensibilidade dos classificadores, cada um dos cinco algoritmos foi treinado e avaliado sob **três abordagens experimentais**: (i) **Sem Reamostragem** (distribuição original desbalanceada); (ii) **Sobreamostragem Sintética via `SMOTE`** (*Synthetic Minority Over-sampling Technique*); e (iii) **Subamostragem Aleatória via `RandomUnderSampler`**, sendo qualquer reamostragem aplicada exclusivamente aos dados de treinamento de cada partição. O treinamento e o ajuste de hiperparâmetros dos 15 experimentos resultantes ($3 \times 5$) foram conduzidos sobre o conjunto de treinamento utilizando validação cruzada estratificada com cinco partições ($k=5$) associada ao método **`RandomizedSearchCV`** (20 iterações por modelo), monitorando simultaneamente **AUC-ROC**, **F1-score** e **F2-score** com seleção de hiperparâmetros (`refit`) pelo **F1-score** (evitando a dupla penalização da precisão nas partições já balanceadas por reamostragem) e acompanhamento intermediário no conjunto de validação (15%). Os principais hiperparâmetros explorados compreenderam: penalidade ($L_1, L_2$) e regularização $C \in [0,01; 10]$ na RL; número de vizinhos $k \in [3, 30]$, métrica de distância ($p \in \{1, 2\}$) e ponderação na KNN; profundidade máxima ($[3, 15]$), critério de divisão (*Gini* ou entropia) e mínimo de amostras por folha e divisão na AD; número de árvores ($[100, 400]$), profundidade máxima ($[4, 15]$) e subconjunto de atributos na RF; e número de estimadores ($[100, 400]$), taxa de aprendizado ($[0,01; 0,15]$), profundidade máxima ($[2, 7]$) e taxas de subamostragem no XGBoost.
> 
> **3.3. Métricas de Avaliação**  
> O desempenho preditivo nos conjuntos de validação e teste foi mensurado por meio de acurácia, precisão, *recall* (sensibilidade), F1-score, F2-score e área sob a curva ROC (AUC-ROC). Deu-se ênfase principal à AUC-ROC, a fim de avaliar a capacidade global de discriminação entre as classes, e ao F2-score ($\beta = 2$), que atribui peso superior ao *recall* em relação à precisão, priorizando a redução de falsos negativos na identificação de indivíduos com DCV em um cenário de triagem clínica.

---

## 3. Preparação para a Apresentação Final (24/11/2026)

- [ ] **3.1. Elaborar o Slide 1 — Dados e Divisão Treino/Validação/Teste (~1 min de fala)**
  - Coorte NHANES 2021–2023: 20 variáveis preditoras (12 contínuas + 8 categóricas), $N = 3.947$ casos completos após `dropna()`, desbalanceamento de $11,4\%$ positivos vs. $88,6\%$ negativos.
  - Divisão estratificada **70% Treino ($2.762$) / 15% Validação ($592$) / 15% Teste ($593$)** e encapsulamento via `ColumnTransformer` + `ImbPipeline` (`StandardScaler` + `OneHotEncoder`) sem vazamento de dados.
  - **Como finalizar a demanda:**
    - Criar o slide no template da apresentação do grupo com um diagrama visual simples de blocos: `NHANES (N=3.947, 20 vars)` $\rightarrow$ `Split Estratificado 70% / 15% / 15%` $\rightarrow$ `ColumnTransformer (Scaler + OneHot) apenas no Treino`.

- [ ] **3.2. Elaborar o Slide 2 — Modelagem, Validação e as 3 Abordagens de Balanceamento (~1,5 min de fala)**
  - Os 5 algoritmos comparados: **RL** (linear), **KNN** (instâncias), **AD** (árvore única), **RF** (*bagging ensemble*) e **XGBoost** (*boosting ensemble*).
  - Matriz experimental $3 \times 5$ (15 experimentos): **1. Sem Reamostragem**, **2. `SMOTE`** e **3. `RandomUnderSampler`**, com monitoramento multi-métrica (`AUC-ROC`, `F1`, `F2`), seleção por `F1` na validação cruzada ($k=5$), verificação na Validação (15%) e teste cego no Holdout (15%).
  - **Como finalizar a demanda:**
    - Inserir um esquema visual mostrando os 3 ramos experimentais (`Sem Reamostragem` vs. `SMOTE` vs. `RandomUnderSampler`) aplicados apenas dentro dos *folds* de treinamento.

- [ ] **3.3. Elaborar o Slide 3 — Critérios de Avaliação (~1 min de fala)**
  - Apresentar as 6 métricas calculadas (Acurácia, Precisão, Recall, F1, F2 e AUC-ROC).
  - Destacar o paradoxo da acurácia em bases desbalanceadas (sem reamostragem, a acurácia chega a $88\%$, mas o *recall* no teste cai para $5,9\%–20,6\%$) e por que **AUC-ROC** e **F2-score** ($\beta = 2$) são essenciais para evitar falsos negativos em triagem cardiovascular.
  - **Como finalizar a demanda:**
    - Incluir a fórmula visual do $F_\beta$ ($\beta = 2$) no slide e ensaiar a transição de fala de ~3 minutos dos seus 3 slides para o colega que apresentará a Seção 4 (Resultados e Discussão).
