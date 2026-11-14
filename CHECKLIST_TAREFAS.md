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

- [x] **1.2. Divisão Treino/Teste**
  - Fixar a separação estratificada na proporção **70/30** (`train_test_split(..., test_size=0.30, stratify=y, random_state=42)`).
  - Conjunto de treinamento: **$2.762$ amostras** ($315$ positivas — $11,40\%$).
  - Conjunto de teste (Holdout): **$1.185$ amostras** ($135$ positivas — $11,39\%$), reservado exclusivamente para a avaliação final.
  - **Como finalizar a demanda:**
    - Já implementado na Célula 14 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb). Reportar a divisão $70/30$ na Seção 3.1 do artigo.

- [x] **1.3. Pipeline de Pré-processamento sem Vazamento (*Data Leakage*)**
  - Implementar remoção de ausentes (`dropna()`), padronização de variáveis numéricas contínuas (`StandardScaler`) e codificação *one-hot* de variáveis categóricas (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) via `ColumnTransformer` e `ImbPipeline`.
  - Garantir que o `.fit()` das transformações ocorra exclusivamente nos dados de treinamento de cada partição.
  - **Como finalizar a demanda:**
    - Código finalizado na Célula 14 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb). Basta manter a descrição na Seção 3.1.

- [x] **1.4. Experimento Comparativo do Desbalanceamento (3 Abordagens $\times$ 5 Modelos)**
  - Avaliar e comparar sistematicamente os **3 cenários de balanceamento** dentro do `ImbPipeline` (aplicados somente nas partições de treino):
    1. **Sem Reamostragem (`'passthrough'`):** Distribuição original desbalanceada ($11,4\%$ positivos).
    2. **Com `SMOTE` (*Over-sampling*):** Geração sintética de exemplos da classe minoritária.
    3. **Com `RandomUnderSampler` (*Under-sampling*):** Subamostragem aleatória da classe majoritária.
  - **Como finalizar a demanda:**
    - Experimento já executado nas Células 16 a 21 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb). Destacar para o grupo o achado central: **sem reamostragem**, os modelos atingem alta acurácia aparente ($84\%–89\%$), mas falham severamente no *Recall* ($9,6\%–26,7\%$) e no *F2-Score* ($0,12–0,27$); **com `SMOTE` e `RandomUnderSampler`**, o *Recall* salta para **$65\%–78\%$** e o *F2-Score* para **$0,49–0,56$**.

- [x] **1.5. Validação Cruzada Estratificada ($k=5$) e Otimização de Hiperparâmetros**
  - Treinar os 5 algoritmos — **Regressão Logística (RL), KNN, Árvore de Decisão (AD), Random Forest (RF) e XGBoost** — nas 3 abordagens (15 treinamentos no total) com `StratifiedKFold(n_splits=5)` e `RandomizedSearchCV` (`n_iter=20`), otimizando o **`F2-score`** ($\beta = 2$).
  - **Como finalizar a demanda:**
    - Execução concluída na Célula 16 de [`nhanes-cvd-diagnosis.ipynb`](nhanes-cvd-diagnosis.ipynb).

- [ ] **1.6. Geração e Envio dos Insumos para a Seção 4 (Resultados e Discussão)**
  - Extrair e enviar para o integrante responsável pela **Seção 4**:
    - [x] Tabela completa das **3 abordagens $\times$ 5 modelos** no conjunto de teste (**Acurácia, Precisão, Recall, F1-score, F2-score e AUC-ROC**).
    - [x] Figura comparativa de barras (**F2-Score, Recall e AUC-ROC** entre Sem Reamostragem, SMOTE e RandomUnderSampler): [`figuras/metricas-teste-holdout.png`](figuras/metricas-teste-holdout.png).
    - [x] Figura com os 3 painéis de **Curvas ROC sobrepostas**: [`figuras/auc-roc-baseline.png`](figuras/auc-roc-baseline.png).
    - [x] Figura com a grade $3 \times 5$ de **Matrizes de Confusão**: [`figuras/matrizes-confusao.png`](figuras/matrizes-confusao.png).
  - **Como finalizar a demanda:**
    - Enviar no grupo a tabela abaixo, as imagens da pasta [`figuras/`](figuras/) e o link direto do Google Colab (`https://colab.research.google.com/github/brunorcorrea/FT108A-final-project/blob/main/nhanes-cvd-diagnosis.ipynb`).

### Tabela Completa para Envio à Seção 4 (Conjunto de Teste Holdout 30%, $N_{\text{teste}} = 1.185$)

| Abordagem | Modelo | Melhor F2 (CV 5-fold) | Accuracy | Precision | Recall | F1-Score | F2-Score | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Sem Reamostragem** | Logistic Regression | 0.2421 | **0.8937** | **0.6154** | 0.1778 | 0.2759 | 0.2073 | 0.8249 |
| **1. Sem Reamostragem** | KNN | 0.1909 | 0.8667 | 0.3284 | 0.1630 | 0.2178 | 0.1812 | 0.6725 |
| **1. Sem Reamostragem** | Decision Tree | 0.3021 | 0.8430 | 0.2927 | 0.2667 | 0.2791 | 0.2715 | 0.6408 |
| **1. Sem Reamostragem** | Random Forest | 0.1063 | 0.8895 | 0.5909 | 0.0963 | 0.1656 | 0.1157 | 0.8255 |
| **1. Sem Reamostragem** | XGBoost | 0.2698 | 0.8878 | 0.5185 | 0.2074 | 0.2963 | 0.2357 | 0.8278 |
| **2. SMOTE (Over-sampling)** | Logistic Regression | **0.5990** | 0.7502 | 0.2770 | 0.7407 | **0.4032** | 0.5549 | 0.8267 |
| **2. SMOTE (Over-sampling)** | KNN | 0.5478 | 0.6641 | 0.2123 | 0.7185 | 0.3277 | 0.4865 | 0.7621 |
| **2. SMOTE (Over-sampling)** | Decision Tree | 0.4907 | 0.7367 | 0.2548 | 0.6815 | 0.3710 | 0.5105 | 0.7754 |
| **2. SMOTE (Over-sampling)** | Random Forest | 0.5542 | 0.7755 | 0.2866 | 0.6519 | 0.3982 | 0.5195 | 0.8247 |
| **2. SMOTE (Over-sampling)** | XGBoost | 0.5491 | 0.7637 | 0.2769 | 0.6667 | 0.3913 | 0.5202 | 0.8249 |
| **3. RandomUnderSampler** | Logistic Regression | 0.5976 | 0.7165 | 0.2531 | 0.7630 | 0.3801 | 0.5438 | 0.8135 |
| **3. RandomUnderSampler** | KNN | 0.5665 | 0.7451 | 0.2725 | 0.7407 | 0.3984 | 0.5513 | 0.7976 |
| **3. RandomUnderSampler** | Decision Tree | 0.5213 | 0.7190 | 0.2435 | 0.6963 | 0.3608 | 0.5076 | 0.7554 |
| **3. RandomUnderSampler** | Random Forest | 0.5951 | 0.7131 | 0.2530 | **0.7778** | 0.3818 | 0.5497 | 0.8279 |
| **3. RandomUnderSampler** | **XGBoost** | 0.5873 | 0.7325 | 0.2655 | 0.7630 | 0.3939 | **0.5550** | **0.8302** |

---

## 2. Escrita da Seção 3 — Metodologia (Artigo IEEE)

- [ ] **2.1. Preencher e revisar o texto definitivo da Seção 3 (Metodologia)**
  - Converter o texto da proposta para o tempo passado, preenchendo todas as lacunas `[...]` com as decisões implementadas no notebook e o desenho comparativo das 3 abordagens de balanceamento.
  - **Como finalizar a demanda:**
    - Copiar o texto **já preenchido abaixo** e colar diretamente no documento compartilhado do grupo (Overleaf / Word / Google Docs no formato IEEE).

### Texto da Seção 3 Já Preenchido com os Dados do Código (Pronto para Colar no Artigo)

> ### 3. Metodologia
> 
> **3.1. Conjunto de Dados e Pré-processamento**  
> Este estudo utilizou dados da coorte NHANES (*National Health and Nutrition Examination Survey*) referente ao ciclo 2021–2023. Após a filtragem de participantes adultos ($\ge 20$ anos), exclusão de gestantes e definição do desfecho cardiovascular ($N = 7.723$ elegíveis, com $12,7\%$ de prevalência de DCV), foram selecionadas 20 variáveis preditoras demográficas, clínicas, comportamentais e laboratoriais (12 contínuas e 8 categóricas/binárias) e removidos os registros com valores ausentes (*complete-case analysis*), totalizando **3.947 registros completos**. A variável-alvo binária indica a presença ou o registro de doenças cardiovasculares (insuficiência cardíaca, doença coronariana, angina, infarto agudo do miocárdio ou acidente vascular cerebral), apresentando distribuição desbalanceada com **11,4% de casos pertencentes à classe positiva (450 registros)** e **88,6% à classe negativa (3.497 registros)**.
> 
> Os dados foram divididos de forma estratificada na proporção de **70/30** entre os conjuntos de treinamento (**2.762** amostras, sendo 315 positivas) e teste (**1.185** amostras, sendo 135 positivas), permanecendo o conjunto de teste estritamente reservado à avaliação final dos modelos. Para evitar vazamento de informações (*data leakage*), todas as etapas de transformação — incluindo a codificação de variáveis categóricas via **`OneHotEncoder`** e a padronização de atributos numéricos contínuos via **`StandardScaler`** — foram encapsuladas por meio das estruturas `Pipeline` e `ColumnTransformer` das bibliotecas *scikit-learn* e *imbalanced-learn*, com ajuste exclusivo sobre as partições de treinamento.
> 
> **3.2. Algoritmos, Treinamento e Tratamento do Desbalanceamento**  
> Foram aplicados e comparados cinco algoritmos de aprendizado de máquina supervisionado de diferentes níveis de complexidade: Regressão Logística (RL), adotada como modelo linear de referência; *K-Nearest Neighbors* (KNN), baseado na proximidade entre observações; Árvore de Decisão (AD), fundamentada em regras de decisão individuais; *Random Forest* (RF), baseado na combinação de múltiplas árvores independentes; e XGBoost, que emprega árvores construídas sequencialmente via *gradient boosting*.
> 
> Para investigar o impacto do desbalanceamento de classes sobre a sensibilidade dos classificadores, cada um dos cinco algoritmos foi treinado e avaliado sob **três abordagens experimentais**: (i) **Sem Reamostragem** (distribuição original desbalanceada); (ii) **Sobreamostragem Sintética via `SMOTE`** (*Synthetic Minority Over-sampling Technique*); e (iii) **Subamostragem Aleatória via `RandomUnderSampler`**, sendo qualquer reamostragem aplicada exclusivamente aos dados de treinamento de cada partição da validação cruzada. O treinamento e o ajuste de hiperparâmetros dos 15 experimentos resultantes ($3 \times 5$) foram conduzidos utilizando validação cruzada estratificada com cinco partições ($k=5$) associada ao método **`RandomizedSearchCV`** (20 iterações por modelo), otimizando diretamente a métrica **F2-score**. Os principais hiperparâmetros explorados compreenderam: penalidade ($L_1, L_2$) e regularização $C \in [0,01; 10]$ na RL; número de vizinhos $k \in [3, 30]$, métrica de distância ($p \in \{1, 2\}$) e ponderação na KNN; profundidade máxima ($[3, 15]$), critério de divisão (*Gini* ou entropia) e mínimo de amostras por folha e divisão na AD; número de árvores ($[100, 400]$), profundidade máxima ($[4, 15]$) e subconjunto de atributos na RF; e número de estimadores ($[100, 400]$), taxa de aprendizado ($[0,01; 0,15]$), profundidade máxima ($[2, 7]$) e taxas de subamostragem no XGBoost.
> 
> **3.3. Métricas de Avaliação**  
> O desempenho preditivo no conjunto de teste foi mensurado por meio de acurácia, precisão, *recall* (sensibilidade), F1-score, F2-score e área sob a curva ROC (AUC-ROC). Deu-se ênfase principal à AUC-ROC, a fim de avaliar a capacidade global de discriminação entre as classes, e ao F2-score ($\beta = 2$), que atribui peso superior ao *recall* em relação à precisão, priorizando a redução de falsos negativos na identificação de indivíduos com DCV em um cenário de triagem clínica.

---

## 3. Preparação para a Apresentação Final (24/11/2026)

- [ ] **3.1. Elaborar o Slide 1 — Dados e Pré-processamento (~1 min de fala)**
  - Coorte NHANES 2021–2023: 20 variáveis preditoras (12 contínuas + 8 categóricas), $N = 3.947$ casos completos após `dropna()`, desbalanceamento de $11,4\%$ positivos vs. $88,6\%$ negativos.
  - Divisão estratificada **70/30** ($2.762$ treino / $1.185$ teste) e encapsulamento via `ColumnTransformer` + `ImbPipeline` (`StandardScaler` + `OneHotEncoder`) sem vazamento de dados.
  - **Como finalizar a demanda:**
    - Criar o slide no template da apresentação do grupo com um diagrama visual simples de blocos: `NHANES (N=3.947, 20 vars)` $\rightarrow$ `Split Estratificado 70/30` $\rightarrow$ `ColumnTransformer (Scaler + OneHot) apenas no Treino`.

- [ ] **3.2. Elaborar o Slide 2 — Modelagem, Validação e as 3 Abordagens de Balanceamento (~1,5 min de fala)**
  - Os 5 algoritmos comparados: **RL** (linear), **KNN** (instâncias), **AD** (árvore única), **RF** (*bagging ensemble*) e **XGBoost** (*boosting ensemble*).
  - Matriz experimental $3 \times 5$ (15 experimentos): **1. Sem Reamostragem**, **2. `SMOTE`** e **3. `RandomUnderSampler`**, com validação cruzada estratificada ($k=5$) e `RandomizedSearchCV`.
  - **Como finalizar a demanda:**
    - Inserir um esquema visual mostrando os 3 ramos experimentais (`Sem Reamostragem` vs. `SMOTE` vs. `RandomUnderSampler`) aplicados apenas dentro dos *folds* de treinamento.

- [ ] **3.3. Elaborar o Slide 3 — Critérios de Avaliação (~1 min de fala)**
  - Apresentar as 6 métricas calculadas (Acurácia, Precisão, Recall, F1, F2 e AUC-ROC).
  - Destacar o paradoxo da acurácia em bases desbalanceadas (sem reamostragem, a acurácia chega a $89\%$, mas o *recall* cai para $9,6\%–26,7\%$) e por que **AUC-ROC** e **F2-score** ($\beta = 2$) são essenciais para evitar falsos negativos em triagem cardiovascular.
  - **Como finalizar a demanda:**
    - Incluir a fórmula visual do $F_\beta$ ($\beta = 2$) no slide e ensaiar a transição de fala de ~3 minutos dos seus 3 slides para o colega que apresentará a Seção 4 (Resultados e Discussão).
