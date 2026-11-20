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

- [x] **2.1. Preencher e revisar o texto definitivo da Seção 3 (Metodologia) no Overleaf (`IEEEtran`)**
  - Converter o texto da proposta para o tempo passado, detalhando a filtragem da coorte ($11.933 \rightarrow 7.809 \rightarrow 7.723 \rightarrow 3.947$), as 20 variáveis preditoras, a divisão **Treino (70%) / Validação (15%) / Teste (15%)**, as 3 abordagens de balanceamento, o monitoramento multi-métrica (`AUC-ROC`, `F1` e `F2` com `refit='F1'`) e as citações BibTeX (`\cite{...}`).
  - **Como finalizar a demanda:**
    - Colar o bloco LaTeX abaixo diretamente em `\section{Metodologia}\label{sec:metodologia}` no Overleaf e adicionar as entradas abaixo no arquivo `bibliografia.bib`.

### Código LaTeX da Seção III — Metodologia (Pronto para o Overleaf)

```latex
\section{Metodologia}\label{sec:metodologia}

\subsection{Conjunto de Dados e Pré-processamento}\label{subsec:dados}
Os dados provêm do ciclo 2021--2023 do \textit{National Health and Nutrition Examination Survey} (NHANES), inicialmente com 11.933 participantes. A coorte foi restrita a adultos ($\ge 20$ anos; $n = 7.809$), faixa etária à qual se aplica o questionário de condições médicas. A variável-alvo binária (\texttt{target}) foi definida a partir do relato prévio de diagnóstico médico de insuficiência cardíaca congestiva (\texttt{MCQ160B}), doença arterial coronariana (\texttt{MCQ160C}), angina (\texttt{MCQ160D}), infarto agudo do miocárdio (\texttt{MCQ160E}) ou acidente vascular cerebral (\texttt{MCQ160F}), atribuindo-se $1$ na presença de ao menos uma condição e $0$ na ausência confirmada de todas. Após descartar respostas indeterminadas (recusa ou desconhecimento, códigos 7 e 9) e excluir gestantes (\texttt{RIDEXPRG} $= 1$), restaram 7.723 adultos elegíveis (982 positivos; 12,72\%).

Na etapa de limpeza e engenharia de atributos, as leituras de pressão arterial sistólica e diastólica foram consolidadas pela média da segunda e terceira aferições (com \textit{fallback} para a primeira leitura), e os valores ausentes de colesterol total e triglicerídeos na amostra de referência foram complementados pelo painel bioquímico padrão. Foram derivadas variáveis ordinais para tabagismo (nunca fumou, fumante diário, fumante ocasional e ex-fumante) e consumo de álcool (abstêmio, ocasional, moderado e intenso).

Foram selecionadas 20 variáveis preditoras: 12 contínuas (idade, razão renda familiar/pobreza [PIR], índice de massa corporal [IMC], circunferência da cintura, pressão arterial sistólica e diastólica, colesterol total, colesterol HDL, triglicerídeos, hemoglobina glicada [HbA1c], proteína C-reativa de alta sensibilidade [PCR] e creatinina sérica) e 8 categóricas ou binárias (sexo, raça/etnia, escolaridade, tabagismo, consumo de álcool e diagnósticos autorreferidos de diabetes, hipertensão e dislipidemia). Para evitar a introdução de valores artificiais em biomarcadores laboratoriais antes da reamostragem, adotou-se a análise de casos completos (\texttt{dropna}), resultando na amostra analítica final de $N = 3.947$ participantes, com 450 casos positivos (11,40\%) e 3.497 negativos (88,60\%).

A base final foi dividida de forma estratificada na proporção de 70\% para treinamento (2.762 amostras; 315 positivas, 11,40\%), 15\% para validação (592 amostras; 67 positivas, 11,32\%) e 15\% para teste (593 amostras; 68 positivas, 11,47\%), mantendo-se o conjunto de teste estritamente isolado para a avaliação final. A preparação dos atributos foi encapsulada em um \texttt{ColumnTransformer} (\textit{scikit-learn}), aplicando padronização (\texttt{StandardScaler}) às 12 variáveis contínuas e codificação binária (\texttt{OneHotEncoder}) às 8 variáveis categóricas, ajustadas exclusivamente sobre os dados de treinamento para impedir vazamento de informação (\textit{data leakage}).

\subsection{Algoritmos, Balanceamento e Ajuste de Hiperparâmetros}\label{subsec:treinamento}
Foram avaliados cinco classificadores supervisionados de diferentes famílias algorítmicas: Regressão Logística (RL) \cite{cox1958regressao}, como modelo linear de referência; \textit{K-Nearest Neighbors} (KNN) \cite{cover1967knn}, baseado em distância entre instâncias; Árvore de Decisão (AD) \cite{breiman1984cart}, baseada em particionamento recursivo; \textit{Random Forest} (RF) \cite{breiman2001rf}, comitê em paralelo baseado em \textit{bagging}; e XGBoost \cite{chen2016xgboost}, comitê sequencial baseado em \textit{gradient boosting}.

Para investigar o efeito do desbalanceamento de classes (11,40\% de positivos), cada algoritmo foi submetido a três abordagens experimentais integradas via \texttt{ImbPipeline} (\textit{imbalanced-learn}), aplicadas exclusivamente dentro das partições de treinamento \cite{he2009desbalanceamento}:
\begin{enumerate}
    \item \textit{Sem Reamostragem (Baseline)}: treinamento direto sobre a distribuição original desbalanceada;
    \item \textit{SMOTE (\textit{Over-sampling})} \cite{chawla2002smote}: geração sintética de exemplos da classe minoritária por interpolação entre vizinhos próximos até o equilíbrio entre as classes;
    \item \textit{RandomUnderSampler (\textit{Under-sampling})} \cite{he2009desbalanceamento}: subamostragem aleatória da classe majoritária até igualar o número de exemplos reais da classe positiva.
\end{enumerate}

O treinamento e a otimização dos hiperparâmetros dos 15 experimentos resultantes ($3 \times 5$) foram conduzidos sobre o conjunto de treinamento utilizando validação cruzada estratificada com cinco partições ($k = 5$) associada ao \texttt{RandomizedSearchCV} (20 iterações por configuração, semente fixa igual a 42). Durante a validação cruzada, monitoraram-se simultaneamente a AUC-ROC, o F1-score e o F2-score, adotando-se o F1-score como critério de seleção final dos hiperparâmetros (\texttt{refit}), de modo a preservar o equilíbrio entre precisão e sensibilidade sem penalizar duplamente a precisão nas partições já balanceadas em 50\%/50\% pela reamostragem. O conjunto de validação (15\%) foi utilizado para verificação intermediária dos modelos ajustados.

O espaço de busca compreendeu: penalidade ($L_1, L_2$) e constante de regularização $C \in [0,01; 10]$ na RL; número de vizinhos $k \in [3, 30]$, métrica de Minkowski ($p \in \{1, 2\}$) e ponderação por distância no KNN; profundidade máxima ($[3, 15]$), critério de pureza (\textit{Gini} ou entropia) e mínimo de amostras por divisão ($[2, 19]$) e por folha ($[2, 14]$) na AD; número de árvores ($[100, 399]$), profundidade máxima ($[4, 15]$) e subconjunto de atributos (\texttt{sqrt} ou \texttt{log2}) no RF; e número de árvores ($[100, 399]$), taxa de aprendizado ($[0,01; 0,15]$), profundidade máxima ($[2, 7]$) e taxas de subamostragem de instâncias e colunas ($[0,6; 1,0]$) no XGBoost.

\subsection{Critérios e Métricas de Avaliação}\label{subsec:metricas}
O desempenho final no conjunto de teste independente ($n = 593$) foi mensurado por acurácia, precisão, \textit{recall} (sensibilidade), F1-score, F2-score e área sob a curva ROC (AUC-ROC) \cite{fawcett2006roc,sokolova2009metricas}, além da análise das matrizes de confusão normalizadas e da sobreposição das curvas ROC. Deu-se ênfase principal à AUC-ROC, para avaliar a capacidade global de discriminação independentemente do limiar de decisão \cite{fawcett2006roc}, e ao F2-score ($\beta = 2$) \cite{sokolova2009metricas}, definido por:
\begin{equation}
    F_2 = \frac{5 \cdot \text{Precisão} \cdot \text{Recall}}{4 \cdot \text{Precisão} + \text{Recall}},
\end{equation}
que atribui peso superior ao \textit{recall} em relação à precisão, priorizando a redução de falsos negativos na identificação de indivíduos com DCV em um cenário de triagem clínica.
```

### Entradas BibTeX Correspondentes (`bibliografia.bib`)

```bibtex
@misc{oms2025dcv,
  author       = {{World Health Organization}},
  title        = {Cardiovascular diseases ({CVDs})},
  year         = {2025},
  howpublished = {\url{https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)}},
  note         = {Acesso em: 01 nov. 2026}
}

@article{ameen2024advances,
  author  = {Ameen, Asmaa and others},
  title   = {Advances in {ECG} and {PCG}-based cardiovascular disease classification: a review of deep learning and machine learning methods},
  journal = {Journal of Big Data},
  volume  = {11},
  pages   = {159},
  year    = {2024},
  doi     = {10.1186/s40537-024-01011-7}
}

@article{curado2021cost,
  author  = {Curado, Daniel da Silva Pereira and others},
  title   = {Direct cost of systemic arterial hypertension and its complications in the circulatory system from the perspective of the {Brazilian} public health system in 2019},
  journal = {PLOS ONE},
  volume  = {16},
  number  = {6},
  pages   = {e0253063},
  year    = {2021},
  doi     = {10.1371/journal.pone.0253063}
}

@article{xu2025development,
  author  = {Xu, Chunming and others},
  title   = {Development and validation of a machine learning model for cardiovascular disease risk prediction in type 2 diabetes patients},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {32818},
  year    = {2025},
  doi     = {10.1038/s41598-025-18443-7}
}

@article{li2023vies,
  author  = {Li, Y. and others},
  title   = {Machine learning for cardiovascular risk prediction and stratification in large clinical cohorts},
  journal = {European Heart Journal - Digital Health},
  volume  = {4},
  pages   = {1--10},
  year    = {2023}
}

@article{singh2024risco,
  author  = {Singh, Manasvi and others},
  title   = {Artificial intelligence for cardiovascular disease risk assessment in personalised framework: a scoping review},
  journal = {eClinicalMedicine},
  volume  = {73},
  pages   = {102660},
  year    = {2024},
  doi     = {10.1016/j.eclinm.2024.102660}
}

@article{cox1958regressao,
  author  = {Cox, David R.},
  title   = {The Regression Analysis of Binary Sequences},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume  = {20},
  number  = {2},
  pages   = {215--242},
  year    = {1958}
}

@article{cover1967knn,
  author  = {Cover, Thomas and Hart, Peter},
  title   = {Nearest neighbor pattern classification},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13},
  number  = {1},
  pages   = {21--27},
  year    = {1967},
  doi     = {10.1109/TIT.1967.1053964}
}

@book{breiman1984cart,
  author    = {Breiman, Leo and Friedman, Jerome H. and Olshen, Richard A. and Stone, Charles J.},
  title     = {Classification and Regression Trees},
  publisher = {Wadsworth International Group},
  address   = {Belmont, CA, USA},
  year      = {1984}
}

@article{breiman2001rf,
  author  = {Breiman, Leo},
  title   = {Random Forests},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001},
  doi     = {10.1023/A:1010933404324}
}

@inproceedings{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016},
  doi       = {10.1145/2939672.2939785}
}

@article{he2009desbalanceamento,
  author  = {He, Haibo and Garcia, Edwardo A.},
  title   = {Learning from Imbalanced Data},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {21},
  number  = {9},
  pages   = {1263--1284},
  year    = {2009},
  doi     = {10.1109/TKDE.2008.239}
}

@article{chawla2002smote,
  author  = {Chawla, Nitesh V. and Bowyer, Kevin W. and Hall, Lawrence O. and Kegelmeyer, W. Philip},
  title   = {{SMOTE}: Synthetic Minority Over-sampling Technique},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {16},
  pages   = {321--357},
  year    = {2002},
  doi     = {10.1613/jair.953}
}

@article{fawcett2006roc,
  author  = {Fawcett, Tom},
  title   = {An Introduction to {ROC} Analysis},
  journal = {Pattern Recognition Letters},
  volume  = {27},
  number  = {8},
  pages   = {861--874},
  year    = {2006},
  doi     = {10.1016/j.patrec.2005.10.010}
}

@article{sokolova2009metricas,
  author  = {Sokolova, Marina and Lapalme, Guy},
  title   = {A Systematic Analysis of Performance Measures for Classification Tasks},
  journal = {Information Processing \& Management},
  volume  = {45},
  number  = {4},
  pages   = {427--437},
  year    = {2009},
  doi     = {10.1016/j.ipm.2009.03.002}
}
```

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
