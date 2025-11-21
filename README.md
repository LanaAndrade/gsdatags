# Sistema Inteligente de Recomendação de Carreira

## Integrantes
- Caio Freitas – RM553190
- Enzzo Monteiro – RM552616
- Lana Andrade – RM552596

---

# 1. Descrição da Solução Proposta

Este projeto implementa um sistema inteligente que utiliza algoritmos de Machine Learning para identificar qual trilha de carreira é mais adequada para um usuário, com base nas informações do seu currículo.

O sistema analisa:
- Linguagens de programação utilizadas
- Bancos de dados já trabalhados
- Anos de experiência
- Nível de formação
- Tipo de emprego
- Skills técnicas informadas

Com essas informações, o modelo recomenda uma trilha de carreira entre:
- Backend  
- Frontend  
- Data Science / Machine Learning  
- DevOps / SRE  
- Mobile  
- QA  
- UX/UI  

Toda a pipeline é automatizada pelo arquivo `main.py`, que executa:
1. Análise exploratória dos dados  
2. Engenharia de atributos  
3. Pré-processamento  
4. Treino e comparação de vários modelos  
5. Seleção do melhor algoritmo  
6. Geração de análises e gráficos  
7. Recomendação final baseada no perfil informado pelo usuário  

---

# 2. Objetivo Geral

Criar um sistema capaz de analisar o perfil profissional do usuário e recomendar automaticamente a trilha de carreira mais compatível, utilizando técnicas de Machine Learning.

---

# 3. Objetivos Específicos

- Implementar uma pipeline completa de Machine Learning.
- Realizar engenharia de atributos no dataset StackOverflow Developer Survey.
- Comparar diferentes algoritmos e selecionar o melhor utilizando validação cruzada.
- Aplicar regularização para evitar overfitting nos modelos lineares.
- Gerar gráficos de desempenho e análise do modelo.
- Criar um sistema de recomendação explicativo e interpretável.
- Unificar todo o processo em um único comando (`python main.py`).

---

# 4. Requirements

## Requisitos Funcionais
- RF01: Carregar e processar dados do usuário.
- RF02: Tratar informações sobre skills, formação e experiência.
- RF03: Executar automaticamente todo o pipeline de Machine Learning.
- RF04: Comparar múltiplos algoritmos com validação cruzada.
- RF05: Selecionar o melhor modelo com base na métrica Macro-F1.
- RF06: Gerar gráficos e análises automáticas.
- RF07: Retornar uma recomendação de carreira com explicação.

## Requisitos Não Funcionais
- RNF01: Código modular e organizado em múltiplos arquivos.
- RNF02: Execução centralizada via `main.py`.
- RNF03: Resposta final da recomendação em menos de 2 segundos.
- RNF04: Reprodutibilidade garantida por semente fixa (`random_state`).
- RNF05: Documentação clara em formato README.

## Requisitos Técnicos
- Python 3.10 ou superior  
- Pandas, NumPy  
- Scikit-Learn  
- Joblib  
- Matplotlib e Seaborn  
- Dataset oficial StackOverflow Developer Survey  

---

# 5. Modelos Utilizados

## Baselines
- Majority Class (classe mais frequente)
- Regras simples baseadas em skills

## Modelos de Machine Learning testados
- Regressão Logística (multiclasse, regularização L2)
- LinearSVC
- KNN (k = 3, 5 e 7)
- RandomForest
- Gradient Boosting
- MLPClassifier (rede neural simples)

## Estratégias de Treinamento
- Validação cruzada estratificada (Stratified 5-Fold)
- GridSearchCV para seleção de hiperparâmetros
- Regularização L2 em modelos lineares
- Métrica principal: Macro-F1
- Métricas adicionais: Acurácia, Precisão, Recall, Matriz de confusão

---

# 6. Dados Utilizados

O dataset utilizado é o **Stack Overflow Developer Survey 2025**.  
O arquivo deve estar em: data/survey_results_public.csv


Principais colunas utilizadas:
- DevType  
- LanguageHaveWorkedWith  
- DatabaseHaveWorkedWith  
- YearsCodePro  
- EdLevel  
- Employment  

---

# 7. Fluxo do Sistema

1. Leitura do dataset.  
2. Engenharia de atributos (transformação de anos, habilidades e categorias).  
3. Vetorização multi-hot de skills.  
4. One-hot encoding de variáveis categóricas.  
5. Divisão treino/teste.  
6. Treino de vários modelos de Machine Learning.  
7. Validação cruzada e escolha do melhor modelo.  
8. Geração automática de análises e gráficos.  
9. Recomendação final de carreira com explicação baseada em skills relevantes.

---

# 8. Como Executar

No terminal:

pip install -r requirements.txt
python main.py

