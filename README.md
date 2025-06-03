# 🧪 Pré-processamento de Dados: Normalização, Padronização e Codificação
Este repositório apresenta um roteiro prático para realizar normalizações, padronizações e codificações de variáveis em um conjunto de dados utilizando pandas e técnicas da biblioteca sklearn. O objetivo é preparar os dados para aplicações de machine learning ou análise estatística.

## 💡 Objetivo
O script é útil para entender e aplicar diferentes técnicas de transformação e codificação de dados, servindo como base para pipelines de data science e machine learning.

## 📂 Estrutura do Projeto
O código está organizado em duas partes principais:

1. Transformações em Variáveis Numéricas
   
    - Normalização com MinMaxScaler: Escala os dados entre 0 e 1.

    - Padronização com StandardScaler: Centraliza os dados em média 0 e desvio padrão 1.

    - Padronização com RobustScaler: Utiliza a mediana e o IQR, sendo resistente a outliers.

  Transformações estatísticas:

    - Box-Cox: Aplica transformação estatística para normalizar a distribuição.

    - Logarítmica: Reduz a assimetria em distribuições enviesadas.

2. Codificação de Variáveis Categóricas
    - One-Hot Encoding: Gera colunas binárias para categorias (ex: estado_civil).

    - Codificação Ordinal: Define ordem explícita entre categorias (ex: nível_educacao).

    - Codificação com .cat.codes: Atribui um código numérico a cada categoria (ex: area_atuacao).

    - LabelEncoder: Transforma rótulos em valores numéricos (ex: estado).

    - Codificação de Frequência: Representa a frequência de ocorrência de uma categoria.

## 🛠️ Bibliotecas Utilizadas
- pandas

- numpy

- scikit-learn

- scipy.stats (para Box-Cox)

