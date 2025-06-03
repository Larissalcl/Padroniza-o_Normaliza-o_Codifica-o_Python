import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
from scipy import stats

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('clientes-v2-tratados.csv')
print(df.head())

# Selecionar quais colunas serão normalizadas ou padronizadas
df = df[['idade', 'salario']]
print(df.head())

# Normalização - MinMaxScaler (ajuste dos valores no intervalo entre 0 e 1)
scaler = MinMaxScaler() #Escala os dados para um intervalo entre 0 e 1.
df['idade_MinMaxScaler'] = scaler.fit_transform(df[['idade']]) #Ajustar e transformar os dados da coluna idade
df['salario_MinMaxScaler'] = scaler.fit_transform(df[['salario']]) #Ajustar e transformar os dados da coluna salário
print('\n Normalização MinMaxScaler: \n', df.head())

# MinMaxScaler - Mudar o Padrão (entre -1 e 1)
min_max_scaler = MinMaxScaler(feature_range=(-1,1))

# Padronização - StandardScaler - (Dimensiona os dados de forma a média 0 e desvio padrão 1)

scaler = StandardScaler()
df['idade_StandardScaler'] = scaler.fit_transform(df[['idade']]) # Ajustar e transformar os dados
df['salario_StandardScaler'] = scaler.fit_transform(df[['salario']])

print('\n Padronização StandardScaler: \n', df.head())

# Padronização - RobustScaler - (Dimensiona os dados usando a mediana e IQR (ignora os outliers))
scaler = RobustScaler()
df['idade_RobustScaler'] = scaler.fit_transform(df[['idade']])
df['salario_RobustScaler'] = scaler.fit_transform(df[['salario']])

print('\n Padronização RobustScaler: \n', df.head())

#Transformação Box-Cox
df['Salario_boxcox'], _ = stats.boxcox(df['salario']+1)

print("\n Dataframe após tranformação Box-Cox em 'salario':\n", df.head())

#Transformação Logarítmica
df['salario_log'] = np.log1p(df['salario']) #log1p é usado para evitar problemas com valores zero

print("\n Dataframe após tranformação logarítmica em 'salario':\n", df.head())

# Selecionar quais colunas serão codificadas
df_2 = pd.read_csv('clientes-v2-tratados.csv')
df_2 = df_2[['estado_civil', 'nivel_educacao', 'area_atuacao', 'estado']]
print(df_2.head())

#Codificação one-hot encoding

## pd.get_dummies - Converte a coluna categórica em colunas binárias (cada categoria gera uma coluna)
## pd.concat - Junta  o DataFrame original df com as novas colunas codificadas.
## axis=1 indica que a junção é feita coluna a coluna (horizontalmente).

df_2 = pd.concat([df_2, pd.get_dummies(df_2['estado_civil'], prefix='estado_civil')], axis=1)
print("\n Dataframe após codificação one-hot para 'estado_civil': \n", df_2.head())

#Codificação ordinal para 'nível_educação'
educacao_ordem = {'Ensino Fundamental':1, 'Ensino Médio':2, 'Ensino Superior':3, 'Pós-graduação':4}
df_2['Nível_educacao_ordinal'] = df_2['nivel_educacao'].map(educacao_ordem)
print("\n Dataframe após codificação ordinal para 'nivel_educacao': \n", df_2.head())

#Codificação com o ".cat.codes"

df_2['area_atuacao_cod'] = df_2['area_atuacao'].astype('category').cat.codes

print("\n Dataframe após transformar 'area_atuacao' em códigos númericos: \n", df_2.head())

#Codificação - LabelEnconder

Enconder = LabelEncoder()
df_2['estado_cod'] = Enconder.fit_transform(df_2['estado'])

print("\n Dataframe após aplicar LabelEncoder em  'estados': \n", df_2.head())

#Codificação de Frequência

estado_freq = df_2['estado'].value_counts() / len(df_2)
df_2['estado_freq'] = df_2['estado'].map(estado_freq)*100

print("\n Dataframe após codificação de frequência em 'estado':\n", df_2.head())

