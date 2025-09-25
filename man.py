# ======================================================
# Atividade 2 - Compreensão Aprofundada dos Dados
# Dataset: boardgame-geek-dataset_organized.csv
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# -------------------------
# A. Estatística Descritiva e Exploração Inicial
# -------------------------

# Carregar dataset
df = pd.read_csv("/Users/joaopedrolimabarbosa/Downloads/boardgame-geek-dataset_organized.csv", sep=",", encoding="utf-8")

# Criar coluna "playing_time" como média entre min_playtime e max_playtime
df["playing_time"] = (df["min_playtime"] + df["max_playtime"]) / 2

# Mostrar primeiras linhas
print("Primeiras linhas do dataset:")
print(df.head())

# Info
print("\nInformações do DataFrame:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Perguntas
print("\nNota média dos jogos (avg_rating):", df["avg_rating"].mean())
print("Tempo médio (playing_time):", df["playing_time"].mean())
print("Desvio padrão (playing_time):", df["playing_time"].std())

print("\nValores nulos por coluna:")
print(df.isnull().sum())

# -------------------------
# B. Tratamento de Dados Ausentes e Outliers
# -------------------------

# Nulos em avg_rating
print("\nValores nulos em avg_rating:", df["avg_rating"].isnull().sum())

# Preencher com mediana
mediana = df["avg_rating"].median()
df["avg_rating"].fillna(mediana, inplace=True)

# Boxplot playing_time
plt.figure()
sns.boxplot(x=df["playing_time"])
plt.title("Boxplot de Playing Time")
plt.show()

# Calcular IQR
Q1 = df["playing_time"].quantile(0.25)
Q3 = df["playing_time"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df["playing_time"] < limite_inferior) | (df["playing_time"] > limite_superior)]
print("\nQuantidade de outliers em playing_time:", len(outliers))

# -------------------------
# C. Visualização e Transformação de Dados
# -------------------------

# Histograma avg_rating
plt.figure()
sns.histplot(df["avg_rating"], bins=30, kde=True)
plt.title("Distribuição de Avg Rating")
plt.show()

# Transformação log em playing_time
df["playing_time_log"] = np.log1p(df["playing_time"])

plt.figure()
sns.histplot(df["playing_time_log"], bins=30, kde=True, color="orange")
plt.title("Distribuição de Playing Time (log transformado)")
plt.show()

# -------------------------
# D. Análise da Relação entre Variáveis
# -------------------------

# Scatter plot min_players x max_players
plt.figure()
sns.scatterplot(x="min_players", y="max_players", data=df, alpha=0.5)
plt.title("Min Players x Max Players")
plt.show()

# Matriz de correlação
correlacao = df.corr(numeric_only=True)
print("\nMatriz de correlação:")
print(correlacao)

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap das Correlações")
plt.show()

# -------------------------
# E. Análise Temporal
# -------------------------

# Criar coluna década a partir de release_year
df["Decada"] = (df["release_year"] // 10) * 10

# Contagem por década
decadas = df["Decada"].value_counts().sort_index()
print("\nQuantidade de jogos lançados por década:")
print(decadas)

# Década com maior número de lançamentos
decada_max = decadas.idxmax()
quantidade_max = decadas.max()
print(f"\nDécada com maior número de lançamentos: {decada_max} ({quantidade_max} jogos)")
