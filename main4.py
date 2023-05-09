import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Importando de cada um dos algorítimos
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Importando time para calcular o tempo de execução dos algorítimos
import time

dados_iris = load_iris()

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
  dados_iris.data, dados_iris.target, test_size=0.3, random_state=42)

#Inicializa os classificadores
arvore_decisao = DecisionTreeClassifier() # Árvore de decisão
knn = KNeighborsClassifier() # K-Nearest Neighbors
regressao_logistica = LogisticRegression() # Regressão logística
naive_bayes = GaussianNB() # Naive Bayes
svm = SVC() # Máquinas de Vetores de Suporte

#Algoritmo árvore de decisao/calcula o tempo de execução
tempo_inicial = time.time()
arvore_decisao.fit(x_treinamento, y_treinamento)
y_previsao_arvore_decisao = arvore_decisao.predict(x_teste)
precisao_arvore_decisao = accuracy_score(y_teste, y_previsao_arvore_decisao)
tempo_final = time.time()
tempo_arvore = tempo_final - tempo_inicial
print(f"Precisão da Árvore de decisão: {precisao_arvore_decisao:.2f}")

#Algoritmos de vizinhos mais próximos/calcula o tempo de execução
tempo_inicial = time.time()
knn.fit(x_treinamento, y_treinamento)
y_previsao_knn = knn.predict(x_teste)
precisao_knn = accuracy_score(y_teste, y_previsao_knn)
tempo_final = time.time()
tempo_knn = tempo_final - tempo_inicial
print(f"Precisão do KNN: {precisao_knn:.2f}")

#Algoritmo de regressão logística/calcula o tempo de execução
tempo_inicial = time.time()
regressao_logistica.fit(x_treinamento, y_treinamento)
y_previsao_regressao_logistica = regressao_logistica.predict(x_teste)
precisao_regressao_logistica = accuracy_score(y_teste, y_previsao_regressao_logistica)
tempo_final = time.time()
tempo_regressao = tempo_final - tempo_inicial
print(f"Precisão da Regressão logística: {precisao_regressao_logistica:.2f}")

#Algoritmo Naive bayes/calcula o tempo de execução
tempo_inicial = time.time()
naive_bayes.fit(x_treinamento, y_treinamento)
y_previsao_naive_bayes = naive_bayes.predict(x_teste)
precisao_naive_bayes = accuracy_score(y_teste, y_previsao_naive_bayes)
tempo_final = time.time()
tempo_naive = tempo_final - tempo_inicial
print(f"Precisão do Naive Bayes: {precisao_naive_bayes:.2f}")

#Algoritmo Máquina de vetores de suporte/calcula o tempo de execução
tempo_inicial = time.time()
svm.fit(x_treinamento, y_treinamento)
y_previsao_svm = svm.predict(x_teste)
precisao_svm = accuracy_score(y_teste, y_previsao_svm)
tempo_final = time.time()
tempo_svm = tempo_final - tempo_inicial
print(f"Precisão do SVM: {precisao_svm:.2f}")

# Gráfico de barras com o tempo de execução de cada algoritmo

#Criando listas para serem usadas como os eixos do gráficos 
plt.bar(['Árvore de Decisão', 'SVM', 'KNN', 'Regressão Logística', 'Naive Bayes'], [tempo_arvore, tempo_svm, tempo_knn, tempo_regressao, tempo_naive])
plt.ylabel('Tempo de Execução (s)')
#Adiciona as etiquetas do gráfico
plt.title('Comparação de tempo de execução dos algoritmos')
plt.xlabel('Algoritmos')
#Exibe o gráfico
plt.show()
