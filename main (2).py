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

dados_iris = load_iris()

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
  dados_iris.data, dados_iris.target, test_size=0.3, random_state=42)

#Inicializa os classificadores
arvore_decisao = DecisionTreeClassifier() # Árvore de decisão
knn = KNeighborsClassifier() # K-Nearest Neighbors
regressao_logistica = LogisticRegression() # Regressão logística
naive_bayes = GaussianNB() # Naive Bayes
svm = SVC() # Máquinas de Vetores de Suporte

#Algoritmo árvore de decisao
arvore_decisao.fit(x_treinamento, y_treinamento)
y_previsao_arvore_decisao = arvore_decisao.predict(x_teste)
precisao_arvore_decisao = accuracy_score(y_teste, y_previsao_arvore_decisao)
print(f"Precisão da Árvore de decisão: {precisao_arvore_decisao:.2f}")

#Algoritmos de vizinhos mais próximos
knn.fit(x_treinamento, y_treinamento)
y_previsao_knn = knn.predict(x_teste)
precisao_knn = accuracy_score(y_teste, y_previsao_knn)
print(f"Precisão do KNN: {precisao_knn:.2f}")

#algoritmo de regressão logística
regressao_logistica.fit(x_treinamento, y_treinamento)
y_previsao_regressao_logistica = regressao_logistica.predict(x_teste)
precisao_regressao_logistica = accuracy_score(y_teste, y_previsao_regressao_logistica)
print(f"Precisão da Regressão logística: {precisao_regressao_logistica:.2f}")

#Algoritmo Naive bayes
naive_bayes.fit(x_treinamento, y_treinamento)
y_previsao_naive_bayes = naive_bayes.predict(x_teste)
precisao_naive_bayes = accuracy_score(y_teste, y_previsao_naive_bayes)
print(f"Precisão do Naive Bayes: {precisao_naive_bayes:.2f}")

#Algoritmo Máquina de vetores de suporte
svm.fit(x_treinamento, y_treinamento)
y_previsao_svm = svm.predict(x_teste)
precisao_svm = accuracy_score(y_teste, y_previsao_svm)
print(f"Precisão do SVM: {precisao_svm:.2f}")

# Grafico barrar horizontais

#Criando listas para serem usadas como os eixos do gráficos 
algoritmos = ['Árvore de decisão', 'KNN', 'Regressão logística', 'Naive Bayes', 'SVM']
precisoes = [precisao_arvore_decisao, precisao_knn, precisao_regressao_logistica, precisao_naive_bayes, precisao_svm]

#função da biblioteca do matplotlib que cria o gráfico de barras
plt.barh(algoritmos, precisoes)

# Adiciona as etiquetas do gráfico
plt.title('Comparação de precisão dos algoritmos')
plt.xlabel('Precisão')
plt.ylabel('Algoritmos')

# Exibir o Gráfico
plt.show()