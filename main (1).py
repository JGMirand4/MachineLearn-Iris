
# Importando as bibliotecas necessárias
# Base de dados
from sklearn.datasets import load_iris
# Biblioteca que separa os conjuntos de dados em teste e treino
from sklearn.model_selection import train_test_split
# Biblioteca do algorítomo KNN
from sklearn.neighbors import KNeighborsClassifier
# Biblioteca que calcula a precisao e a matriz de confusao
from sklearn.metrics import accuracy_score, confusion_matrix
# Biblioteca que plota e exibe os gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados da iris
dados_iris = load_iris()

# Dividindo os dados em conjunto de treinamento e teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
  dados_iris.data, dados_iris.target, test_size=0.3, random_state=40)

# Treinando o modelo com o algoritmo K-NN, usando 3 vizinhos próximos
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_treinamento, y_treinamento)

# Previsão das possíveis classes das instâncias no conjunto de teste
y_previsao = knn.predict(x_teste)

# Verificando se as previsões do conjunto de teste bate com as classes das instâncias dele
precisao = accuracy_score(y_teste, y_previsao)

# Imprimindo a precisão do modelo
print("A precisão do algorítimo é de {:.2f}%\n".format(precisao * 100))

# Gerando a matriz de confusão
matriz_confusao = confusion_matrix(y_teste, y_previsao)
sns.heatmap(matriz_confusao, annot=True, cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Verdadeiros')

# Exibir matriz
plt.show()
