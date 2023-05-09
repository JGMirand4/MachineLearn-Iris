
# Importando as bibliotecas necessárias
# Base de dados
from sklearn.datasets import load_iris
# Biblioteca que separa os conjuntos de dados em teste e treino
from sklearn.model_selection import train_test_split
# Biblioteca do algorítomo KNN
from sklearn.neighbors import KNeighborsClassifier
# Biblioteca que calcula a precisao 
from sklearn.metrics import accuracy_score


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

# Pedindo ao usuário as características da flor que ele quer testar
comprimento_sep = float(input("Comprimento da sépala (em cm): "))
largura_sep = float(input("Largura da sépala (em cm): "))
comprimento_pet = float(input("Comprimento da pétala (em cm): "))
largura_pet = float(input("Largura da pétala (em cm): "))

# Criando uma nova flor de íris com os valores fornecidos pelo usuário
import numpy as np
nova_iris = np.array([[comprimento_sep, largura_sep, comprimento_pet, largura_pet]])

# Fazendo a previsão para a nova íris e exibindo a classe prevista
prediction = knn.predict(nova_iris)
print("\nA Ìris é do tipo:", dados_iris.target_names[prediction])
