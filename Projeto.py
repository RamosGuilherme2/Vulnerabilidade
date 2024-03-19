import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dicionário com dados (sem a coluna 'grau_vulnerabilidade')
data = {
    'idade': [65, 70, 80, 85, 90, 95, 100, 70, 75, 80, 85, 90, 95, 100, 70, 75, 80, 85, 90, 95],
    'doencas_cronicas': [2, 3, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    'vive_sozinho': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Convertendo o dicionário em um DataFrame
df = pd.DataFrame(data)

# Supondo que 'grau_vulnerabilidade' será a variável que queremos prever,
# vamos criar uma coluna fictícia para representar isso
# Por exemplo, podemos usar uma função para gerar esses dados (aqui usamos a média das outras colunas)
df['grau_vulnerabilidade'] = df.mean(axis=1)

# Separando os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df.drop('grau_vulnerabilidade', axis=1), df['grau_vulnerabilidade'], test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões com o conjunto de teste
predictions = model.predict(X_test)

# Definindo função para converter resultados de Predictions em porcentagem
predictions_percent = [f"{round(num, 2)}%" for num in predictions]

# Criando um DataFrame para exibir os resultados de forma organizada
results_df = pd.DataFrame({
    'Idosos|': range(1, len(predictions_percent) + 1),
    'Grau de Vulnerabilidade| (%)': predictions_percent
})

# exibição final *-*
print(results_df)