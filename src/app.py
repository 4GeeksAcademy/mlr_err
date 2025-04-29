from utils import db_connect
engine = db_connect()

# your code here
# Paso 1: Carga del conjunto de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv"
df = pd.read_csv(url)

df.rename(columns={"19-Oct": "9-10"}, inplace=True)

print(df.head())
df.info()

# Paso 2: Realiza un EDA completo
df["Adult_population"] = df["20-29"] + df["30-39"] + df["40-49"] + df["50-59"] + df["60-69"] + df["70-79"] + df["80+"]

print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["Adult_population"], kde=True)
plt.title("Distribución de la Población Adulta por Condado")
plt.show()

print("Valores nulos por columna:")
print(df.isnull().sum())

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Adult_population"])
y = df["Adult_population"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")

# Paso 3: Construcción del modelo de regresión
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Modelo Lineal - MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

alphas = np.linspace(0, 20, 20)
r2_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred_lasso))

plt.figure(figsize=(8,5))
plt.plot(alphas, r2_scores, marker='o', linestyle='-')
plt.xlabel("Valor de alfa")
plt.ylabel("R²")
plt.title("Evolución del R² con Lasso")
plt.show()

# Paso 4: Optimización del modelo de regresión anterior
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_opt = Lasso(alpha=5.0)
lasso_opt.fit(X_train_scaled, y_train)

y_pred_lasso_opt = lasso_opt.predict(X_test_scaled)

mae_lasso_opt = mean_absolute_error(y_test, y_pred_lasso_opt)
mse_lasso_opt = mean_squared_error(y_test, y_pred_lasso_opt)
r2_lasso_opt = r2_score(y_test, y_pred_lasso_opt)

print(f"Modelo Lasso Optimizado - MAE: {mae_lasso_opt:.2f}, MSE: {mse_lasso_opt:.2f}, R²: {r2_lasso_opt:.2f}")