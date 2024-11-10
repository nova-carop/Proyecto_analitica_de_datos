# %% importar librerias
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from funpymodeling.exploratory import status
from ydata_profiling import ProfileReport
import numpy as np                           

# %% cargar el dataframe con la info del csv de manera generica
df = pd.read_csv('Tema_10.csv')

# %% [[[[ANALISIS EXPLORATORIO DEL DATAFRAME]]]]
df.head()
df.shape
status(df)
# ejecutar el Profile Report
profile = ProfileReport(df, title="Reporte_Tema10", explorative=True)
profile.to_file(output_file="Profiling.html")



# %% [[[[VISUALIZACIONES]]]]



# %% [[[[PRETRATAMIENTO DE DATOS]]]]

# %% DUPLICADOS: eliminar filas duplicadas
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# %% VARIABLE DATE: 
# convertir a datetime (formato: MM-DD-YYYY)
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
# completar valores faltantes - realizar interpolación
df['date'] = df['date'].interpolate()
# funcion para ajustar la fecha interpolada en caso de que no coincida con el valor en "day" (nombre del dia)
def adjust_date(row):
    if pd.notna(row['day']) and pd.notna(row['date']):
        interp_day = row['date'].day_name()                        # obtener el dia de la semana de date (que contiene los datos interpolados)
        #si el dia de la semana no coincide ajustar la fecha en date, comenzando en el dia anterior
        if interp_day != row['day']:
            target_day = row['day']
            adjusted_date = row['date'] - pd.Timedelta(days=1)     #empezar 1 dia antes de la fecha
            while adjusted_date.day_name() != target_day:
                adjusted_date += pd.Timedelta(days=1)              #sumar de a 1 dia hasta que la fecha coincida con el valor de "day" (nombre del dia)
            return adjusted_date
    return row['date']
# aplicar la funcion adjust_date
df['date'] = df.apply(adjust_date, axis=1)

# %% VARIABLE DAY: 
# funcion para rellenar los valores NaN, en base al valor de date
def assign_day(row):
    if pd.isna(row['day']):              
        return row['date'].day_name()
    return row['day']
# aplicar la funcion assign_day en la columna day para cada fila
df['day'] = df.apply(assign_day, axis=1)

# %% VARIABLE QUARTER: 
# funcion para calcular la semana del mes
def calcular_semana(fecha):
    return (fecha.day - 1) // 7 + 1
# crear la columna semana (equivalente a quarter) aplicando la funcion semana a cada fila segun el dato en 'date'
df['semana'] = df['date'].apply(calcular_semana)

# %% VARIABLES TEAM y DEPARTMENT: 
# en DEPARTMENT elimina los espacios vacios al principio y al final
df['department'] = df['department'].str.strip()
# en TEAM cambiar el tipo a "int" y asignar NaN a valores vacios o al valor "invalid_value" 
df['team'] = pd.to_numeric(df['team'], errors='coerce').astype('Int64')        
# ordenar el dataframe por fecha, team y department
df.sort_values(by=['date', 'team', 'department']).reset_index(drop=True)
# completar los valores faltantes en ambas variables usando ffill y bfill: completar los NaN con el valor mas cercano en cualquier direccion (hacia adelante o atras)
df['department'] = df['department'].ffill().bfill()
df['team'] = df['team'].ffill().bfill()


# %% VARIABLE TARGETED_PRODUCTIVITY
#Verificar escala de los valores targeted_productivity
df.plot(x='date', y=['targeted_productivity'])

df['targeted_productivity'] = df['targeted_productivity'].fillna(df['targeted_productivity'].median())  # Rellenar valores faltantes con la mediana
df['targeted_productivity'] = df['targeted_productivity'].apply(lambda x: min(0.3,max(x, 0))  # Limitar a un rango de minimo 0.3 según gráfico



# %% VARIABLE SMV
#Verificar escala de los valores SMV
df.plot(x="date", y=["smv"])

df['smv'] = df['smv'].fillna(df['smv'].median())  # Rellenar valores faltantes con la mediana
df['smv'] = df['smv'].apply(lambda x: min(max(x, 0), 100))  # Limitar a un rango máximo de 100, según gráfico


# %% VARIABLE NO_OF_STYLE_CHANGE
#Verificar escala de los valores no_of_style_change
df.plot(x='date', y=['no_of_style_change'])

df['no_of_style_change'] = df['no_of_style_change'].fillna(df['no_of_style_change'].median())  # Rellenar valores faltantes con la mediana

# %% VARIABLE NO_OF_WORKERS
#Verificar escala de los valores no_of_workers
df.plot(x='date', y=['no_of_workers'])

df['no_of_workers'] = df['no_of_workers'].fillna(df['no_of_workers'].median())  # Rellenar valores faltantes con la mediana
df['no_of_workers'] = df['no_of_workers'].apply(lambda x: min(max(x, 0), 60))  # Limitar a un rango máximo de 60, según gráfico 


# %% VARIABLE WIP
df['wip'] = df['wip'].fillna(df['wip'].median())  # Rellenar valores faltantes con la mediana
df['wip'] = df['wip'].apply(lambda x: min(max(x, 0), 5000))  # Limitar a un rango máximo de 5000 

# %% VARIABLE OVER_TIME
df['over_time'] = df['over_time'].fillna(0)
df['over_time'] = df['over_time'].apply(lambda x: min(max(x, 0), 720))  # Limitar a un máximo de 720 minutos

# %% VARIABLE INCENTIVE
df['incentive'] = df['incentive'].fillna(0)
df['incentive'] = df['incentive'].apply(lambda x: min(max(x, 0), 500))  # Ajuste a un máximo de 500 

# %% VARIABLE IDLE_TIME
df['idle_time'] = df['idle_time'].fillna(0)
df['idle_time'] = df['idle_time'].apply(lambda x: min(max(x, 0), 60))  # Limitar a un máximo de 60 minutos

# %% VARIABLE IDLE_MEN
df['idle_men'] = df['idle_men'].fillna(0).astype(int)
df['idle_men'] = df['idle_men'].apply(lambda x: min(max(x, 0), 10))  # Limitar a un máximo de 10 personas

# %% VARIABLE ACTUAL_PRODUCTIVITY
df['actual_productivity'] = df['actual_productivity'].fillna(0)  # Rellenar valores faltantes con 0
df['actual_productivity'] = df['actual_productivity'].apply(lambda x: min(max(x, 0), 1))  # Limitar al rango [0, 1]

#%% Verificar que no hay datos faltantes
df.isnull().sum()






# %% [[[[VISUALIZACIONES POST LIMPIEZA]]]]




# %% [[[[MODELO PREDICTIVO]]]]

# %% importar librerias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# %% aplicar codificación one-hot a las columnas categoricas 'day', 'department'
df = pd.get_dummies(df, columns=['day','department'])

# %% eliminar columnas innecesarias
df.drop(['date','quarter','day','department'], axis=1, inplace=True)

# %% normalizar datos
def normalizar_datos(data):
    '''Normalizar datos'''
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data
df = normalizar_datos(df)

# %% Dividir datos en entrenamiento, validación y prueba
y = df['actual_productivity']
X = df.drop('actual_productivity', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %% Entrenar el Modelo Regresión lineal
regresion_lineal = LinearRegression()
regresion_lineal.fit(X_train, y_train)

# %% Predicciones
y_train_pred = regresion_lineal.predict(X_train)
y_val_pred = regresion_lineal.predict(X_val)

# %% Calcular Metricas de Evaluacion
# Error cuadrático medio
print('Error cuadrático medio entrenamiento: ', mean_squared_error(y_train, y_train_pred))
print('Error cuadrático medio validación: ', mean_squared_error(y_val, y_val_pred))
# R2
print('R2 entrenamiento: ', r2_score(y_train, y_train_pred))
print('R2 validación: ', r2_score(y_val, y_val_pred))
# Error absoluto medio
print('Error absoluto medio entrenamiento: ', np.mean(np.abs(y_train - y_train_pred)))
print('Error absoluto medio validación: ', np.mean(np.abs(y_val - y_val_pred)))
# Error RMSE
print('Error RMSE entrenamiento: ', np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('Error RMSE validación: ', np.sqrt(mean_squared_error(y_val, y_val_pred)))
# Error absoluto medio porcentual
print('Error absoluto medio porcentual entrenamiento: ', np.mean(np.abs(y_train - y_train_pred)/y_train))
print('Error absoluto medio porcentual validación: ', np.mean(np.abs(y_val - y_val_pred)/y_val))


