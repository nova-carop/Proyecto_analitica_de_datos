# %% Importar librerías
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from funpymodeling.exploratory import status
from ydata_profiling import ProfileReport
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %% Funciones de carga y análisis exploratorio
def cargar_csv(ruta='Tema_10.csv', na_values=['-', 'N/A', 'n/a']):
    df = pd.read_csv(ruta, na_values=na_values)
    print("Información general del DataFrame:")
    print(df.head())
    print(f"Forma del DataFrame: {df.shape}")
    print(status(df))

    profile = ProfileReport(df, title="Reporte_Tema10", explorative=True)
    profile.to_file(output_file="Profiling.html")
    return df

# %% Funciones de limpieza de datos
def limpiar_duplicados(df):
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def limpiar_fecha(df):
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    df['date'] = df['date'].interpolate()

    def adjust_date(row):
        if pd.notna(row['day']) and pd.notna(row['date']):
            interp_day = row['date'].day_name()
            if interp_day != row['day']:
                target_day = row['day']
                adjusted_date = row['date'] - pd.Timedelta(days=1)
                while adjusted_date.day_name() != target_day:
                    adjusted_date += pd.Timedelta(days=1)
                return adjusted_date
        return row['date']

    df['date'] = df.apply(adjust_date, axis=1)
    return df

def limpiar_dia(df):
    def assign_day(row):
        if pd.isna(row['day']) and not pd.isna(row['date']):
            return row['date'].day_name()
        return row['day']
    df['day'] = df.apply(assign_day, axis=1)
    return df

def limpiar_department(df):
    df['department'] = df['department'].str.strip()
    return df

def limpiar_team(df):
    df['team'] = pd.to_numeric(df['team'], errors='coerce').astype('Int64')
    return df

def limpiar_campos_numericos(df):
    campos_con_rango = {
        'wip': (0, 5000),
        'over_time': (0, 720),
        'incentive': (0, 500),
        'idle_time': (0, 60),
        'idle_men': (0, 10),
        'actual_productivity': (0, 1)
    }
    for campo, (min_val, max_val) in campos_con_rango.items():
        df[campo] = df[campo].fillna(df[campo].median()).apply(lambda x: min(max(x, min_val), max_val))
    return df

#Visualizacion de datos
def visualizacion(x,y,df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x,y,df)
    plt.title(f'Distribucion de variables{x,y}')
    plt.xlabel(f'Variable{x}')
    plt.ylabel(f'Variable{y}')
    return plt.show()

# %% Funciones de modelado
def preparar_datos_modelo(df):
    df = pd.get_dummies(df, columns=['day', 'department'])
    df.drop(['date','quarter'], axis=1, inplace=True)
    y = df['actual_productivity']
    X = df.drop('actual_productivity', axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def normalizar_datos(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def entrenar_modelo(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

def entrenar_y_evaluar_modelo(X_train, X_val, y_train, y_val):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train)
    y_val_pred = modelo.predict(X_val)

    print("Evaluación del modelo:")
    print('Error cuadrático medio (train): ', mean_squared_error(y_train, y_train_pred))
    print('Error cuadrático medio (val): ', mean_squared_error(y_val, y_val_pred))
    print('R2 (train): ', r2_score(y_train, y_train_pred))
    print('R2 (val): ', r2_score(y_val, y_val_pred))
    print('Error absoluto medio (train): ', np.mean(np.abs(y_train - y_train_pred)))
    print('Error absoluto medio (val): ', np.mean(np.abs(y_val - y_val_pred)))
    print('RMSE (train): ', np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print('RMSE (val): ', np.sqrt(mean_squared_error(y_val, y_val_pred)))
    print('MAPE (train): ', np.mean(np.abs(y_train - y_train_pred) / y_train))
    print('MAPE (val): ', np.mean(np.abs(y_val - y_val_pred) / y_val))


# %% Ejecución del flujo de limpieza y modelado
def main():
    df = cargar_csv()
    print("\nVisualizacion exploratoria de los datos")
    df = visualizacion()

    df = limpiar_duplicados(df)
    df = limpiar_fecha(df)
    df = limpiar_dia(df)
    df = limpiar_department(df)
    df = limpiar_team(df)
    df = limpiar_campos_numericos(df)

    print("\nDataFrame después de la limpieza:")
    print(df.head())
    
    print("\nVisualizacion final:")
    df = visualizacion()
    
    #Modelo
    df.isnull().sum()
    X_train, X_test, y_train, y_test = preparar_datos_modelo(df)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train, X_val, X_test = normalizar_datos(X_train, X_val, X_test)

    entrenar_y_evaluar_modelo(X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()



