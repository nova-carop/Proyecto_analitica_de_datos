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

# Funciones de carga y análisis exploratorio
def cargar_csv(ruta='Tema_10.csv', na_values=['-', 'N/A', 'n/a']):
    df = pd.read_csv(ruta, na_values=na_values)
    print("Información general del DataFrame:")
    print(df.head())
    print(f"Forma del DataFrame: {df.shape}")
    print(status(df))

    profile = ProfileReport(df, title="Reporte_Tema10", explorative=True)
    profile.to_file(output_file="Profiling.html")
    return df

# Funciones de limpieza de datos
def limpiar_duplicados(df):
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def limpiar_fecha(df):
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    df['date'] = df['date'].interpolate()

    def ajustar_fecha(row):
        if pd.notna(row['day']) and pd.notna(row['date']):
            interp_day = row['date'].day_name()
            if interp_day != row['day']:
                target_day = row['day']
                adjusted_date = row['date'] - pd.Timedelta(days=1)
                while adjusted_date.day_name() != target_day:
                    adjusted_date += pd.Timedelta(days=1)
                return adjusted_date
        return row['date']

    df['date'] = df.apply(ajustar_fecha, axis=1)
    # crear la columna semana (equivalente a quarter) segun el dato en 'date'
    df['semana'] = df['date'].apply(lambda x:4 if x.day > 28 else (x.day - 1) // 7 +1)
    return df

def limpiar_dia(df):
    def asignar_dia(row):
        if pd.isna(row['day']) and not pd.isna(row['date']):
            return row['date'].day_name()
        return row['day']
    df['day'] = df.apply(asignar_dia, axis=1)
    return df


def limpiar_team_limpiar_department(df):
    df['team'] = pd.to_numeric(df['team'], errors='coerce').astype('Int64')
    df['department'] = df['department'].str.strip()
    df.sort_values(by=['date', 'team', 'department']).reset_index(drop=True)
    df['department'] = df['department'].ffill().bfill()
    df['team'] = df['team'].ffill().bfill()
    return df


def limpiar_campos_numericos(df):
    # Campos con sus rangos establecidos
    campos_con_rango = {
        'wip': (0, 5000),
        'over_time': (0, 20000),
        'incentive': (0, 500),
        'idle_time': (0, 60),
        'actual_productivity': (0, 1),
        'targeted_productivity': (0.3, 1),  
        'smv': (0, 100),
        'no_of_workers': (0, 60)
    }
    for campo, (min_val, max_val) in campos_con_rango.items():
        #Se les hace la mediana a todos los campos
        df[campo] = df[campo].fillna(df[campo].median()).apply(lambda x: min(max(x, min_val), max_val))

    # Ubicadas por fuera porque no necesitan ser limitadas con un rango
    df['no_of_style_change'] = df['no_of_style_change'].fillna(df['no_of_style_change'].median())
    df['idle_men'] = df['idle_men'].fillna(df['idle_men'].median())
    return df


# Visualización de datos
def visualizacion(x, y, df, etapa):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(f'Distribución de variables {x}, {y} - {etapa}')
    plt.xlabel(f'Variable {x}')
    plt.ylabel(f'Variable {y}')
    plt.savefig(f'grafico_{y}_{etapa}.png')  
    plt.close()

# Funciones de modelado
def preparar_datos_modelo(df):
    y = df['actual_productivity']
    X = df.drop('actual_productivity', axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def normalizar_datos(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def entrenar_y_evaluar_modelo(X_train, X_val, y_train, y_val, X_test, y_test):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train)
    y_val_pred = modelo.predict(X_val)
    y_test_pred = modelo.predict(X_test)

    print("Evaluación del modelo:")
    # Error absoluto medio (MAE)
    print('Error absoluto medio entrenamiento: ', np.mean(np.abs(y_train - y_train_pred)))
    print('Error absoluto medio validación: ', np.mean(np.abs(y_val - y_val_pred)))
    print('Error absoluto medio test: ', np.mean(np.abs(y_test - y_test_pred)))
    # Error (RMSE)
    print('Error RMSE entrenamiento: ', np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print('Error RMSE validación: ', np.sqrt(mean_squared_error(y_val, y_val_pred)))
    print('Error RMSE test: ', np.sqrt(mean_squared_error(y_test, y_test_pred)))
    # Error absoluto medio porcentual
    print('Error absoluto medio porcentual entrenamiento: ', np.mean(np.abs(y_train - y_train_pred)/y_train))
    print('Error absoluto medio porcentual validación: ', np.mean(np.abs(y_val - y_val_pred)/y_val))
    print('Error absoluto medio porcentual test: ', np.mean(np.abs(y_test - y_test_pred)/y_test))


def main():
    df = cargar_csv()
    # Visualización antes de la limpieza
    for col in ['actual_productivity', 'targeted_productivity', 'smv', 'wip', 'over_time',
                'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers']:
        visualizacion('team', col, df, etapa='antes de la limpieza')

    #Limpieza de campos
    df = limpiar_duplicados(df)
    df = limpiar_fecha(df)
    df = limpiar_dia(df)
    df = limpiar_team_limpiar_department(df)
    df = limpiar_campos_numericos(df)

    #Visualización después de la limpieza
    for col in ['actual_productivity', 'targeted_productivity', 'smv', 'wip', 'over_time',
            'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers']:
        visualizacion('team', col, df, etapa='despues de la limpieza')

    #Aplicacion de la codificación one-hot a 'day' y 'department' inmediatamente después de limpiarlos
    df = pd.get_dummies(df, columns=['day', 'department'])
    # Eliminar las columnas 'date' y 'quarter' que ya no se necesitan
    df.drop(['date', 'quarter'], axis=1, inplace=True)

    print("\nDataFrame después de la limpieza:")
    print(df.head())
    
    # Modelo
    print(df.isnull().sum())
    X_train, X_test, y_train, y_test = preparar_datos_modelo(df)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train, X_val, X_test = normalizar_datos(X_train, X_val, X_test)
    entrenar_y_evaluar_modelo(X_train, X_val, y_train, y_val, X_test, y_test)

if __name__ == "__main__":
    main()
