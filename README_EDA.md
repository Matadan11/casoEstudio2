# 📊 Guía de Uso: AnalisisDatosExploratorio Mejorado

## 🚀 Inicio Rápido

```python
from GuiaEda import AnalisisDatosExploratorio

# Crear instancia (detecta automáticamente el separador)
eda = AnalisisDatosExploratorio('diabetes_V2.csv')

# Mostrar información básica
eda.info_basica()

# Obtener estadísticas descriptivas
stats = eda.estadisticas_descriptivas()

# Ver gráficos
eda.visualizar_boxplot()
eda.visualizar_histograma()
eda.visualizar_densidad()
eda.visualizar_correlacion()
```

---

## 📋 Métodos Disponibles

### 🔍 Información y Estadísticas

| Método | Descripción | Retorna |
|--------|------------|---------|
| `info_basica()` | Muestra dimensiones, tipos de datos, valores faltantes | None |
| `estadisticas_descriptivas()` | Calcula media, mediana, std, varianza, asimetría, curtosis | dict |
| `correlaciones()` | Matriz de correlación entre variables numéricas | pd.DataFrame |

### 📈 Visualizaciones

| Método | Descripción | Retorna |
|--------|------------|---------|
| `visualizar_boxplot(figsize, mostrar)` | Gráfico de caja para detectar outliers | (fig, ax) |
| `visualizar_histograma(figsize, bins, mostrar)` | Distribución de frecuencias | (fig, ax) |
| `visualizar_densidad(figsize, mostrar)` | Función de densidad de probabilidad | (fig, ax) |
| `visualizar_correlacion(figsize, mostrar)` | Mapa de calor de correlaciones | (fig, ax) |

### 🔧 Procesamiento de Datos

| Método | Descripción | Retorna |
|--------|------------|---------|
| `filtrar_numerico()` | Mantiene solo columnas numéricas | self |
| `one_hot_encoding()` | Convierte categóricas a numéricas | self |
| `analisis_completo()` | Ejecuta análisis y visualizaciones completas | None |

### 📦 Acceso a Datos

| Propiedad | Descripción |
|-----------|------------|
| `eda.df` | Accede/modifica el DataFrame actual |

---

## 💡 Ejemplos Prácticos

### Ejemplo 1: Análisis Rápido Completo

```python
eda = AnalisisDatosExploratorio('diabetes_V2.csv')
eda.analisis_completo()  # Todo en una línea
```

### Ejemplo 2: Análisis Selectivo

```python
eda = AnalisisDatosExploratorio('potabilidad_V2.csv')

# Solo información básica
eda.info_basica()

# Solo gráficos específicos
eda.visualizar_boxplot(figsize=(16, 8))
eda.visualizar_correlacion(figsize=(12, 10))
```

### Ejemplo 3: Method Chaining (Encadenamiento)

```python
eda = AnalisisDatosExploratorio('diabetes_V2.csv')

# Filtrar datos numéricos
eda.filtrar_numerico()
stats = eda.estadisticas_descriptivas()
```

### Ejemplo 4: Acceso Directo al DataFrame

```python
eda = AnalisisDatosExploratorio('diabetes_V2.csv')

# Acceder al DataFrame
df = eda.df

# Usar pandas normalmente
print(df.shape)
print(df.describe())
print(df.isnull().sum())
```

### Ejemplo 5: Extracción de Estadísticas para Procesamiento

```python
eda = AnalisisDatosExploratorio('diabetes_V2.csv')

# Obtener estadísticas como diccionario
stats = eda.estadisticas_descriptivas()

# Usar estadísticas en otro análisis
media = stats['media']
std = stats['desv_estandar']
asimetria = stats['asimetria']
```

---

## 🎯 Parámetros Importantes

### Constructor

```python
AnalisisDatosExploratorio(
    path,              # Ruta al archivo CSV
    sep=None,          # Separador (auto-detecta si es None)
    decimal=".",       # Símbolo decimal
    index_col=0        # Columna de índice
)
```

### Visualizadores

```python
# Todos los visualizadores soportan:
visualizar_boxplot(
    figsize=(15, 8),   # Tamaño de figura
    mostrar=True       # Mostrar gráfico (False para retornar solo)
)
```

---

## 🎓 Mejoras vs Versión Original

| Aspecto | Original | Mejorado |
|--------|----------|----------|
| **Separador CSV** | Manual (1 o 2) | Auto-detectado |
| **Documentación** | Sin docstrings | Docstrings completos |
| **Visualizaciones** | Solo mostrar | Retornan (fig, ax) |
| **Encadenamiento** | No | Sí (con `self`) |
| **Errores** | No manejados | Try-except con mensajes claros |
| **Gráficos** | Privados | Públicos y retornables |
| **Estadísticas** | Solo impresión | Retornan diccionario |
| **Type hints** | No | Implícitos en docstrings |

---

## 📝 Notas Importantes

1. **Jupyter Notebooks**: Los gráficos se muestran automáticamente
2. **Uso en scripts**: Pasar `mostrar=False` para retornar figuras
3. **Método chaining**: Funciona con `filtrar_numerico()` y `one_hot_encoding()`
4. **Property `df`**: Es modificable, puedes actualizar el DataFrame

---

## 🚨 Manejo de Errores

```python
try:
    eda = AnalisisDatosExploratorio('archivo_inexistente.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")

try:
    eda.df = "no es un dataframe"
except TypeError as e:
    print(f"Error: {e}")
```

---

## 🎨 Personalización de Gráficos

```python
# Cambiar tamaño de figura
fig, ax = eda.visualizar_boxplot(figsize=(20, 10))

# Cambiar número de bins en histograma
fig, ax = eda.visualizar_histograma(bins=30, figsize=(14, 8))

# Retornar figura sin mostrar (para guardar después)
fig, ax = eda.visualizar_correlacion(mostrar=False)
fig.savefig('correlacion.png', dpi=300, bbox_inches='tight')
```

---

**¡Listo para usar! 🎉**
