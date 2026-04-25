

# =============================================================================
# CLASE PARA ANÁLISIS DE REGRESIÓN
# =============================================================================

class Regresion:
    """
    Clase para análisis de modelos de regresión con diferentes algoritmos
    """
    
    def __init__(self, df, target_col='target'):
        """
        Inicializa la clase con un DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con los datos
        target_col : str
            Nombre de la columna objetivo
        """
        self.__df = df
        self.__target_col = target_col
        self.__scaler = StandardScaler()
    
    @property
    def df(self):
        return self.__df
    
    @df.setter
    def df(self, valor):
        if not isinstance(valor, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame")
        self.__df = valor
    
    def __preparar_datos(self, test_size=0.25, random_state=42):
        """
        Prepara los datos: separación y escalado
        """
        X = self.__df.drop(columns=[self.__target_col])
        X = pd.DataFrame(self.__scaler.fit_transform(X), columns=X.columns)
        y = self.__df[self.__target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def __evaluar_modelo(self, model, X_test, y_test, nombre_modelo=""):
        """
        Evalúa un modelo de regresión
        """
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"Modelo: {nombre_modelo}")
        print(f"{'='*60}")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")
        print(f"{'='*60}")
        
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "y_pred": y_pred}
    
    def LinearRegression_simple(self):
        """
        Regresión Lineal Simple (una variable predictiva)
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, "Linear Regression")
        
        print(f"Coeficientes: {model.coef_}")
        print(f"Intercepto: {model.intercept_}")
        
        return model, resultados
    
    def LinearRegression_multiple(self):
        """
        Regresión Lineal Múltiple
        """
        return self.LinearRegression_simple()
    
    def Lasso_simple(self, alpha=1.0):
        """
        Lasso Regression
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, f"Lasso (alpha={alpha})")
        
        print(f"Coeficientes no-nulos: {np.sum(model.coef_ != 0)}/{len(model.coef_)}")
        
        return model, resultados
    
    def LassoCV(self, cv=5):
        """
        Lasso con validación cruzada para encontrar alpha óptimo
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = LassoCV(cv=cv, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, f"LassoCV (alpha_óptimo={model.alpha_:.6f})")
        
        print(f"Alpha óptimo: {model.alpha_:.6f}")
        print(f"Coeficientes no-nulos: {np.sum(model.coef_ != 0)}/{len(model.coef_)}")
        
        return model, resultados
    
    def Ridge_simple(self, alpha=1.0):
        """
        Ridge Regression
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, f"Ridge (alpha={alpha})")
        
        return model, resultados
    
    def RidgeCV(self, cv=5):
        """
        Ridge con validación cruzada para encontrar alpha óptimo
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        alphas = np.logspace(-2, 10, 13)
        model = RidgeCV(alphas=alphas, cv=cv)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, f"RidgeCV (alpha_óptimo={model.alpha_:.6f})")
        
        print(f"Alpha óptimo: {model.alpha_:.6f}")
        
        return model, resultados
    
    def SVR_simple(self, kernel='rbf', C=1.0):
        """
        Support Vector Regression
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = SVR(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, f"SVR (kernel={kernel}, C={C})")
        
        return model, resultados
    
    def DecisionTreeRegressor_simple(self, max_depth=5, min_samples_split=2):
        """
        Árbol de Decisión para Regresión
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, 
                                          f"DecisionTreeRegressor (max_depth={max_depth})")
        
        return model, resultados
    
    def RandomForestRegressor_simple(self, n_estimators=100, max_depth=10, min_samples_split=2):
        """
        Random Forest para Regresión
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                      min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        
        resultados = self.__evaluar_modelo(model, X_test, y_test, 
                                          f"RandomForestRegressor (n_est={n_estimators})")
        
        # Feature importance
        importancias = pd.Series(model.feature_importances_, index=X_train.columns)
        print(f"\nTop 5 Features importantes:")
        print(importancias.nlargest(5))
        
        return model, resultados
    
    def benchmarking(self):
        """
        Realiza un benchmarking entre todos los algoritmos
        """
        print("\n" + "="*80)
        print("BENCHMARKING DE ALGORITMOS DE REGRESIÓN")
        print("="*80)
        
        resultados_all = {}
        
        # Algoritmos
        algoritmos = [
            ("Linear Regression", self.LinearRegression_simple, {}),
            ("Lasso", self.Lasso_simple, {"alpha": 0.1}),
            ("LassoCV", self.LassoCV, {"cv": 5}),
            ("Ridge", self.Ridge_simple, {"alpha": 1.0}),
            ("RidgeCV", self.RidgeCV, {"cv": 5}),
            ("SVR", self.SVR_simple, {"kernel": "rbf", "C": 100}),
            ("DecisionTreeRegressor", self.DecisionTreeRegressor_simple, {"max_depth": 10}),
            ("RandomForestRegressor", self.RandomForestRegressor_simple, {"n_estimators": 100}),
        ]
        
        for nombre, metodo, params in algoritmos:
            try:
                modelo, resultados = metodo(**params)
                resultados_all[nombre] = resultados
            except Exception as e:
                print(f"Error en {nombre}: {str(e)}")
        
        # Tabla comparativa
        df_resultados = pd.DataFrame({
            alg: {"RMSE": res["RMSE"], "MAE": res["MAE"], "R2": res["R2"]}
            for alg, res in resultados_all.items()
        }).T
        
        print("\n" + "="*80)
        print("TABLA COMPARATIVA")
        print("="*80)
        print(df_resultados.to_string())
        print("="*80)
        
        return df_resultados, resultados_all
    
    def grid_search_model(self, modelo_tipo="Ridge", param_grid=None):
        """
        Realiza Grid Search para optimizar hiperparámetros
        
        Parameters:
        -----------
        modelo_tipo : str
            Tipo de modelo: 'Ridge', 'Lasso', 'SVR', 'DecisionTree', 'RandomForest'
        param_grid : dict
            Diccionario de hiperparámetros para buscar
        """
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        
        # Modelos disponibles
        modelos = {
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'SVR': SVR(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(),
        }
        
        # Parámetros por defecto si no se especifican
        param_grids_default = {
            'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
            'Lasso': {'alpha': [0.01, 0.1, 1, 10]},
            'SVR': {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']},
            'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
            'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
        }
        
        if param_grid is None:
            param_grid = param_grids_default.get(modelo_tipo, {})
        
        modelo = modelos.get(modelo_tipo)
        
        if modelo is None:
            raise ValueError(f"Modelo {modelo_tipo} no disponible")
        
        print(f"\nGrid Search: {modelo_tipo}")
        print(f"Parámetros: {param_grid}")
        
        grid = GridSearchCV(modelo, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print(f"Mejores parámetros: {grid.best_params_}")
        print(f"Mejor score CV: {grid.best_score_:.6f}")
        
        resultados = self.__evaluar_modelo(grid.best_estimator_, X_test, y_test, 
                                          f"{modelo_tipo} (GridSearch óptimo)")
        
        return grid.best_estimator_, resultados, grid.best_params_
