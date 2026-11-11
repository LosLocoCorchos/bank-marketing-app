"""Grillas ampliadas: ¿mejora algo salir de los bordes de la grilla del app?
Uso: python grillas_ampliadas.py [dataset_con_titulos.csv]
Elige por validación cruzada (5 pliegues, los mismos del app) y recién después mira la prueba.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, mean_squared_error
import modelos as m

ruta = sys.argv[1] if len(sys.argv) > 1 else "dataset_con_titulos.csv"
df = m.preparar_df(pd.read_csv(ruta))

def correr(titulo, res, cand, grillas, tarea):
    print(f"\n== {titulo}")
    for n, grilla in grillas.items():
        pipe = cand[n][0]
        scoring = "roc_auc" if tarea == "c" else "neg_root_mean_squared_error"
        gs = GridSearchCV(pipe, grilla, cv=res["pliegues"], scoring=scoring, n_jobs=1).fit(res["X_ent"], res["y_ent"])
        actual = res["params"][n] if n in res["params"] else None
        if tarea == "c":
            pr = roc_auc_score(res["y_pru"], gs.predict_proba(res["X_pru"])[:, 1])
            ref = res["prueba"][n]["AUC"]; cvref = res["cv"][n]["auc"].mean()
            print(f"{n:22s} CV ampliada {gs.best_score_:.4f} (app {cvref:.4f}) | prueba ampliada {pr:.4f} (app {ref:.4f}) | mejor {gs.best_params_}")
        else:
            pr = float(np.sqrt(mean_squared_error(res["y_pru"], gs.predict(res["X_pru"]))))
            ref = res["prueba"][n]["RMSE"]; cvref = res["cv"][n]["RMSE"].mean()
            print(f"{n:22s} CV ampliada {-gs.best_score_:.4f} (app {cvref:.4f}) | prueba ampliada {pr:.4f} (app {ref:.4f}) | mejor {gs.best_params_}")

cats, nums = m.columnas_clasificacion()
rc = m.entrenar_clasificacion(df, cats, nums, ["Regresión logística", "Random Forest", "HistGradientBoosting"], ajustar=False)
correr("Clasificación (AUC)", rc, m._candidatos_clasificacion(cats, nums, 42), {
    "Regresión logística": {"logisticregression__C": [0.03, 0.1, 0.3]},
    "Random Forest": {"randomforestclassifier__min_samples_leaf": [1, 3, 5, 10]},
    "HistGradientBoosting": {"histgradientboostingclassifier__learning_rate": [0.03, 0.05],
                             "histgradientboostingclassifier__max_depth": [6, 9, None],
                             "histgradientboostingclassifier__min_samples_leaf": [20, 50]}}, "c")

cats, nums = m.columnas_regresion()
rr = m.entrenar_regresion(df, cats, nums, ["Ridge (lineal)", "Random Forest", "HistGradientBoosting"], ajustar=False)
correr("Predicción de campaign sin mes ni día (RMSE)", rr, m._candidatos_regresion(cats, nums, 42), {
    "Ridge (lineal)": {"ridge__alpha": [100, 300, 1000, 3000]},
    "Random Forest": {"randomforestregressor__min_samples_leaf": [50, 100, 200]},
    "HistGradientBoosting": {"histgradientboostingregressor__learning_rate": [0.03, 0.05],
                             "histgradientboostingregressor__max_depth": [3, 6, 9],
                             "histgradientboostingregressor__min_samples_leaf": [20, 100]}}, "r")
