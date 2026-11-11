"""Entrena los dos modelos elegidos y los guarda con joblib en modelos_guardados/.
Uso:  python entrenar_y_guardar.py
Hay que volver a correrlo si cambia scikit-learn (el archivo guardado depende de la versión)."""
from pathlib import Path
import pandas as pd
import modelos as m
import produccion as p

df = m.preparar_df(pd.read_csv(Path(__file__).with_name("dataset_con_titulos.csv")))
for construir in (p.construir_clasificacion, p.construir_regresion):
    art = construir(df)
    p.guardar(art)
    print(art["tarea"], art["nombre"], {k: round(v, 4) for k, v in art["metricas"].items()})
