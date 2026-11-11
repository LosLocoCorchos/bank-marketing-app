"""Modelos para producción: se entrenan una vez (entrenar_y_guardar.py), se serializan con joblib y la app los carga sin reentrenar.
No depende de Streamlit."""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import modelos as m

CARPETA = Path(__file__).with_name("modelos_guardados")
ARCHIVOS = {"clasificacion": CARPETA / "clasificacion.joblib", "regresion": CARPETA / "regresion.joblib"}


def _rangos(df, cats, nums):
    r = {}
    for c in cats:
        r[c] = {"tipo": "cat", "opciones": sorted(df[c].unique().tolist()), "defecto": df[c].mode().iloc[0]}
    for c in nums:
        r[c] = {"tipo": "int", "min": int(df[c].min()), "max": int(df[c].max()), "defecto": int(df[c].median())}
    return r


def construir_clasificacion(df):
    cats, nums = m.columnas_clasificacion()            # sin campaign ni duracion: escenario "rankear antes de llamar"
    res = m.entrenar_clasificacion(df, cats, nums, ["HistGradientBoosting"])
    n = "HistGradientBoosting"
    s = res["puntajes_pru"][n]
    y_ord = res["y_pru"].values[np.argsort(-s)]
    return {"tarea": "clasificacion", "nombre": n, "modelo": res["modelos"][n], "umbral": float(res["umbrales"][n]),
            "cats": cats, "nums": nums, "rangos": _rangos(df, cats, nums),
            "puntajes_prueba": np.sort(s).astype("float32"),
            "tasas_decil": [float(d.mean()) for d in np.array_split(y_ord, 10)],
            "tasa_base": float(res["y_pru"].mean()),
            "metricas": {"auc_cv": float(res["cv"][n]["auc"].mean()), "auc_cv_sd": float(res["cv"][n]["auc"].std()),
                         **{k: float(v) for k, v in res["prueba"][n].items()}},
            "sklearn": sklearn.__version__}


def construir_regresion(df):
    cats, nums = m.columnas_regresion(False)           # sin mes, dia ni duracion
    res = m.entrenar_regresion(df, cats, nums, ["Random Forest"])
    n = "Random Forest"
    p = res["prueba"]
    cuenta = df["campaign"].clip(upper=15).value_counts().sort_index()
    return {"tarea": "regresion", "nombre": n, "modelo": res["modelos"][n], "cats": cats, "nums": nums,
            "rangos": _rangos(df, cats, nums),
            "distribucion": {int(k): int(v) for k, v in cuenta.items()},       # 15 = 15 o más
            "promedio": float(df["campaign"].mean()), "mediana": float(df["campaign"].median()),
            "metricas": {"rmse_cv": float(np.mean(res["cv"][n]["RMSE"])), "rmse_prueba": float(p[n]["RMSE"]),
                         "r2_prueba": float(p[n]["R2"]), "rmse_promedio_prueba": float(p["Promedio (ficticio)"]["RMSE"])},
            "sklearn": sklearn.__version__}


def guardar(art):
    CARPETA.mkdir(exist_ok=True)
    joblib.dump(art, ARCHIVOS[art["tarea"]], compress=3)


def cargar(tarea):
    """Devuelve el diccionario guardado o None si el archivo no existe."""
    ruta = ARCHIVOS[tarea]
    if not ruta.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(ruta)


def _fila(art, valores):
    cols = art["cats"] + art["nums"]
    return pd.DataFrame([{c: valores[c] for c in cols}], columns=cols)


def clasificar(art, valores):
    p = float(art["modelo"].predict_proba(_fila(art, valores))[0, 1])
    arriba = float(np.mean(art["puntajes_prueba"] > p))          # fracción de clientes de la prueba con mayor puntaje
    decil = min(int(arriba * 10) + 1, 10)
    return {"probabilidad": p, "umbral": art["umbral"], "marcar": p >= art["umbral"],
            "porcentaje_superior": max(arriba, 0.01), "decil": decil, "tasa_decil": art["tasas_decil"][decil - 1]}


def predecir_contactos(art, valores):
    return float(art["modelo"].predict(_fila(art, valores))[0])


def fig_decil_cliente(art, decil):
    t = [100 * x for x in art["tasas_decil"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(range(1, 11), t, color=[m.COLOR["HistGradientBoosting"] if i + 1 == decil else "#c9c8c2" for i in range(10)], width=0.65)
    ax.axhline(100 * art["tasa_base"], color=m.GRIS, lw=1.2, ls="--")
    ax.text(10.4, 100 * art["tasa_base"] + 1, f"Tasa base {m.pct(art['tasa_base'])}", ha="right", fontsize=8, color=m.TINTA2)
    for i, v in enumerate(t, 1):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set(xlabel="Tramo de la lista (1 = el 10 % con mayor puntaje). En verde, donde cae este cliente",
           ylabel="% que se suscribió", xticks=range(1, 11), title="Tasa de suscripción histórica por tramo")
    fig.tight_layout()
    return fig


def fig_distribucion(art, pred):
    d = art["distribucion"]; ks = sorted(d); tot = sum(d.values())
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(ks, [100 * d[k] / tot for k in ks], color="#c9c8c2", width=0.7)
    ax.axvline(pred, color=m.COLOR["Random Forest"], lw=2)
    ax.text(pred + 0.2, ax.get_ylim()[1] * 0.9, f"Predicción: {m.num(pred, 1)}", color=m.COLOR["Random Forest"], fontsize=9)
    ax.set(xlabel="Contactos de la campaña (15 = 15 o más)", ylabel="% de clientes", xticks=ks,
           title="Cómo se distribuyen los contactos en los datos")
    fig.tight_layout()
    return fig
