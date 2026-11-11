"""Lógica de entrenamiento y gráficos de la app de Bank Marketing (no usa Streamlit).

Mismo protocolo que los Colab: 80 % entrenamiento / 20 % prueba, validación cruzada con los mismos
pliegues para todos los modelos, línea de base, umbral elegido en validación cruzada, prueba abierta una vez.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
                             brier_score_loss, mean_squared_error, mean_absolute_error, r2_score)

# ----------------------------------------------------------------------------- constantes
CATEGORICAS = ["trabajo", "estado_civil", "educacion", "default", "vivienda", "prestamo", "contacto", "mes", "poutcome"]
NUMERICAS = ["edad", "balance", "dia", "campaign", "pdays", "previous", "duracion"]
COLUMNAS_REQUERIDAS = CATEGORICAS + NUMERICAS + ["Class"]

MODELOS_CLASIFICACION = ["Regresión logística", "Random Forest", "HistGradientBoosting"]
MODELOS_REGRESION = ["Ridge (lineal)", "Random Forest", "HistGradientBoosting"]

COLOR = {"Regresión logística": "#2a78d6", "Ridge (lineal)": "#2a78d6",
         "Random Forest": "#eb6834", "HistGradientBoosting": "#1baf7a"}
GRIS, TINTA, TINTA2, FONDO, ROJO = "#8a8985", "#0b0b0b", "#52514e", "#fcfcfb", "#e34948"


def estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
        "text.color": TINTA, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 100})


estilo()


def pct(x, d=1):
    return f"{100 * x:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


# ----------------------------------------------------------------------------- datos
def preparar_df(df):
    """Valida las columnas y agrega la columna y (1 = se suscribió) sin tocar la original."""
    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise ValueError("Faltan columnas en el archivo: " + ", ".join(faltan))
    valores = sorted(df["Class"].unique())
    out = df.copy()
    if valores == [1, 2]:
        out["y"] = (df["Class"] == 2).astype(int)
    elif valores == [0, 1]:
        out["y"] = df["Class"].astype(int)
    else:
        raise ValueError(f"Valores inesperados en Class: {valores}")
    return out


def columnas_clasificacion(incluir_campaign=False, incluir_duracion=False):
    nums = [c for c in ["edad", "balance", "dia", "campaign", "pdays", "previous"] if c != "campaign" or incluir_campaign]
    if incluir_duracion:
        nums.append("duracion")
    return list(CATEGORICAS), nums


def columnas_regresion(incluir_fecha=False):
    cats = [c for c in CATEGORICAS if incluir_fecha or c != "mes"]
    nums = [c for c in ["edad", "balance", "dia", "pdays", "previous"] if incluir_fecha or c != "dia"]
    return cats, nums


# ----------------------------------------------------------------------------- preparación por modelo
def _prep_lineal(cats, nums):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                              ("num", StandardScaler(), nums)])


def _prep_bosque(cats, nums):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                              ("num", "passthrough", nums)])


def _prep_histograma(cats, nums):
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
                              ("num", "passthrough", nums)])


def _candidatos_clasificacion(cats, nums, semilla):
    mascara = [True] * len(cats) + [False] * len(nums)
    return {
        "Regresión logística": (
            make_pipeline(_prep_lineal(cats, nums), LogisticRegression(max_iter=3000)),
            {"logisticregression__C": [0.01, 0.1, 1, 10]},
            {"logisticregression__C": 0.1}),
        "Random Forest": (
            make_pipeline(_prep_bosque(cats, nums), RandomForestClassifier(
                n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=semilla)),
            {"randomforestclassifier__min_samples_leaf": [10, 30]},
            {"randomforestclassifier__min_samples_leaf": 10}),
        "HistGradientBoosting": (
            make_pipeline(_prep_histograma(cats, nums), HistGradientBoostingClassifier(
                categorical_features=mascara, max_iter=200, random_state=semilla)),
            {"histgradientboostingclassifier__learning_rate": [0.05, 0.1],
             "histgradientboostingclassifier__max_depth": [3, 6]},
            {"histgradientboostingclassifier__learning_rate": 0.05,
             "histgradientboostingclassifier__max_depth": 6}),
    }


def _candidatos_regresion(cats, nums, semilla):
    mascara = [True] * len(cats) + [False] * len(nums)
    con_fecha = "mes" in cats            # ganadoras de la grilla: con fecha (alpha 10, hoja 10) / sin fecha (alpha 100, hoja 50)
    alfa, hoja = (10, 10) if con_fecha else (100, 50)
    return {
        "Ridge (lineal)": (
            make_pipeline(_prep_lineal(cats, nums), Ridge()),
            {"ridge__alpha": [0.1, 1, 10, 100]},
            {"ridge__alpha": alfa}),
        "Random Forest": (
            make_pipeline(_prep_bosque(cats, nums), RandomForestRegressor(
                n_estimators=200, max_features=0.33, n_jobs=-1, random_state=semilla)),
            {"randomforestregressor__min_samples_leaf": [10, 20, 50]},
            {"randomforestregressor__min_samples_leaf": hoja}),
        "HistGradientBoosting": (
            make_pipeline(_prep_histograma(cats, nums), HistGradientBoostingRegressor(
                categorical_features=mascara, max_iter=200, random_state=semilla)),
            {"histgradientboostingregressor__learning_rate": [0.05, 0.1],
             "histgradientboostingregressor__max_depth": [3, 6]},
            {"histgradientboostingregressor__learning_rate": 0.05,
             "histgradientboostingregressor__max_depth": 6}),
    }


def _sin_prefijo(params):
    return {k.split("__", 1)[1]: v for k, v in params.items()}


def _avisar(progreso, fraccion, texto):
    if progreso is not None:
        progreso(min(max(fraccion, 0.0), 1.0), texto)


# ----------------------------------------------------------------------------- clasificación
def umbral_max_f1(y_real, p):
    pr, rc, th = precision_recall_curve(y_real, p)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[np.argmax(f1)])


def entrenar_clasificacion(df, cats, nums, nombres, ajustar=False, n_splits=5, semilla=42, progreso=None):
    X, y = df[cats + nums], df["y"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, stratify=y, random_state=semilla)
    pliegues = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    idxs = [idx for _, idx in pliegues.split(X_ent, y_ent)]
    cand = _candidatos_clasificacion(cats, nums, semilla)

    res = {"tarea": "clasificacion", "cats": cats, "nums": nums, "nombres": list(nombres),
           "X_ent": X_ent, "X_pru": X_pru, "y_ent": y_ent, "y_pru": y_pru, "pliegues": pliegues,
           "n_splits": n_splits, "semilla": semilla, "modelos": {}, "params": {}, "cv": {},
           "puntajes_cv": {}, "puntajes_pru": {}, "umbrales": {}, "prueba": {}}
    total = len(nombres) + 1
    for i, n in enumerate(nombres):
        _avisar(progreso, i / total, f"Entrenando {n}…")
        pipe, grilla, fijos = cand[n]
        if ajustar:
            gs = GridSearchCV(pipe, grilla, cv=pliegues, scoring="roc_auc", n_jobs=1).fit(X_ent, y_ent)
            modelo, params = gs.best_estimator_, _sin_prefijo(gs.best_params_)
        else:
            modelo, params = pipe.set_params(**fijos), _sin_prefijo(fijos)
        p = cross_val_predict(modelo, X_ent, y_ent, cv=pliegues, method="predict_proba")[:, 1]
        res["cv"][n] = {"auc": np.array([roc_auc_score(y_ent.iloc[ix], p[ix]) for ix in idxs]),
                        "ap": np.array([average_precision_score(y_ent.iloc[ix], p[ix]) for ix in idxs])}
        umbral = umbral_max_f1(y_ent, p)
        modelo.fit(X_ent, y_ent)
        s = modelo.predict_proba(X_pru)[:, 1]
        pred = (s >= umbral).astype(int)
        res["modelos"][n], res["params"][n] = modelo, params
        res["puntajes_cv"][n], res["puntajes_pru"][n], res["umbrales"][n] = p, s, umbral
        res["prueba"][n] = {"AUC": roc_auc_score(y_pru, s), "AP": average_precision_score(y_pru, s), "Umbral": umbral,
                            "Precisión": precision_score(y_pru, pred, zero_division=0),
                            "Exhaustividad": recall_score(y_pru, pred), "F1": f1_score(y_pru, pred)}

    _avisar(progreso, len(nombres) / total, "Calculando la línea de base…")
    azar = DummyClassifier(strategy="stratified", random_state=semilla)
    pa = cross_val_predict(azar, X_ent, y_ent, cv=pliegues, method="predict_proba")[:, 1]
    res["cv"]["Azar (ficticio)"] = {"auc": np.array([roc_auc_score(y_ent.iloc[ix], pa[ix]) for ix in idxs]),
                                    "ap": np.array([average_precision_score(y_ent.iloc[ix], pa[ix]) for ix in idxs])}
    azar.fit(X_ent, y_ent)
    pz = azar.predict(X_pru)
    res["prueba"]["Azar (ficticio)"] = {"AUC": float(roc_auc_score(y_pru, azar.predict_proba(X_pru)[:, 1])), "AP": float(y_pru.mean()), "Umbral": np.nan,
                                        "Precisión": precision_score(y_pru, pz, zero_division=0),
                                        "Exhaustividad": recall_score(y_pru, pz), "F1": f1_score(y_pru, pz)}
    res["elegido"] = max(nombres, key=lambda n: res["cv"][n]["auc"].mean())
    _avisar(progreso, 1.0, "Listo")
    return res


def tabla_cv_clasificacion(res):
    filas = {n: {"AUC (media ± desvío)": f"{num(v['auc'].mean())} ± {num(v['auc'].std())}",
                 "AP (media ± desvío)": f"{num(v['ap'].mean())} ± {num(v['ap'].std())}"}
             for n, v in res["cv"].items()}
    return pd.DataFrame(filas).T


def tabla_prueba_clasificacion(res):
    return pd.DataFrame(res["prueba"]).T.round(3)


def captura_por_lista(res, fracciones=(0.10, 0.20, 0.30, 0.50)):
    """% de todos los suscriptores que se captan llamando solo a la parte mejor puntuada de la lista."""
    y_pru = res["y_pru"].values
    filas = []
    for k in fracciones:
        fila = {"si se llama al": pct(k, 0)}
        for n in res["nombres"]:
            yo = y_pru[np.argsort(-res["puntajes_pru"][n])]
            m = int(len(yo) * k)
            fila[n] = f"{pct(yo[:m].sum() / yo.sum())} (precisión {pct(yo[:m].mean(), 0)})"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("si se llama al")


def tasa_por_mes_y_anio(df):
    """Tasa de suscripción por mes y año inferido. Supone el archivo ordenado por fecha: el año sube cada vez que el mes 'da la vuelta'."""
    meses = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    n = df["mes"].map({k: i for i, k in enumerate(meses)}).values
    anio = np.concatenate([[0], np.cumsum(n[1:] < n[:-1])]) + 1
    t = df.assign(anio=anio).pivot_table(index="mes", columns="anio", values="y", aggfunc=["mean", "size"])
    t = t.reindex([k for k in meses if k in t.index])
    out = pd.DataFrame(index=t.index)
    for a in t["mean"].columns:
        out[f"Año {a}"] = [f"{pct(r)} ({int(k):,})".replace(",", ".") if k == k else "—" for r, k in zip(t["mean"][a], t["size"][a])]
    por_anio = df.assign(anio=anio).groupby("anio")["y"].agg(["mean", "size"])
    return out, por_anio


def tabla_temporal_clasificacion(df, cats, nums, nombres, semilla=42, proporcion=0.8):
    """Entrena con el primer 80 % del archivo (en su orden) y prueba con el último 20 %. Con los parámetros por defecto del app."""
    k = int(len(df) * proporcion)
    a, b = df.iloc[:k], df.iloc[k:]
    cand = _candidatos_clasificacion(cats, nums, semilla)
    filas = {}
    for n in nombres:
        pipe, _, fijos = cand[n]
        modelo = pipe.set_params(**fijos).fit(a[cats + nums], a["y"])
        s = modelo.predict_proba(b[cats + nums])[:, 1]
        yo = b["y"].values[np.argsort(-s)]
        top = int(len(yo) * 0.2)
        filas[n] = {"AUC temporal": num(roc_auc_score(b["y"], s)), "Captura llamando al 20 %": pct(yo[:top].sum() / yo.sum())}
    filas["(tasa de suscripción: primer 80 % → último 20 %)"] = {"AUC temporal": f"{pct(a['y'].mean())} → {pct(b['y'].mean())}", "Captura llamando al 20 %": ""}
    return pd.DataFrame(filas).T


def _veredicto(d):
    if abs(d.mean()) > 2 * d.std() and ((d > 0).all() or (d < 0).all()):
        return "se distinguen"
    return "empate técnico"


def comparar_pliegues(res):
    """Compara los modelos de a pares, pliegue a pliegue (AUC en clasificación, RMSE en predicción)."""
    nombres = res["nombres"]
    clasif = res["tarea"] == "clasificacion"
    filas = []
    for j in range(1, len(nombres)):
        for i in range(j):
            a, b = nombres[j], nombres[i]
            if clasif:
                d = res["cv"][a]["auc"] - res["cv"][b]["auc"]
            else:
                d = res["cv"][b]["rmse"] - res["cv"][a]["rmse"]      # positivo = a se equivoca menos
            filas.append({"Comparación": f"{a} vs {b}", "Ventaja media": round(float(d.mean()), 4),
                          "Variación entre pliegues": round(float(d.std()), 4),
                          "Cociente |media|/desvío": round(float(abs(d.mean()) / d.std()), 2) if d.std() > 0 else float("inf"),
                          "Gana en": f"{int((d > 0).sum())} de {len(d)} pliegues", "Veredicto": _veredicto(d)})
    return pd.DataFrame(filas).set_index("Comparación")


def fig_curvas(res):
    y_ent = res["y_ent"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for n in res["nombres"]:
        p = res["puntajes_cv"][n]
        pr, rc, _ = precision_recall_curve(y_ent, p)
        ax1.plot(rc[:-1], pr[:-1], color=COLOR[n], lw=2, label=f"{n} (AP {num(average_precision_score(y_ent, p), 2)})")
        fpr, tpr, _ = roc_curve(y_ent, p)
        ax2.plot(fpr, tpr, color=COLOR[n], lw=2, label=f"{n} (AUC {num(roc_auc_score(y_ent, p), 2)})")
    ax1.axhline(y_ent.mean(), color=GRIS, lw=1.5, ls="--", label=f"Piso: {pct(y_ent.mean())}")
    ax2.plot([0, 1], [0, 1], color=GRIS, lw=1.5, ls="--", label="Azar")
    ax1.set(title="Curva PR (validación cruzada)", xlabel="Exhaustividad", ylabel="Precisión", xlim=(0, 1), ylim=(0, 1))
    ax2.set(title="Curva ROC (validación cruzada)", xlabel="Tasa de falsos positivos",
            ylabel="Tasa de verdaderos positivos", xlim=(0, 1), ylim=(0, 1))
    ax1.legend(fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_matriz(res, nombre):
    pred = (res["puntajes_pru"][nombre] >= res["umbrales"][nombre]).astype(int)
    cm = confusion_matrix(res["y_pru"], pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay(cm, display_labels=["No se suscribe", "Se suscribe"]).plot(
        cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.grid(False)
    ax.set(xlabel="Lo que dijo el modelo", ylabel="Lo que pasó realmente", title=f"Matriz de confusión: {nombre}")
    fig.tight_layout()
    return fig


def texto_matriz(res, nombre):
    pred = (res["puntajes_pru"][nombre] >= res["umbrales"][nombre]).astype(int)
    vn, fp, fn, vp = confusion_matrix(res["y_pru"], pred).ravel()
    base = res["y_pru"].mean()
    return (f"Había {vp + fn} suscriptores: el modelo detectó {vp} ({pct(vp / (vp + fn))}) y dejó pasar {fn}. "
            f"Marcó {vp + fp} clientes: {vp} acertaron ({pct(vp / (vp + fp))}) y {fp} fueron llamadas de más. "
            f"Sin modelo, un cliente cualquiera se suscribe el {pct(base)}.")


def fig_deciles(res, nombre):
    y_ord = res["y_pru"].values[np.argsort(-res["puntajes_pru"][nombre])]
    tasas = [d.mean() for d in np.array_split(y_ord, 10)]
    base = res["y_pru"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(1, 11), tasas, color=COLOR[nombre], width=0.65)
    ax.axhline(base, color=GRIS, lw=1.5, ls="--")
    ax.text(10.4, base, f"Tasa base {pct(base)}", va="bottom", ha="right", color=TINTA2, fontsize=9)
    for i, t in enumerate(tasas, 1):
        ax.text(i, t + 0.006, pct(t, 0), ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="Tramo de la lista (1 = el 10 % con mayor puntaje)", ylabel="Proporción que se suscribió",
           xticks=range(1, 11), title=f"Tasa de suscripción por tramo ({nombre}, prueba)")
    ax.set_ylim(0, max(tasas) * 1.15)
    fig.tight_layout()
    return fig


def fig_calibracion(res):
    y_ent = res["y_ent"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    tope = 0.1
    for n in res["nombres"]:
        real, dicha = calibration_curve(y_ent, res["puntajes_cv"][n], n_bins=10, strategy="quantile")
        tope = max(tope, real.max(), dicha.max())
        ax.plot(dicha, real, color=COLOR[n], lw=2, marker="o", ms=4,
                label=f"{n} (Brier {num(brier_score_loss(y_ent, res['puntajes_cv'][n]), 4)})")
    tope *= 1.08
    ax.plot([0, tope], [0, tope], color=GRIS, lw=1.5, ls="--", label="Calibración perfecta")
    ax.set(xlabel="Probabilidad que dijo el modelo", ylabel="Fracción que realmente se suscribió",
           xlim=(0, tope), ylim=(0, tope), title="Diagrama de confiabilidad")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- predicción
def entrenar_regresion(df, cats, nums, nombres, ajustar=False, n_splits=5, semilla=42, progreso=None):
    X, y = df[cats + nums], df["campaign"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=semilla)
    pliegues = KFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    idxs = [idx for _, idx in pliegues.split(X_ent, y_ent)]
    cand = _candidatos_regresion(cats, nums, semilla)

    def metricas(y_real, pred):
        return {"RMSE": float(np.sqrt(mean_squared_error(y_real, pred))),
                "MAE": float(mean_absolute_error(y_real, pred)), "R2": float(r2_score(y_real, pred))}

    def por_pliegue(p):
        ms = [metricas(y_ent.iloc[ix], p[ix]) for ix in idxs]
        return {k: np.array([m[k] for m in ms]) for k in ("RMSE", "MAE", "R2")} | {"rmse": np.array([m["RMSE"] for m in ms])}

    res = {"tarea": "regresion", "cats": cats, "nums": nums, "nombres": list(nombres), "X_ent": X_ent, "X_pru": X_pru,
           "y_ent": y_ent, "y_pru": y_pru, "pliegues": pliegues, "n_splits": n_splits, "semilla": semilla,
           "modelos": {}, "params": {}, "cv": {}, "pred_cv": {}, "pred_pru": {}, "prueba": {}, "metricas": metricas}
    total = len(nombres) + 1
    for i, n in enumerate(nombres):
        _avisar(progreso, i / total, f"Entrenando {n}…")
        pipe, grilla, fijos = cand[n]
        if ajustar:
            gs = GridSearchCV(pipe, grilla, cv=pliegues, scoring="neg_root_mean_squared_error", n_jobs=1).fit(X_ent, y_ent)
            modelo, params = gs.best_estimator_, _sin_prefijo(gs.best_params_)
        else:
            modelo, params = pipe.set_params(**fijos), _sin_prefijo(fijos)
        p = cross_val_predict(modelo, X_ent, y_ent, cv=pliegues)
        res["cv"][n], res["pred_cv"][n] = por_pliegue(p), p
        modelo.fit(X_ent, y_ent)
        res["modelos"][n], res["params"][n] = modelo, params
        res["pred_pru"][n] = modelo.predict(X_pru)
        res["prueba"][n] = metricas(y_pru, res["pred_pru"][n])

    _avisar(progreso, len(nombres) / total, "Calculando la línea de base…")
    for etiqueta, estrategia in [("Promedio (ficticio)", "mean"), ("Mediana (ficticio)", "median")]:
        d = DummyRegressor(strategy=estrategia)
        res["cv"][etiqueta] = por_pliegue(cross_val_predict(d, X_ent, y_ent, cv=pliegues))
        d.fit(X_ent, y_ent)
        res["pred_pru"][etiqueta] = d.predict(X_pru)
        res["prueba"][etiqueta] = metricas(y_pru, res["pred_pru"][etiqueta])
    res["elegido"] = min(nombres, key=lambda n: res["cv"][n]["rmse"].mean())
    _avisar(progreso, 1.0, "Listo")
    return res


def tabla_cv_regresion(res):
    filas = {n: {m: f"{num(v[m].mean())} ± {num(v[m].std())}" for m in ("RMSE", "MAE", "R2")} for n, v in res["cv"].items()}
    return pd.DataFrame(filas).T


def tabla_prueba_regresion(res):
    return pd.DataFrame(res["prueba"]).T.round(3)


def fig_rmse_cv(res):
    orden = list(res["nombres"]) + ["Promedio (ficticio)"]
    medias = [res["cv"][n]["rmse"].mean() for n in orden]
    desvios = [res["cv"][n]["rmse"].std() for n in orden]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(len(orden)), medias, yerr=desvios, color=[COLOR.get(n, GRIS) for n in orden], width=0.6,
           capsize=4, ecolor=TINTA2)
    for i, v in enumerate(medias):
        ax.text(i, v / 2, num(v, 2), ha="center", color="white" if i != len(orden) - 1 else TINTA, fontweight="bold")
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([n.replace(" (ficticio)", "\n(piso)").replace(" (lineal)", "\n(lineal)") for n in orden], fontsize=9)
    ax.set(ylabel="RMSE en contactos (menor es mejor)", title="Error en validación cruzada")
    fig.tight_layout()
    return fig


def fig_real_vs_predicho(res):
    y_pru = res["y_pru"]
    etiquetas = ["1", "2", "3", "4", "5", "6 a 10", "11 o más"]
    grupo = pd.cut(y_pru, bins=[0, 1, 2, 3, 4, 5, 10, 1000], labels=etiquetas)
    real_g, n_g = y_pru.groupby(grupo).mean(), y_pru.groupby(grupo).size()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(range(len(etiquetas)), real_g.values, color=GRIS, lw=2, ls="--", marker="o", ms=4, label="Real (promedio del grupo)")
    for n in res["nombres"]:
        p_g = pd.Series(res["pred_pru"][n], index=y_pru.index).groupby(grupo).mean()
        ax.plot(range(len(etiquetas)), p_g.values, color=COLOR[n], lw=2, marker="o", ms=4, label=f"{n} (predicho)")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels([f"{e}\n(n = {n_g[e]})" for e in etiquetas], fontsize=8)
    ax.set(xlabel="Contactos reales del cliente", ylabel="Contactos (promedio del grupo)",
           title="Predicho contra real, por grupo (prueba)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_errores(res, nombre):
    err = res["y_pru"].values - res["pred_pru"][nombre]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(np.clip(err, -6, 15), bins=np.arange(-6.5, 15.6, 1), color=COLOR[nombre], rwidth=0.85)
    ax.axvline(0, color=GRIS, lw=1.5, ls="--")
    ax.set(xlabel="Error = real − predicho, en contactos (agrupado en -6 y 15)", ylabel="Cantidad de clientes",
           title=f"Errores de {nombre} (prueba)")
    fig.tight_layout()
    return fig


def tabla_practica_regresion(res):
    """Acierto con error de 1 contacto o menos, sin y con predicción redondeada (la mediana ficticia siempre da un entero)."""
    y = res["y_pru"].values
    dificil = (res["y_pru"] >= 5).astype(int)

    def acierta(p):
        return pct(float((np.abs(y - p) <= 1).mean()))

    filas = {}
    for n in res["nombres"]:
        p = res["pred_pru"][n]
        filas[n] = {"Error ≤ 1 (sin redondear)": acierta(p), "Error ≤ 1 (redondeada)": acierta(np.round(p)),
                    "AUC para señalar clientes de 5+ contactos": num(roc_auc_score(dificil, p))}
    p = res["pred_pru"]["Mediana (ficticio)"]
    filas["Mediana (ficticio)"] = {"Error ≤ 1 (sin redondear)": acierta(p), "Error ≤ 1 (redondeada)": acierta(np.round(p)),
                                   "AUC para señalar clientes de 5+ contactos": "0,500"}
    return pd.DataFrame(filas).T


def perfil_vs_calendario(res):
    """Reentrena el Random Forest con la fecha sola y sin la fecha, para ver qué explica los contactos."""
    if "Random Forest" not in res["modelos"]:
        raise ValueError("Hace falta entrenar el Random Forest")
    params = res["modelos"]["Random Forest"][-1].get_params()

    def r2_con(cats_u, nums_u):
        modelo = make_pipeline(_prep_bosque(cats_u, nums_u), RandomForestRegressor(**params))
        cols = cats_u + nums_u
        modelo.fit(res["X_ent"][cols], res["y_ent"])
        return res["metricas"](res["y_pru"], modelo.predict(res["X_pru"][cols]))

    cats, nums = res["cats"], res["nums"]
    filas = {"Todas las variables del modelo": res["prueba"]["Random Forest"]}
    if "mes" in cats and "dia" in nums:
        filas["Solo mes y día (fecha del último contacto)"] = r2_con(["mes"], ["dia"])
        filas["Sin mes ni día (cliente, medio de contacto y campaña anterior)"] = r2_con([c for c in cats if c != "mes"],
                                                                      [c for c in nums if c != "dia"])
    return pd.DataFrame(filas).T.round(3)


# ----------------------------------------------------------------------------- importancias y predicción individual
def importancia_permutacion(res, nombres=None, repeticiones=5):
    nombres = nombres or [n for n in res["nombres"] if n != "Regresión logística" and n != "Ridge (lineal)"]
    scoring = "roc_auc" if res["tarea"] == "clasificacion" else "r2"
    out = {}
    for n in nombres:
        r = permutation_importance(res["modelos"][n], res["X_pru"], res["y_pru"], scoring=scoring,
                                   n_repeats=repeticiones, random_state=res["semilla"], n_jobs=1)
        out[n] = pd.Series(r.importances_mean, index=res["X_pru"].columns)
    return pd.DataFrame(out).sort_values(nombres[-1], ascending=False)


def fig_importancia(imp, unidad):
    top = imp.mean(axis=1).sort_values(ascending=False).head(8).index
    fig, ax = plt.subplots(figsize=(7, 4.3))
    k = len(imp.columns)
    alto = 0.8 / k
    for i, n in enumerate(imp.columns):
        ax.barh(np.arange(len(top)) + (i - (k - 1) / 2) * alto, imp.loc[top[::-1], n].values,
                height=alto - 0.04, color=COLOR[n], label=n)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(list(top[::-1]))
    ax.set(xlabel=f"Caída del {unidad} al barajar la variable (prueba)", title="Importancia por permutación")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def predecir_cliente(res, nombre, valores):
    """valores: dict {columna: valor}. Devuelve la probabilidad (clasificación) o los contactos esperados (predicción)."""
    cols = res["cats"] + res["nums"]
    fila = pd.DataFrame([{c: valores[c] for c in cols}], columns=cols)
    modelo = res["modelos"][nombre]
    if res["tarea"] == "clasificacion":
        p = float(modelo.predict_proba(fila)[0, 1])
        return {"probabilidad": p, "umbral": res["umbrales"][nombre], "marcar": p >= res["umbrales"][nombre]}
    return {"contactos": float(modelo.predict(fila)[0])}


# ----------------------------------------------------------------------------- glosario
# Descripciones contrastadas con datasetDiccionario.arff (diccionario de la cátedra, Bank Marketing de Moro, Laureano y Cortez).
GLOSARIO = [
    ("edad", "Cliente", "Edad en años."),
    ("trabajo", "Cliente", "Tipo de trabajo (admin., blue-collar, technician, retired, student, etc.)."),
    ("estado_civil", "Cliente", "Casado, soltero o divorciado (divorciado incluye viudo)."),
    ("educacion", "Cliente", "Nivel educativo: primary, secondary, tertiary o unknown."),
    ("default", "Cliente", "Si tiene crédito en mora (yes/no)."),
    ("balance", "Cliente", "Saldo promedio anual de la cuenta, en euros (puede ser negativo)."),
    ("vivienda", "Cliente", "Si tiene préstamo hipotecario (yes/no)."),
    ("prestamo", "Cliente", "Si tiene un préstamo personal (yes/no)."),
    ("contacto", "Último contacto de esta campaña", "Medio de la última llamada: cellular, telephone o unknown."),
    ("dia", "Último contacto de esta campaña", "Día del mes del último contacto (1 a 31). No es del cliente: es del calendario de la campaña."),
    ("mes", "Último contacto de esta campaña", "Mes del último contacto. No es del cliente: es del calendario de la campaña."),
    ("duracion", "Último contacto de esta campaña", "Duración de la última llamada en segundos. Solo se conoce al cortar, cuando ya se sabe si compró: fuga de información."),
    ("campaign", "Esta campaña", "Cantidad de contactos con este cliente durante esta campaña, incluido el último."),
    ("pdays", "Campaña anterior", "Días desde el último contacto de una campaña anterior. -1 = nunca fue contactado antes (el 82 % de los casos)."),
    ("previous", "Campaña anterior", "Cantidad de contactos con este cliente antes de esta campaña."),
    ("poutcome", "Campaña anterior", "Resultado de la campaña anterior: success, failure, other o unknown."),
    ("Class", "Objetivo", "Si el cliente se suscribió al depósito a plazo (la `y` de la clasificación)."),
]
DESCRIPCION = {c: d for c, _, d in GLOSARIO}


def tabla_glosario():
    return pd.DataFrame(GLOSARIO, columns=["Columna", "Qué es", "Descripción"]).set_index("Columna")


# ----------------------------------------------------------------------------- exportar resultados
def _tablas_resultado(res):
    t = {}
    if res["tarea"] == "clasificacion":
        t["validacion_cruzada"] = tabla_cv_clasificacion(res)
        t["prueba"] = tabla_prueba_clasificacion(res)
        t["a_cuanta_gente_llamar"] = captura_por_lista(res)
    else:
        t["validacion_cruzada"] = tabla_cv_regresion(res)
        t["prueba"] = tabla_prueba_regresion(res)
        t["en_la_practica"] = tabla_practica_regresion(res)
    if len(res["nombres"]) > 1:
        t["comparacion_de_a_pares"] = comparar_pliegues(res)
    t["configuracion_elegida"] = pd.DataFrame({n: {k: v for k, v in p.items()} for n, p in res["params"].items()}).T
    return t


NOTAS = [
    "± = desvío entre pliegues calculado con ddof=0 (con ddof=1 el cociente de HGB vs Ridge en regresión baja de 2,03 a 1,82 y pasaría a empate técnico; los demás veredictos no cambian).",
    "HistGradientBoosting usa early stopping automático (10 % interno del entrenamiento): puede usar menos de las 200 iteraciones.",
    "La prueba (20 %) es una sola partición aleatoria: la cifra principal es la de validación cruzada.",
    "El archivo parece ordenado en el tiempo (inferido del orden de los meses) y la partición es aleatoria: el AUC agrupado mide también la separación entre épocas (mes, contacto, pdays y poutcome funcionan como reloj). Para una campaña nueva usar la validación temporal, que da bastante menos.",
    "La tasa de suscripción por mes mezcla años distintos (2008, 2009 y 2010 tienen tasas muy distintas): no prueba un efecto del mes del año.",
    "El cociente |media|/desvío debe superar 2 (y el signo repetirse en todos los pliegues) para decir que dos modelos se distinguen.",
]


def exportar_txt(res):
    titulo = "CLASIFICACIÓN (¿se suscribe?)" if res["tarea"] == "clasificacion" else "PREDICCIÓN (contactos = campaign)"
    L = [titulo, f"Variables: {', '.join(res['cats'] + res['nums'])}",
         f"Pliegues: {res['n_splits']} | Semilla: {res['semilla']} | Entrenamiento: {len(res['y_ent'])} | Prueba: {len(res['y_pru'])}",
         f"Mejor en validación cruzada: {res['elegido']}", ""]
    for nombre, tabla in _tablas_resultado(res).items():
        L += [f"== {nombre} ==", tabla.to_string(), ""]
    notas = NOTAS if res["tarea"] == "clasificacion" else NOTAS[:3] + [NOTAS[5], "La fecha (`mes`, `dia`) se excluyó a propósito de esta predicción: es la fecha del último contacto y depende de cuántas veces se contactó."]
    L += ["== notas =="] + [f"- {t}" for t in notas]
    return "\n".join(L)


def exportar_zip(res):
    """Devuelve los bytes de un .zip con un CSV por tabla (separador ; y coma decimal, para abrir en Excel en español)."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("resultados.txt", exportar_txt(res))
        for nombre, tabla in _tablas_resultado(res).items():
            z.writestr(f"{nombre}.csv", tabla.to_csv(sep=";", decimal=",", encoding="utf-8-sig"))
    return buf.getvalue()
