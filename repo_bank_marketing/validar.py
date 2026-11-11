"""Validación independiente del proyecto Bank Marketing (modelo final: clasificación sin `campaign`, predicción sin `mes` ni `dia`).

Uso:  python validar.py [--grilla]   (tarda ~5-6 minutos; --grilla suma la búsqueda en grilla, unos minutos más)
Reejecuta todo desde cero y compara contra los valores esperados (ESPERADOS.md). No modifica nada.
Las secciones 3 y 4 verifican que el código del app coincide con scikit-learn usado directo (no son evidencia independiente del protocolo);
las 7 y 8 son pruebas de estrés: ahí está lo que dice cuánto vale el modelo hacia adelante en el tiempo.
"""
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import modelos as m
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_predict
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error

RES = []


def chequeo(nombre, ok, detalle=""):
    RES.append(bool(ok))
    print(f"[{'OK   ' if ok else 'FALLA'}] {nombre}" + (f"  ->  {detalle}" if detalle else ""), flush=True)


def cerca(nombre, obtenido, esperado, tol):
    chequeo(nombre, abs(obtenido - esperado) <= tol, f"obtenido {obtenido:.4f}, esperado {esperado} ± {tol}")


def hgb_clas(cats, nums):
    return m._candidatos_clasificacion(cats, nums, 42)["HistGradientBoosting"][0].set_params(
        histgradientboostingclassifier__learning_rate=0.05, histgradientboostingclassifier__max_depth=6)


print("=== 1. Datos ===")
bruto = pd.read_csv("dataset_con_titulos.csv")
df = m.preparar_df(bruto)
chequeo("45.211 filas y 17 columnas", bruto.shape == (45211, 17), str(bruto.shape))
chequeo("5.289 suscripciones (11,7 %)", int(df["y"].sum()) == 5289, f"{int(df['y'].sum())} ({df['y'].mean():.4f})")
chequeo("Sin valores nulos", int(bruto.isna().sum().sum()) == 0)
chequeo("Sin filas duplicadas (ni con ni sin `Class`)",
        int(bruto.duplicated().sum()) == 0 and int(bruto.duplicated(subset=[x for x in bruto.columns if x != "Class"]).sum()) == 0)

print("\n=== 2. Clasificación final: sin `campaign` (5 pliegues, semilla 42) ===")
c, n = m.columnas_clasificacion()
chequeo("`campaign` y `duracion` no están entre las variables", "campaign" not in n and "duracion" not in n, str(n))
rc = m.entrenar_clasificacion(df, c, n, m.MODELOS_CLASIFICACION)
chequeo("Entrenamiento 36.168 / prueba 9.043", (len(rc["y_ent"]), len(rc["y_pru"])) == (36168, 9043))
for nombre, auc_cv, auc_pr, ap in [("Regresión logística", .759, .769, .407), ("Random Forest", .793, .803, .451),
                                   ("HistGradientBoosting", .798, .804, .460)]:
    cerca(f"{nombre}: AUC validación cruzada", rc["cv"][nombre]["auc"].mean(), auc_cv, 0.002)
    cerca(f"{nombre}: AUC prueba", rc["prueba"][nombre]["AUC"], auc_pr, 0.002)
    cerca(f"{nombre}: AP prueba", rc["prueba"][nombre]["AP"], ap, 0.003)
chequeo("Orden: HGB > RF > logística en AUC de validación cruzada y de prueba",
        rc["cv"]["HistGradientBoosting"]["auc"].mean() > rc["cv"]["Random Forest"]["auc"].mean() > rc["cv"]["Regresión logística"]["auc"].mean()
        and rc["prueba"]["HistGradientBoosting"]["AUC"] > rc["prueba"]["Random Forest"]["AUC"] > rc["prueba"]["Regresión logística"]["AUC"])
chequeo("Elegido en validación cruzada: HistGradientBoosting", rc["elegido"] == "HistGradientBoosting", rc["elegido"])
cerca("Azar: AUC en validación cruzada ≈ 0,5", rc["cv"]["Azar (ficticio)"]["auc"].mean(), 0.5, 0.01)
cerca("Azar: AP ≈ tasa base (0,117)", rc["cv"]["Azar (ficticio)"]["ap"].mean(), 0.117, 0.005)
comp = m.comparar_pliegues(rc)
chequeo("RF vs logística: se distinguen", comp.loc["Random Forest vs Regresión logística", "Veredicto"] == "se distinguen")
chequeo("HGB vs logística: se distinguen", comp.loc["HistGradientBoosting vs Regresión logística", "Veredicto"] == "se distinguen")
chequeo("HGB vs RF: empate técnico (HGB gana en 5 de 5 pliegues, cociente 1,9 < 2: borderline)",
        comp.loc["HistGradientBoosting vs Random Forest", "Veredicto"] == "empate técnico")
h = rc["prueba"]["HistGradientBoosting"]
cerca("HGB prueba: precisión", h["Precisión"], .457, 0.01)
cerca("HGB prueba: exhaustividad", h["Exhaustividad"], .544, 0.01)

print("\n=== 3. Consistencia del protocolo (el código del app contra scikit-learn directo) ===")
chequeo("Entrenamiento y prueba no comparten filas", len(set(rc["X_ent"].index) & set(rc["X_pru"].index)) == 0)
# pliegues y umbral recalculados por fuera de modelos.py
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
indep = StratifiedKFold(5, shuffle=True, random_state=42)
p_ind = cross_val_predict(hgb_clas(c, n), rc["X_ent"], rc["y_ent"], cv=indep, method="predict_proba")[:, 1]
cerca("HGB: AUC de validación cruzada recalculado con un StratifiedKFold propio", float(np.mean([roc_auc_score(rc["y_ent"].iloc[ix], p_ind[ix]) for _, ix in indep.split(rc["X_ent"], rc["y_ent"])])),
      float(rc["cv"]["HistGradientBoosting"]["auc"].mean()), 0.0005)
grilla_u = np.linspace(0.02, 0.9, 881)
f1s = [2 * precision_score(rc["y_ent"], p_ind >= u) * recall_score(rc["y_ent"], p_ind >= u) / (precision_score(rc["y_ent"], p_ind >= u) + recall_score(rc["y_ent"], p_ind >= u) + 1e-12) for u in grilla_u[::10]]
u_ind = grilla_u[::10][int(np.argmax(f1s))]
chequeo("HGB: el umbral que maximiza F1 en validación cruzada (búsqueda propia) coincide con el del app, ±0,03",
        abs(u_ind - rc["umbrales"]["HistGradientBoosting"]) < 0.03, f"propio {u_ind:.3f}, app {rc['umbrales']['HistGradientBoosting']:.3f}")
sp = rc["puntajes_pru"]["HistGradientBoosting"]
chequeo("HGB: el umbral del app aplicado a la prueba da la precisión y exhaustividad reportadas",
        abs(precision_score(rc["y_pru"], sp >= rc["umbrales"]["HistGradientBoosting"]) - rc["prueba"]["HistGradientBoosting"]["Precisión"]) < 1e-9
        and abs(recall_score(rc["y_pru"], sp >= rc["umbrales"]["HistGradientBoosting"]) - rc["prueba"]["HistGradientBoosting"]["Exhaustividad"]) < 1e-9)
cerca("Azar: AUC en prueba medido (no asignado a mano)", rc["prueba"]["Azar (ficticio)"]["AUC"], 0.5, 0.02)
rng = np.random.default_rng(0)
y_baj = pd.Series(rng.permutation(rc["y_ent"].values), index=rc["y_ent"].index)
cerca("Control de fuga: con la etiqueta barajada el AUC en prueba es ~0,5",
      roc_auc_score(rc["y_pru"], hgb_clas(c, n).fit(rc["X_ent"], y_baj).predict_proba(rc["X_pru"])[:, 1]), 0.5, 0.025)

print("\n=== 4. Reimplementación con scikit-learn directo (sin modelos.py; mismo corte 80/20 y semilla 42 a propósito) ===")
X = bruto[m.CATEGORICAS + ["edad", "balance", "dia", "pdays", "previous"]].copy()
for col in m.CATEGORICAS:
    X[col] = X[col].astype("category")
y = bruto["Class"].replace({1: 0, 2: 1}) if sorted(bruto["Class"].unique()) == [1, 2] else bruto["Class"]
Xe, Xp, ye, yp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
hg = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=200, learning_rate=0.05, max_depth=6, random_state=42).fit(Xe, ye)
cerca("HGB reimplementado a mano: AUC en prueba (categorías nativas)", roc_auc_score(yp, hg.predict_proba(Xp)[:, 1]), .804, 0.006)

print("\n=== 5. Fuga de información: `duracion` ===")
r_d = m.entrenar_clasificacion(df, *m.columnas_clasificacion(False, True), ["HistGradientBoosting"])
cerca("HGB con `duracion`: AUC en prueba (sube de .804 a ~0,93)", r_d["prueba"]["HistGradientBoosting"]["AUC"], .932, 0.005)

print("\n=== 6. Predicción final de `campaign`: sin `mes` ni `dia` ===")
c3, n3 = m.columnas_regresion()
chequeo("`mes` y `dia` no están entre las variables", "mes" not in c3 and "dia" not in n3)
rr = m.entrenar_regresion(df, c3, n3, m.MODELOS_REGRESION)
for nombre, rmse, r2 in [("Ridge (lineal)", 2.984, .016), ("Random Forest", 2.968, .026), ("HistGradientBoosting", 2.974, .023)]:
    cerca(f"{nombre}: RMSE en prueba", rr["prueba"][nombre]["RMSE"], rmse, 0.01)
    cerca(f"{nombre}: R² en prueba", rr["prueba"][nombre]["R2"], r2, 0.003)
cerca("Promedio ficticio: RMSE en prueba", rr["prueba"]["Promedio (ficticio)"]["RMSE"], 3.008, 0.01)
chequeo("Elegido (menor RMSE en validación cruzada): Random Forest", rr["elegido"] == "Random Forest", rr["elegido"])
chequeo("La mediana ficticia tiene mejor MAE que los modelos entrenados para la media (esperable: ellos optimizan RMSE)",
        rr["prueba"]["Mediana (ficticio)"]["MAE"] < min(rr["prueba"][k]["MAE"] for k in m.MODELOS_REGRESION))
tp = m.tabla_practica_regresion(rr)
f = lambda fila, col: float(tp.loc[fila, col].rstrip("%").replace(",", "."))
cerca("RF: acierto ±1 con predicción redondeada (%)", f("Random Forest", "Error ≤ 1 (redondeada)"), 57.9, 0.5)
cerca("'Siempre 2': acierto ±1 (%)", f("Mediana (ficticio)", "Error ≤ 1 (redondeada)"), 78.5, 0.5)
chequeo("Los modelos NO superan a 'siempre 2' en acierto ±1", f("Random Forest", "Error ≤ 1 (redondeada)") < f("Mediana (ficticio)", "Error ≤ 1 (redondeada)"))
cerca("RF: AUC para señalar clientes de 5+ contactos (débil)", float(tp.loc["Random Forest", "AUC para señalar clientes de 5+ contactos"].replace(",", ".")), .616, 0.01)
comp_r = m.comparar_pliegues(rr)
chequeo("Regresión: RF y HGB ganan a Ridge en 5 de 5 pliegues",
        comp_r.loc["Random Forest vs Ridge (lineal)", "Gana en"] == "5 de 5 pliegues" and comp_r.loc["HistGradientBoosting vs Ridge (lineal)", "Gana en"] == "5 de 5 pliegues")
chequeo("Regresión: HGB vs Ridge, 'se distinguen' pero BORDERLINE (cociente 2,03, apenas sobre 2)", 
        comp_r.loc["HistGradientBoosting vs Ridge (lineal)", "Veredicto"] == "se distinguen" and 2.0 < float(comp_r.loc["HistGradientBoosting vs Ridge (lineal)", "Cociente |media|/desvío"]) < 2.2)
chequeo("Regresión: RF vs HGB es empate técnico", comp_r.loc["HistGradientBoosting vs Random Forest", "Veredicto"] == "empate técnico")

print("\n=== 7. Robustez y pruebas de estrés ===")
casos = [([.1, .1, .1, .1, .1], "se distinguen"), ([.1, -.1, .1, -.1, .1], "empate técnico"),
         ([.01, .1, .01, .1, .01], "empate técnico"), ([.05, .05, .05, .05, -.01], "empate técnico"), ([.02, .03, .025, .03, .028], "se distinguen")]
chequeo("Regla 'se distinguen / empate técnico': 5 casos sintéticos", all(m._veredicto(np.array(d)) == e for d, e in casos))
# 7a. con `campaign` (versión anterior): el costo de sacarla es chico
r_cc = m.entrenar_clasificacion(df, *m.columnas_clasificacion(True, False), ["HistGradientBoosting"])
cerca("Con `campaign` (versión anterior): AUC en prueba .808, o sea sacarla cuesta ~.004", r_cc["prueba"]["HistGradientBoosting"]["AUC"], .808, 0.003)
# 7b. escenario estrictamente ex ante: sin fecha ni medio del contacto
cs = [x for x in c if x not in ("mes", "contacto")]
ns = [x for x in n if x != "dia"]
r_ex = m.entrenar_clasificacion(df, cs, ns, ["HistGradientBoosting"])
cerca("Estrictamente ex ante (sin `mes`, `dia`, `contacto`): AUC de prueba ≈ 0,74", r_ex["prueba"]["HistGradientBoosting"]["AUC"], .738, 0.006)
cerca("Estrictamente ex ante: AUC de validación cruzada", r_ex["cv"]["HistGradientBoosting"]["auc"].mean(), .735, 0.006)
# 7c. otras particiones aleatorias
aucs = []
for sem in [1, 2, 3, 4, 5, 6]:
    Xe_, Xp_, ye_, yp_ = train_test_split(df[c + n], df["y"], test_size=0.2, stratify=df["y"], random_state=sem)
    aucs.append(roc_auc_score(yp_, hgb_clas(c, n).fit(Xe_, ye_).predict_proba(Xp_)[:, 1]))
cerca("AUC de prueba con 6 particiones distintas: promedio ~0,800 (semilla 42: .804)", float(np.mean(aucs)), .800, 0.006)
# 7d. validación temporal: el diccionario confirma que el archivo está ordenado por fecha (mayo 2008 a noviembre 2010)
k = int(len(df) * 0.8)
a_, b_ = df.iloc[:k], df.iloc[k:]
auc_t = roc_auc_score(b_["y"], hgb_clas(c, n).fit(a_[c + n], a_["y"]).predict_proba(b_[c + n])[:, 1])
cerca("Validación temporal (primer 80 % entrena, último 20 % prueba): AUC cae a ~0,67", auc_t, .669, 0.005)
cerca("  tasa de suscripción en ese último 20 % (corrimiento fuerte: 6,7 % → 31,6 %)", float(b_["y"].mean()), .316, 0.01)
# 7e. el calendario: con `mes` y `dia` la predicción de `campaign` sube a R² ~0,20 (pero son del último contacto)
c4, n4 = m.columnas_regresion(True)
r_f = m.entrenar_regresion(df, c4, n4, m.MODELOS_REGRESION)
cerca("Con `mes` y `dia`: RF R² en prueba ≈ 0,20", r_f["prueba"]["Random Forest"]["R2"], .203, 0.005)
pc = m.perfil_vs_calendario(r_f)
cerca("  R² con mes y día solos", pc.loc["Solo mes y día (fecha del último contacto)", "R2"], .161, 0.005)
cerca("  R² sin mes ni día (config. de la versión con fecha: hoja 10, por eso .019 y no .026)", pc.loc["Sin mes ni día (cliente, medio de contacto y campaña anterior)", "R2"], .019, 0.004)
# 7f. modelo de mediana (cuantil 0,5)
q = m._candidatos_regresion(c3, n3, 42)["HistGradientBoosting"][0]
q.steps[-1][1].set_params(loss="quantile", quantile=0.5)
q.fit(rr["X_ent"], rr["y_ent"])
pq = q.predict(rr["X_pru"])
yv = rr["y_pru"].values
cerca("HGB de mediana (cuantil 0,5): MAE en prueba, mejor que la mediana ficticia (1,527)", mean_absolute_error(yv, pq), 1.518, 0.02)
cerca("HGB de mediana: acierto ±1 redondeado (%), empata con 'siempre 2'", 100 * float((np.abs(yv - np.round(pq)) <= 1).mean()), 77.7, 1.0)

print("\n=== 8. Qué vale el modelo hacia adelante en el tiempo (el .80 agrupado mezcla épocas) ===")
# 8a. el año confunde la estacionalidad (año inferido por los saltos de mes del archivo ordenado)
_, por_anio = m.tasa_por_mes_y_anio(df)
chequeo("El archivo da 2 saltos de mes: 3 'años'", len(por_anio) == 3, str(len(por_anio)))
for a, esp in zip(por_anio.index, [.051, .171, .516]):
    cerca(f"Tasa de suscripción del año inferido {a} (la tasa por mes mezcla estos años)", float(por_anio.loc[a, "mean"]), esp, 0.002)
# 8b. Nadeau-Bengio: los dos casos borderline se tratan igual
def nb_p(d, ntr, nte):
    d = np.asarray(d, float); t_ = d.mean() / np.sqrt(d.var(ddof=1) * (1 / len(d) + nte / ntr)); return 2 * (1 - stats.t.cdf(abs(t_), len(d) - 1))
ntr, nte = len(rc["y_ent"]) * 0.8, len(rc["y_ent"]) * 0.2
p_clas = nb_p(rc["cv"]["HistGradientBoosting"]["auc"] - rc["cv"]["Random Forest"]["auc"], ntr, nte)
p_reg = nb_p(rr["cv"]["Ridge (lineal)"]["rmse"] - rr["cv"]["HistGradientBoosting"]["rmse"], ntr, nte)
cerca("Nadeau-Bengio, clasificación HGB vs RF: p ≈ .063 (borderline; el código dice empate)", p_clas, .063, 0.005)
cerca("Nadeau-Bengio, regresión HGB vs Ridge: p ≈ .054 (borderline; el código dice se distinguen)", p_reg, .054, 0.005)
chequeo("Los dos casos están entre p=.05 y p=.10: se informan como borderline, ninguno como diferencia firme", 0.05 < p_clas < 0.10 and 0.05 < p_reg < 0.10)
yb = rc["y_pru"].values; sa, sb = rc["puntajes_pru"]["HistGradientBoosting"], rc["puntajes_pru"]["Random Forest"]
rg = np.random.default_rng(0); dd = []
for _ in range(500):
    ix = rg.integers(0, len(yb), len(yb)); dd.append(roc_auc_score(yb[ix], sa[ix]) - roc_auc_score(yb[ix], sb[ix]))
lo, hi = np.percentile(dd, [2.5, 97.5])
chequeo("Bootstrap pareado en prueba: el IC95 de ΔAUC (HGB − RF) incluye 0 (HGB es desempate, no 'mejor')", lo < 0 < hi, f"[{lo:.4f}, {hi:.4f}]")
# 8b2. Brier (calibración) y sensibilidad al estimador del desvío
from sklearn.metrics import brier_score_loss
for nom, bcv, bpr in [("Regresión logística", .0866, .0857), ("Random Forest", .0825, .0815), ("HistGradientBoosting", .0817, .0806)]:
    cerca(f"Brier validación cruzada, {nom}", brier_score_loss(rc["y_ent"], rc["puntajes_cv"][nom]), bcv, 0.0003)
    cerca(f"Brier prueba, {nom}", brier_score_loss(rc["y_pru"], rc["puntajes_pru"][nom]), bpr, 0.0003)
dh = rr["cv"]["Ridge (lineal)"]["rmse"] - rr["cv"]["HistGradientBoosting"]["rmse"]
cerca("HGB vs Ridge: cociente con ddof=1 (el código usa ddof=0 y da 2,03: es un caso borde)", abs(dh.mean()) / dh.std(ddof=1), 1.82, 0.03)
# 8c. validación temporal: primer 80 % entrena, último 20 % prueba, los tres modelos
def temporal(cs, ns):
    k_ = int(len(df) * 0.8); a_, b_ = df.iloc[:k_], df.iloc[k_:]; cand = m._candidatos_clasificacion(cs, ns, 42); out = {}
    for nom in m.MODELOS_CLASIFICACION:
        p_, _, f_ = cand[nom]; mod = p_.set_params(**f_).fit(a_[cs + ns], a_["y"]); s_ = mod.predict_proba(b_[cs + ns])[:, 1]
        yo = b_["y"].values[np.argsort(-s_)]; out[nom] = (roc_auc_score(b_["y"], s_), yo[:int(len(yo) * .2)].sum() / yo.sum())
    return out
tt = temporal(c, n)
for nom, esp in [("Regresión logística", .686), ("Random Forest", .697), ("HistGradientBoosting", .669)]:
    cerca(f"Temporal 80/20, {nom}: AUC", tt[nom][0], esp, 0.005)
chequeo("Hacia adelante en el tiempo NO se sostiene la ventaja de los ensambles: la logística no queda por debajo de HGB",
        tt["Regresión logística"][0] >= tt["HistGradientBoosting"][0])
cerca("Temporal 80/20, HGB: captura de suscriptores llamando al 20 % (el 63 % agrupado de la validación aleatoria es optimista)", tt["HistGradientBoosting"][1], .330, 0.01)
cs_ex = [x for x in c if x not in ("mes", "contacto")]; ns_ex = [x for x in n if x != "dia"]
tx = temporal(cs_ex, ns_ex)
for nom, esp in [("Regresión logística", .718), ("Random Forest", .704), ("HistGradientBoosting", .695)]:
    cerca(f"Temporal 80/20 estrictamente ex ante, {nom}: AUC", tx[nom][0], esp, 0.005)
# 8d. ventana creciente (6 tramos; entrena con los anteriores y prueba el siguiente), HGB
aristas = np.linspace(0, len(df), 7).astype(int); vent = []
for i_ in range(1, 6):
    a_ = df.iloc[:aristas[i_]]; b_ = df.iloc[aristas[i_]:aristas[i_ + 1]]
    vent.append(roc_auc_score(b_["y"], hgb_clas(c, n).fit(a_[c + n], a_["y"]).predict_proba(b_[c + n])[:, 1]))
for i_, esp in enumerate([.515, .532, .520, .657, .633]):
    cerca(f"Ventana creciente, HGB, paso {i_ + 1}: AUC", vent[i_], esp, 0.008)
chequeo("En los 3 primeros pasos el AUC hacia adelante es casi azar (< .56)", max(vent[:3]) < 0.56, str([round(v, 3) for v in vent]))
# 8e. validación por bloques contiguos (5 bloques; entrena con los otros 4), AUC y captura DENTRO del bloque
def por_bloques(cs, ns):
    p_al = cross_val_predict(hgb_clas(cs, ns), df[cs + ns], df["y"], cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    p_bl = cross_val_predict(hgb_clas(cs, ns), df[cs + ns], df["y"], cv=KFold(5, shuffle=False), method="predict_proba")[:, 1]
    def dentro(p_):
        a_, c_ = [], []
        for ix in np.array_split(np.arange(len(df)), 5):
            yy = df["y"].values[ix]; a_.append(roc_auc_score(yy, p_[ix])); o = yy[np.argsort(-p_[ix])]; c_.append(o[:int(len(o) * .2)].sum() / o.sum())
        return np.mean(a_), np.mean(c_)
    return roc_auc_score(df["y"], p_al), roc_auc_score(df["y"], p_bl), dentro(p_al), dentro(p_bl)
g_al, g_bl, (a_al, c_al), (a_bl, c_bl) = por_bloques(c, n)
cerca("Completo: AUC agrupado con validación cruzada aleatoria (.80)", g_al, .801, 0.003)
cerca("Completo: AUC agrupado con bloques contiguos (cae a .56)", g_bl, .562, 0.008)
cerca("Completo: AUC medio DENTRO de cada bloque, validación por bloques", a_bl, .593, 0.01)
cerca("Completo: captura llamando al 20 % DENTRO de cada bloque, validación por bloques (el 63 % agrupado es de otra cosa)", c_bl, .296, 0.015)
g_al2, g_bl2, (a_al2, c_al2), (a_bl2, c_bl2) = por_bloques(cs_ex, ns_ex)
cerca("Estrictamente ex ante: AUC medio dentro de cada bloque, validación por bloques", a_bl2, .600, 0.01)
cerca("Estrictamente ex ante: captura al 20 % dentro de cada bloque, validación por bloques", c_bl2, .288, 0.015)

if "--grilla" in sys.argv:
    print("\n=== 9. (opcional, --grilla) La búsqueda en grilla elige los valores por defecto ===")
    rg_c = m.entrenar_clasificacion(df, c, n, m.MODELOS_CLASIFICACION, ajustar=True)
    chequeo("GridSearch clasificación: C=0,1, hoja 10, lr 0,05 / prof. 6", rg_c["params"] == {"Regresión logística": {"C": 0.1}, "Random Forest": {"min_samples_leaf": 10}, "HistGradientBoosting": {"learning_rate": 0.05, "max_depth": 6}}, str(rg_c["params"]))
    rg_r = m.entrenar_regresion(df, c3, n3, m.MODELOS_REGRESION, ajustar=True)
    chequeo("GridSearch regresión: alpha 100, hoja 50, lr 0,05 / prof. 6", rg_r["params"] == {"Ridge (lineal)": {"alpha": 100}, "Random Forest": {"min_samples_leaf": 50}, "HistGradientBoosting": {"learning_rate": 0.05, "max_depth": 6}}, str(rg_r["params"]))

print(f"\n=== Resumen: {sum(RES)} OK, {len(RES) - sum(RES)} FALLA de {len(RES)} chequeos ===")
sys.exit(0 if all(RES) else 1)
