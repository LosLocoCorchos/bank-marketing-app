"""App Streamlit: entrena y compara los modelos de clasificación y predicción de Bank Marketing.

Ejecutar:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import modelos as m
import produccion

st.set_page_config(page_title="¿A quién llamar primero? Bank Marketing", layout="wide")
st.title("¿A quién llamar primero?")
st.caption("Campaña telefónica de un banco para vender depósitos a plazo fijo (Bank Marketing, 45.211 llamadas, 2008–2010). "
           "Un modelo estima la probabilidad de que un cliente se suscriba (HistGradientBoosting) y otro cuántos contactos va a necesitar (Random Forest). "
           "Protocolo: 80/20, validación cruzada de 5 pliegues, prueba abierta una sola vez.")

# ------------------------------------------------------------------ datos
ARCHIVO_LOCAL = Path(__file__).with_name("dataset_con_titulos.csv")


@st.cache_data(show_spinner=False)
def leer_csv(origen):
    return pd.read_csv(origen)


def campo_numerico(contenedor, df, c, key):
    """Entero si la columna es entera en los datos (las 6 numéricas lo son); siempre acotado al rango observado."""
    if pd.api.types.is_integer_dtype(df[c]):
        mn, mx, med, paso = int(df[c].min()), int(df[c].max()), int(df[c].median()), 1
    else:
        mn, mx, med, paso = float(df[c].min()), float(df[c].max()), float(df[c].median()), None
    return contenedor.number_input(c, min_value=mn, max_value=mx, value=med, step=paso, key=key,
                                   help=m.DESCRIPCION.get(c, "") + f" Rango observado: {mn:g} a {mx:g}.")


def campo_categorico(contenedor, df, c, key):
    """Opciones ordenadas; por defecto, la categoría más frecuente (el cliente típico)."""
    opciones = sorted(df[c].unique())
    return contenedor.selectbox(c, opciones, index=opciones.index(df[c].mode().iloc[0]), key=key, help=m.DESCRIPCION.get(c))


def avisar_incoherencia(v):
    """En los datos, pdays = -1, previous = 0 y poutcome = unknown van siempre juntos (salvo 5 filas)."""
    if not all(k in v for k in ("pdays", "previous", "poutcome")):
        return
    sin_historia = (v["pdays"] == -1, v["previous"] == 0, v["poutcome"] == "unknown")
    if len(set(sin_historia)) > 1:
        st.warning("Combinación que casi no existe en los datos: `pdays` = -1 (nunca contactado antes), `previous` = 0 y `poutcome` = unknown "
                   "van siempre juntos. El resultado es una extrapolación poco confiable.")


with st.sidebar:
    st.header("Datos")
    subido = st.file_uploader("Subí dataset_con_titulos.csv", type="csv")
    st.header("Entrenamiento")
    ajustar = st.checkbox("Ajustar hiperparámetros (GridSearchCV)", value=False,
                          help="Apagado usa la configuración elegida en los Colab (rápido). "
                               "Prendido busca la mejor en una grilla con los mismos pliegues (tarda varios minutos).")
    st.caption("Pliegues: 5 · Semilla: 42 · Prueba: 20 %")

if subido is not None:
    bruto = leer_csv(subido)
elif ARCHIVO_LOCAL.exists():
    bruto = leer_csv(ARCHIVO_LOCAL)
else:
    st.info("Subí el archivo dataset_con_titulos.csv en la barra lateral para empezar.")
    st.stop()

try:
    df = m.preparar_df(bruto)
except ValueError as e:
    st.error(str(e))
    st.stop()

tab_uso, tab_datos, tab_clas, tab_pred = st.tabs(["Usar el modelo", "Datos", "Laboratorio: clasificación", "Laboratorio: predicción"])


# ------------------------------------------------------------------ usar el modelo (modelos serializados con joblib, sin reentrenar)
@st.cache_resource(show_spinner=False)
def cargar_modelo(tarea):
    return produccion.cargar(tarea)


def formulario_cliente(art, prefijo):
    """Un campo por variable, construido con los rangos observados guardados junto al modelo."""
    cols = st.columns(3)
    valores = {}
    for i, c in enumerate(art["cats"]):
        r = art["rangos"][c]
        valores[c] = cols[i % 3].selectbox(c, r["opciones"], index=r["opciones"].index(r["defecto"]),
                                           key=f"{prefijo}_{c}", help=m.DESCRIPCION.get(c))
    for i, c in enumerate(art["nums"]):
        r = art["rangos"][c]
        valores[c] = cols[i % 3].number_input(c, min_value=r["min"], max_value=r["max"], value=r["defecto"], step=1, key=f"{prefijo}_{c}",
                                              help=m.DESCRIPCION.get(c, "") + f" Rango observado: {r['min']} a {r['max']}.")
    return valores


with tab_uso:
    art_c, art_r = cargar_modelo("clasificacion"), cargar_modelo("regresion")
    if art_c is None or art_r is None:
        st.info("Faltan los modelos guardados. Corré `python entrenar_y_guardar.py` y volvé a abrir la app.")
    else:
        u1, u2 = st.tabs(["¿Conviene llamar a este cliente?", "¿Cuántos contactos va a necesitar?"])
        with u1:
            st.caption(f"Modelo: {art_c['nombre']}, entrenado con el 80 % de los datos. Escenario: rankear clientes antes de llamar; "
                       "se asume que se conocen la fecha y el medio planeados de la llamada (`mes`, `dia`, `contacto`).")
            with st.form("uso_c"):
                v_c = formulario_cliente(art_c, "uc")
                enviado_c = st.form_submit_button("Calcular probabilidad", type="primary")
            if enviado_c:
                avisar_incoherencia(v_c)
                r = produccion.clasificar(art_c, v_c)
                k1, k2, k3 = st.columns(3)
                k1.metric("Probabilidad de suscripción", m.pct(r["probabilidad"]))
                k2.metric("Umbral del modelo", m.pct(r["umbral"]))
                k3.metric("Posición en la lista", f"Entre el {r['porcentaje_superior'] * 100:.0f} % mejor puntuado")
                if r["marcar"]:
                    st.success(f"**Conviene llamarlo.** En los datos, el {m.pct(r['tasa_decil'], 0)} de los clientes de su tramo se suscribió.")
                else:
                    st.warning(f"**No entra en la lista de llamadas.** En los datos, el {m.pct(r['tasa_decil'], 0)} de los clientes de su tramo se suscribió.")
                st.pyplot(produccion.fig_decil_cliente(art_c, r["decil"]))
            mt = art_c["metricas"]
            st.caption(f"Confiabilidad: AUC {m.num(mt['auc_cv'])} en validación cruzada y {m.num(mt['AUC'])} en la prueba; precisión {m.pct(mt['Precisión'], 0)} y "
                       f"exhaustividad {m.pct(mt['Exhaustividad'], 0)} con el umbral. Vale dentro del período 2008–2010: para una campaña nueva "
                       "esperá un AUC entre 0,60 y 0,70. El resultado ordena clientes, no promete que uno en particular se suscriba.")
        with u2:
            st.caption(f"Modelo: {art_r['nombre']}, entrenado con el 80 % de los datos, sin `mes`, `dia` ni `duracion`.")
            with st.form("uso_r"):
                v_r = formulario_cliente(art_r, "ur")
                enviado_r = st.form_submit_button("Predecir contactos", type="primary")
            if enviado_r:
                avisar_incoherencia(v_r)
                pred = produccion.predecir_contactos(art_r, v_r)
                st.metric("Contactos esperados", m.num(pred, 1))
                st.pyplot(produccion.fig_distribucion(art_r, pred))
            mt = art_r["metricas"]
            st.caption(f"Confiabilidad baja: R² {m.num(mt['r2_prueba'], 2)} y RMSE {m.num(mt['rmse_prueba'], 2)} en la prueba, contra "
                       f"{m.num(mt['rmse_promedio_prueba'], 2)} de predecir siempre el promedio ({m.num(art_r['promedio'], 1)}). "
                       "Con lo que se sabe antes de llamar, la cantidad de contactos casi no se puede anticipar: tomalo como orientación.")

# ------------------------------------------------------------------ datos
with tab_datos:
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(df):,}".replace(",", "."))
    c2.metric("Suscripciones", f"{int(df['y'].sum()):,}".replace(",", "."))
    c3.metric("Tasa de suscripción", m.pct(df["y"].mean()))
    st.dataframe(bruto.head(20))
    with st.expander("¿Qué es cada columna?", expanded=False):
        st.dataframe(m.tabla_glosario())
        st.caption("Solo las primeras 8 describen al cliente. `contacto`, `dia`, `mes`, `duracion` y `campaign` describen el contacto "
                   "actual de la campaña, y `pdays`, `previous` y `poutcome` la campaña anterior.")
    st.write("**Suscripción por mes** (cuidado: mezcla años distintos, no prueba un efecto del mes del año)")
    por_mes = df.groupby("mes")["y"].agg(["mean", "size"]).rename(columns={"mean": "Tasa", "size": "Clientes"})
    orden = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    por_mes = por_mes.reindex([o for o in orden if o in por_mes.index])
    por_mes["Tasa"] = por_mes["Tasa"].map(m.pct)
    st.dataframe(por_mes.T)
    with st.expander("Mismo cuadro por año inferido (el archivo está ordenado por fecha)"):
        try:
            tabla_anio, por_anio = m.tasa_por_mes_y_anio(df)
            st.dataframe(tabla_anio.T)
            st.caption("Año inferido por los saltos de mes (año 1 = el primero del archivo). Tasa de suscripción de cada año: "
                       + "; ".join(f"año {a}: {m.pct(r['mean'])} ({int(r['size']):,} filas)".replace(",", ".") for a, r in por_anio.iterrows())
                       + ". La tasa por mes refleja en qué año cayó cada mes, no una estacionalidad.")
        except Exception as e:
            st.info(f"No se pudo inferir el año: {e}")


def barra_progreso():
    barra = st.progress(0.0, text="Empezando…")
    return barra, lambda f, t: barra.progress(f, text=t)


# ------------------------------------------------------------------ clasificación
with tab_clas:
    st.subheader("¿Este cliente va a suscribirse al depósito?")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        modelos_c = st.multiselect("Modelos", m.MODELOS_CLASIFICACION, default=m.MODELOS_CLASIFICACION, key="mc")
    with col_b:
        usar_campaign = st.checkbox("Incluir `campaign` (contactos de la campaña, incluido el último: fuga estricta)", value=False, key="cc")
        usar_duracion = st.checkbox("Incluir `duracion` (fuga de información: solo se conoce al terminar la llamada)",
                                    value=False, key="cd")
    st.caption("Escenario: rankear clientes antes de llamar. Se asume que se conocen la fecha y el medio planeados de la llamada "
               "(`mes`, `dia`, `contacto`); sin ellos el AUC baja a ≈ 0,74.")
    if usar_duracion:
        st.warning("`duracion` se conoce recién cuando termina la llamada, o sea cuando ya se sabe si compró. "
                   "El modelo se ve mucho mejor de lo que sería en la práctica: no sirve para decidir a quién llamar.")

    if st.button("Entrenar clasificación", type="primary", disabled=not modelos_c):
        cats, nums = m.columnas_clasificacion(usar_campaign, usar_duracion)
        barra, cb = barra_progreso()
        st.session_state["clas"] = m.entrenar_clasificacion(df, cats, nums, modelos_c, ajustar=ajustar, progreso=cb)
        barra.empty()

    rc = st.session_state.get("clas")
    if rc is None:
        st.info("Elegí los modelos y apretá **Entrenar clasificación**.")
    else:
        st.success(f"Mejor en validación cruzada (AUC): **{rc['elegido']}**")
        d1, d2 = st.columns(2)
        d1.download_button("Descargar resultados (.txt)", m.exportar_txt(rc), file_name="resultados_clasificacion.txt", mime="text/plain", key="dt_clasificacion")
        d2.download_button("Descargar tablas (.zip de CSV)", m.exportar_zip(rc), file_name="resultados_clasificacion.zip", mime="application/zip", key="dz_clasificacion")
        st.write("**Validación cruzada, 5 pliegues** (sobre el 80 % de entrenamiento): esta es la cifra principal; la prueba es una sola partición")
        st.dataframe(m.tabla_cv_clasificacion(rc))
        if len(rc["nombres"]) > 1:
            st.write("**Comparación de a pares, pliegue a pliegue** (diferencia de AUC)")
            st.dataframe(m.comparar_pliegues(rc))
            st.caption("“Se distinguen” = la diferencia media supera 2 desvíos y tiene el mismo signo en todos los pliegues.")
        st.write("**Prueba (20 % que no se tocó)**, umbral elegido en validación cruzada para maximizar F1")
        st.caption("El archivo parece ordenado en el tiempo y la partición es aleatoria: los resultados valen dentro del período. "
                   "Notas: HGB usa early stopping automático; ± con ddof=0.")
        st.dataframe(m.tabla_prueba_clasificacion(rc))
        st.pyplot(m.fig_curvas(rc))

        elegido = st.selectbox("Ver en detalle", rc["nombres"], index=rc["nombres"].index(rc["elegido"]), key="det_c")
        g1, g2 = st.columns(2)
        g1.pyplot(m.fig_matriz(rc, elegido))
        g2.pyplot(m.fig_deciles(rc, elegido))
        st.write(m.texto_matriz(rc, elegido))
        st.write("**¿A cuánta gente hay que llamar?** (% de los suscriptores que se captan según la parte de la lista que se llama)")
        st.dataframe(m.captura_por_lista(rc))
        st.pyplot(m.fig_calibracion(rc))
        st.caption("Ojo: esta captura y el AUC de arriba son agrupados sobre todo el período (mezclan épocas con tasas muy distintas). "
                   "Para ordenar la lista de una campaña nueva, mirá la validación temporal de abajo.")
        with st.expander("Validación temporal (entrena con el primer 80 % del archivo y prueba con el último 20 %)"):
            if st.button("Calcular validación temporal", key="temp_c"):
                with st.spinner("Entrenando con el pasado…"):
                    st.dataframe(m.tabla_temporal_clasificacion(df, rc["cats"], rc["nums"], rc["nombres"], rc["semilla"]))
                st.caption("Supone que el archivo está ordenado por fecha (lo dice el diccionario). Es una de las pruebas hacia adelante (la más favorable de las que se probaron): para una campaña nueva esperá entre 0,60 y 0,70.")

        with st.expander("Importancia de las variables (permutación, tarda un poco)"):
            if st.button("Calcular importancias", key="imp_c"):
                with st.spinner("Barajando variables…"):
                    imp = m.importancia_permutacion(rc)
                st.pyplot(m.fig_importancia(imp, "AUC"))
                st.dataframe(imp.round(4))

        with st.expander("Probar con un cliente"):
            with st.form("cliente_c"):
                st.caption("`mes`, `dia`, `contacto` y `campaign` no son atributos del cliente: describen cuándo y cómo se lo contacta. "
                           "Solo aparecen los que el modelo usa; los campos numéricos se limitan al rango observado en los datos (fuera de ese rango el modelo no aprendió nada). "
                           "Pasá el mouse por el `?` de cada campo para ver qué es.")
                cols = st.columns(3)
                valores = {}
                for i, c in enumerate(rc["cats"]):
                    valores[c] = campo_categorico(cols[i % 3], df, c, f"c_{c}")
                for i, c in enumerate(rc["nums"]):
                    valores[c] = campo_numerico(cols[i % 3], df, c, f"n_{c}")
                modelo_cli = st.selectbox("Modelo", rc["nombres"], index=rc["nombres"].index(rc["elegido"]), key="mod_cli_c")
                enviado = st.form_submit_button("Calcular probabilidad")
            if enviado:
                r = m.predecir_cliente(rc, modelo_cli, valores)
                avisar_incoherencia(valores)
                st.metric("Probabilidad de suscripción", m.pct(r["probabilidad"]))
                st.write(f"Umbral del modelo: {m.pct(r['umbral'])}. "
                         + ("**Conviene llamarlo.**" if r["marcar"] else "**No entra en la lista de llamadas.**"))

# ------------------------------------------------------------------ predicción
with tab_pred:
    st.subheader("¿Cuántos contactos va a necesitar un cliente (`campaign`)?")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        modelos_r = st.multiselect("Modelos", m.MODELOS_REGRESION, default=m.MODELOS_REGRESION, key="mr")
    with col_b:
        usar_fecha = st.checkbox("Incluir `mes` y `dia` (fecha del último contacto: en parte consecuencia de cuántas veces se contactó)", value=False, key="rf")
    if st.button("Entrenar predicción", type="primary", disabled=not modelos_r):
        cats, nums = m.columnas_regresion(usar_fecha)
        barra, cb = barra_progreso()
        st.session_state["reg"] = m.entrenar_regresion(df, cats, nums, modelos_r, ajustar=ajustar, progreso=cb)
        barra.empty()

    rr = st.session_state.get("reg")
    if rr is None:
        st.info("Elegí los modelos y apretá **Entrenar predicción**.")
    else:
        st.success(f"Menor error en validación cruzada (RMSE): **{rr['elegido']}**")
        d1, d2 = st.columns(2)
        d1.download_button("Descargar resultados (.txt)", m.exportar_txt(rr), file_name="resultados_prediccion.txt", mime="text/plain", key="dt_prediccion")
        d2.download_button("Descargar tablas (.zip de CSV)", m.exportar_zip(rr), file_name="resultados_prediccion.zip", mime="application/zip", key="dz_prediccion")
        st.write("**Validación cruzada, 5 pliegues**")
        st.dataframe(m.tabla_cv_regresion(rr))
        st.caption("Los modelos optimizan el error cuadrático (predicen la media) y la mediana optimiza el error absoluto: por eso la mediana "
                   "tiene mejor MAE. Los modelos se comparan por RMSE. ± = desvío entre pliegues (ddof=0).")
        if len(rr["nombres"]) > 1:
            st.write("**Comparación de a pares, pliegue a pliegue** (diferencia de RMSE)")
            st.dataframe(m.comparar_pliegues(rr))
            st.caption("Si dice “empate técnico”, el elegido es solo ligeramente mejor.")
        st.write("**Prueba (20 % que no se tocó)**")
        st.dataframe(m.tabla_prueba_regresion(rr))
        g1, g2 = st.columns(2)
        g1.pyplot(m.fig_rmse_cv(rr))
        g2.pyplot(m.fig_real_vs_predicho(rr))
        elegido = st.selectbox("Ver errores de", rr["nombres"], index=rr["nombres"].index(rr["elegido"]), key="det_r")
        st.pyplot(m.fig_errores(rr, elegido))
        st.write("**En la práctica** (contra la mediana ficticia: “siempre 2”, que da un entero: por eso se muestra también la predicción redondeada)")
        st.dataframe(m.tabla_practica_regresion(rr))

        if "Random Forest" in rr["modelos"] and usar_fecha:
            with st.expander("¿Perfil o calendario? (reentrena el Random Forest dos veces)"):
                if st.button("Calcular", key="pc"):
                    with st.spinner("Reentrenando…"):
                        st.dataframe(m.perfil_vs_calendario(rr))
                    st.caption("Si la fecha sola explica casi todo y el perfil casi nada, `campaign` depende más del calendario de la campaña que del cliente.")

        with st.expander("Importancia de las variables (permutación, tarda un poco)"):
            if st.button("Calcular importancias", key="imp_r"):
                with st.spinner("Barajando variables…"):
                    imp = m.importancia_permutacion(rr)
                st.pyplot(m.fig_importancia(imp, "R²"))
                st.dataframe(imp.round(4))

        with st.expander("Probar con un cliente"):
            with st.form("cliente_r"):
                st.caption("`mes`, `dia`, `contacto` y `campaign` no son atributos del cliente: describen cuándo y cómo se lo contacta. "
                           "Solo aparecen los que el modelo usa; los campos numéricos se limitan al rango observado en los datos (fuera de ese rango el modelo no aprendió nada). "
                           "Pasá el mouse por el `?` de cada campo para ver qué es.")
                cols = st.columns(3)
                valores = {}
                for i, c in enumerate(rr["cats"]):
                    valores[c] = campo_categorico(cols[i % 3], df, c, f"rc_{c}")
                for i, c in enumerate(rr["nums"]):
                    valores[c] = campo_numerico(cols[i % 3], df, c, f"rn_{c}")
                modelo_cli = st.selectbox("Modelo", rr["nombres"], index=rr["nombres"].index(rr["elegido"]), key="mod_cli_r")
                enviado = st.form_submit_button("Predecir contactos")
            if enviado:
                r = m.predecir_cliente(rr, modelo_cli, valores)
                avisar_incoherencia(valores)
                st.metric("Contactos esperados", m.num(r["contactos"], 1))
