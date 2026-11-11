# ¿A quién llamar primero? Bank Marketing

Aplicación web (Streamlit) que usa dos modelos de Aprendizaje Automático sobre el conjunto *Bank Marketing* (campaña telefónica de un banco portugués para vender depósitos a plazo fijo; 45.211 llamadas, 2008–2010):

| Pregunta | Modelo | Desempeño (prueba, 20 % no tocado) |
|---|---|---|
| ¿Conviene llamar a este cliente? (clasificación) | HistGradientBoosting | AUC 0,80; precisión 0,46; exhaustividad 0,54 |
| ¿Cuántos contactos va a necesitar? (predicción de `campaign`) | Random Forest | RMSE 2,97 (predecir el promedio: 3,01); R² 0,03 |

**App pública:** https://bank-marketing-unraf.streamlit.app/

## Qué hace
- **Usar el modelo:** formulario con un campo por variable, limitado al rango observado en los datos (enteros donde corresponde, valores por defecto = mediana o moda). Devuelve la probabilidad de suscripción, el umbral del modelo, la recomendación, la posición del cliente en la lista y la tasa histórica de su tramo; y los contactos esperados con su ubicación en la distribución. Avisa cuando `pdays`, `previous` y `poutcome` son incoherentes entre sí.
- **Datos:** descripción del conjunto, glosario y tasa de suscripción por mes y por año inferido.
- **Laboratorio:** reentrena y compara los tres algoritmos de cada tarea con el mismo protocolo (80/20, validación cruzada de 5 pliegues, semilla 42, prueba abierta una sola vez).

## Escenario y variables
Se asume el escenario "rankear antes de llamar": se conocen el perfil del cliente, su historial previo y la fecha y el medio planeados. Por eso la clasificación **no** usa `duracion` (solo se conoce al terminar la llamada) ni `campaign` (incluye el último contacto); la predicción de `campaign` tampoco usa `mes`, `dia` ni `duracion`.

## Limitaciones
- El AUC de 0,80 vale **dentro del período 2008–2010**. Entrenando con el primer 80 % del archivo y probando con el último 20 % baja a 0,67–0,70; sin `mes`, `dia` ni `contacto` queda en ≈ 0,74.
- `campaign` casi no se puede predecir con lo que se sabe antes de llamar (R² ≈ 0,03).
- Los datos son de una sola institución y las importancias miden asociación, no causalidad.

## Correr en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Los modelos ya están guardados en `modelos_guardados/` (formato joblib) y la app los carga sin reentrenar. Para regenerarlos (hace falta si cambia la versión de scikit-learn):
```bash
python entrenar_y_guardar.py
```

## Archivos
- `app.py`: interfaz.
- `modelos.py`: preprocesamiento, modelos, validación cruzada, tablas y gráficos del laboratorio.
- `produccion.py`: construcción, serialización y carga de los modelos finales.
- `entrenar_y_guardar.py`: entrena y guarda los modelos con joblib.
- `modelos_guardados/`: `clasificacion.joblib` y `regresion.joblib` (modelo, umbral, rangos de entrada y datos de referencia).
- `dataset_con_titulos.csv`: datos con encabezados.
- `validar.py`: 96 chequeos que reproducen los números del informe (`python validar.py`).
- `grillas_ampliadas.py`: comprueba que ampliar las grillas de hiperparámetros no mejora la prueba.

## Datos y referencias
S. Moro, P. Rita y P. Cortez, "Bank Marketing", UCI Machine Learning Repository, 2014, doi: 10.24432/C5K306 (licencia CC BY 4.0).
S. Moro, P. Cortez y P. Rita, "A data-driven approach to predict the success of bank telemarketing", *Decision Support Systems*, 62, 22–31, 2014.

Autores: Elías Farinoli, Guillermo Fioramonte, Marcos Galla y José Santagostino. Universidad Nacional de Rafaela (UNRaf), Ingeniería en Computación.
