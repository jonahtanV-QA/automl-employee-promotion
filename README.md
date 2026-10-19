# Predicción de candidatos elegibles para promoción laboral (AutoML)

> **EN (summary):** Binary classification of employee promotions with two AutoML platforms (H2O AutoML and PyCaret), plus automated EDA (YData Profiling, SweetViz). Both reach ~93 % accuracy against a 91 % "never promote" baseline, but only 33–40 % recall on the promoted class — the notebook discusses why accuracy alone is misleading here.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonahtanV-QA/automl-employee-promotion/blob/main/Reto_AutoML_Promocion_Valdez_Jonahtan.ipynb)

## Objetivo

Predecir `is_promoted` (si un empleado fue promovido) a partir de datos de RR. HH. y comparar dos plataformas de AutoML con configuración equivalente.

## Qué incluye el notebook

1. **EDA automatizado:** reporte de YData Profiling y comparación por género con SweetViz (`compare_intra`, objetivo `is_promoted`).
2. **Preparación de datos:** eliminación de faltantes, partición 80:20 y conversión a `H2OFrame`.
3. **H2O AutoML:** `seed=1`, 5 minutos, `balance_classes=True`, métrica de ordenamiento AUC.
4. **PyCaret:** `session_id=1`, `fix_imbalance=True` (SMOTE), `compare_models` con presupuesto de 5 minutos, `tune_model` optimizando accuracy y evaluación con `evaluate_model`.
5. **Interpretación** de cada resultado y conclusiones con limitaciones.

## Resultados (conjunto de prueba)

| | H2O AutoML | PyCaret |
|---|---|---|
| Modelo final | Stacked Ensemble | Random Forest |
| Accuracy | 0.9361 | 0.9317 |
| Baseline "nunca promover" | 0.9123 | 0.9120 |
| Precisión (clase promovido) | 0.757 | 0.755 |
| Recall (clase promovido) | 0.399 | 0.331 |
| AUC | 0.821 | 0.783 |

> Los conjuntos de prueba son distintos (9,681 y 9,732 empleados), por lo que las dos plataformas **no son estrictamente comparables** entre sí.

## Lo que aprendí

- **El accuracy engaña con clases desbalanceadas.** Solo ≈ 8.7 % de los empleados fue promovido: siempre predecir "no promovido" ya da ≈ 91 %. La mejora real es de unos 2 puntos.
- **El modelo es conservador:** cuando dice "promovido" acierta ≈ 75 % de las veces, pero detecta solo 33–40 % de los promovidos reales.
- **El ajuste de hiperparámetros no siempre mejora.** En PyCaret, `tune_model` no superó al Random Forest original y se conservó este último.
- **La importancia de variables hay que leerla con cuidado.** En el Random Forest, `gender` aparece en cuarto lugar; antes de usar un modelo así para decidir sobre personas habría que revisar la equidad.
- **Qué probaría después:** optimizar F1 o AUC-PR, ajustar el umbral de decisión y tratar a los empleados de primer año (sin calificación previa) como categoría propia en lugar de descartarlos.

## Cómo ejecutarlo

1. Abre el notebook en Colab con el botón de arriba.
2. Selecciona la versión de entorno **2026.07** (PyCaret requiere Python 3.12).
3. Sube `EmployeePromotion.csv` al panel de archivos de Colab (ver *Datos*).
4. Ejecuta todas las celdas. La corrida completa toma unos 30 minutos (H2O 5 min, `compare_models` 5 min, `tune_model` ≈ 11 min).

## Datos

El archivo `EmployeePromotion.csv` **no se incluye en este repositorio**. Colócalo en el directorio de trabajo de Colab antes de ejecutar.

## Herramientas

Python · H2O AutoML · PyCaret 3.3.2 · YData Profiling · SweetViz · pandas · scikit-learn

## Autor

**Jonahtan Valdez** — QA / Test Lead · AI Technologies Specialist (Tecnológico de Monterrey – The Learning Gate)
