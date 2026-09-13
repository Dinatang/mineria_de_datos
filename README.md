# API de Minería de Datos - Shopify

## Proyecto Académico

Este proyecto corresponde a la asignatura de **Minería de Datos** y tiene como finalidad desarrollar un sistema de predicción basado en datos de pedidos de Shopify.

El proyecto integra tres componentes principales:

1. Un **Notebook de Google Colab** donde se realizó el análisis y entrenamiento de los modelos.
2. Una **API desarrollada con FastAPI** para realizar predicciones.
3. Una **página web** que permite al usuario ingresar los datos de un pedido y obtener el monto estimado.

---

## Integrantes

- **Ronnal Eulogio Montoya Zamora**
- **Dina María Tanguila Vargas**
- **Anthony Sebastián Rubio Flores**

---

## Objetivo del proyecto

Desarrollar un modelo de Machine Learning capaz de estimar el monto de un pedido de Shopify a partir de algunas características del pedido.

El modelo seleccionado fue **Gradient Boosting Regressor**, debido a que presentó el mejor desempeño entre los modelos evaluados.

El modelo fue entrenado en **Google Colab**, exportado mediante Joblib y posteriormente integrado dentro de una API desarrollada con FastAPI.

---

# Estructura del proyecto

```text
Mineria-Datos-shopify-api/
│
├── Dataset_shopify (3).ipynb
├── main.py
├── modelo_gradient_boosting.joblib
├── requirements.txt
├── README.md
│
└── static/
    └── index.html