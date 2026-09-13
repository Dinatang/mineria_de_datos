from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURACIÓN DE FASTAPI
# ============================================================

app = FastAPI(
    title="API de Minería de Datos - Shopify",
    description=(
        "API para predecir el monto de un pedido utilizando "
        "el modelo Gradient Boosting entrenado en Google Colab."
    ),
    version="1.0.0"
)


# ============================================================
# 1. CARGA DEL DATASET
# ============================================================

url = (
    "https://raw.githubusercontent.com/"
    "Dinatang/mineria_de_datos/refs/heads/main/Shopify.csv"
)

df = pd.read_csv(url)


# ============================================================
# 2. TRATAMIENTO DE VALORES ATÍPICOS
# ============================================================

variable = "order_amount"

Q1 = df[variable].quantile(0.25)
Q3 = df[variable].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df_sin_atipicos = df[
    (df[variable] >= limite_inferior)
    & (df[variable] <= limite_superior)
].copy()


# ============================================================
# 3. TRANSFORMACIÓN LOGARÍTMICA
# ============================================================

df_sin_atipicos["order_amount_log"] = np.log1p(
    df_sin_atipicos["order_amount"]
)


# ============================================================
# 4. CODIFICACIÓN DEL MÉTODO DE PAGO
# ============================================================

df_sin_atipicos["payment_method"] = (
    df_sin_atipicos["payment_method"].astype("category")
)

df_sin_atipicos["payment_method_code"] = (
    df_sin_atipicos["payment_method"].cat.codes
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

df_sin_atipicos["created_at"] = pd.to_datetime(
    df_sin_atipicos["created_at"]
)

df_sin_atipicos["order_dayofweek"] = (
    df_sin_atipicos["created_at"].dt.dayofweek
)

df_sin_atipicos["order_month"] = (
    df_sin_atipicos["created_at"].dt.month
)


# ============================================================
# 6. VARIABLES PREDICTORAS
# ============================================================

X = df_sin_atipicos[
    [
        "total_items",
        "payment_method_code",
        "order_dayofweek",
        "order_month",
    ]
]


# ============================================================
# 7. CARGAR MODELO ENTRENADO EN GOOGLE COLAB
# ============================================================

modelo = joblib.load("modelo_gradient_boosting.joblib")

print("Modelo Gradient Boosting cargado correctamente.")


# ============================================================
# 8. MODELO DE DATOS PARA LA API
# ============================================================

class Pedido(BaseModel):
    total_items: int
    payment_method: str
    created_at: str


# ============================================================
# 9. PÁGINA WEB
# ============================================================

@app.get("/")
def pagina_web():
    return FileResponse("static/index.html")


# ============================================================
# 10. INFORMACIÓN DE LA API
# ============================================================

@app.get("/api")
def informacion_api():
    return {
        "mensaje": "API de predicción Shopify funcionando correctamente",
        "modelo": "Gradient Boosting",
        "origen_modelo": "Google Colab"
    }


# ============================================================
# 11. ENDPOINT DE PREDICCIÓN
# ============================================================

@app.post("/predict")
def predecir(pedido: Pedido):

    # Códigos utilizados durante el entrenamiento
    codigos_pago = {
        "cash": 0,
        "credit_card": 1,
        "debit": 2,
    }

    # Convertir método de pago a minúsculas
    metodo = pedido.payment_method.lower()

    # Validar método de pago
    if metodo not in codigos_pago:
        raise HTTPException(
            status_code=400,
            detail=(
                "Método de pago inválido. "
                "Use cash, credit_card o debit."
            ),
        )

    # Convertir fecha
    try:
        fecha = pd.to_datetime(pedido.created_at)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Formato de fecha inválido.",
        )

    # Validar cantidad de productos
    if pedido.total_items <= 0:
        raise HTTPException(
            status_code=400,
            detail="La cantidad de productos debe ser mayor que 0.",
        )

    # Crear los datos que necesita el modelo
    datos = pd.DataFrame(
        [
            {
                "total_items": pedido.total_items,
                "payment_method_code": codigos_pago[metodo],
                "order_dayofweek": fecha.dayofweek,
                "order_month": fecha.month,
            }
        ]
    )

    # Realizar predicción
    prediccion_log = modelo.predict(datos)[0]

    # Convertir de logaritmo a valor original
    prediccion_original = np.expm1(prediccion_log)

    # Devolver resultado
    return {
        "total_items": pedido.total_items,
        "payment_method": metodo,
        "created_at": pedido.created_at,
        "prediccion_order_amount_log": round(
            float(prediccion_log), 4
        ),
        "prediccion_order_amount": round(
            float(prediccion_original), 2
        ),
    }