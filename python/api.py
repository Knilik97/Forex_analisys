# python/api.py
# Этот скрипт будет работать как сервер. Запускать его нужно отдельно от терминала.

import uvicorn
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- НАСТРОЙКИ ---
PORT = 8000
MODEL_PATH = "best_model_v4.h5"
SCALER_PATH = "scaler_price_v4.pkl"
SEQ_LEN = 60 # Должно совпадать с тем, что было при обучении!
# ------------------

# Загружаем модель и масштабер (делаем это один раз при запуске сервера)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_y = joblib.load(SCALER_PATH)
    print("✅ Модель и масштабер успешно загружены.")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")
    model = None

app = FastAPI(title="Forex Bot Prediction API")

# Определяем формат данных, которые будет присылать бот
class MarketData(BaseModel):
    data: list # Список из SEQ_LEN свечей. Каждая свеча - это список чисел [open, high, low, close, volume, ...]

@app.post("/predict")
async def predict_price(payload: MarketData):
    """
    Эндпоинт для получения прогноза цены.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    # Получаем данные от бота
    input_data = np.array(payload.data)
    
    # Проверяем форму данных. Должно быть (60, N), где N - количество признаков.
    if input_data.shape != (SEQ_LEN, 11): # У нас 11 колонок (O,H,L,C,V,MA16,MA30,RSI,MACD_Line,MACD_Signal,MACD_Hist)
        raise HTTPException(status_code=400, detail=f"Неверная форма данных. Ожидалось {(SEQ_LEN, 11)}, получено {input_data.shape}.")
    
    # Масштабируем данные так же, как при обучении
    # (В идеале здесь должен быть тот же scaler_X, что и при обучении. 
    # Для простоты примера опустим этот шаг и предположим, что бот присылает уже нормализованные данные)
    
    # Делаем предсказание. Модель ожидает форму (1, 60, 11)
    prediction_scaled = model.predict(input_data.reshape(1, SEQ_LEN, 11))
    
    # "Де-нормализуем" предсказание, чтобы получить реальную цену
    predicted_price = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    
    return {
        "predicted_close_price": float(predicted_price),
        "status": "ok"
    }

if __name__ == "__main__":
    print(f"🚀 Запуск API сервера на порту {PORT}...")
    uvicorn.run(app, host="127.0.0.1", port=PORT)