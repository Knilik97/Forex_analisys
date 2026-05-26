import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
from collections import deque
import MetaTrader5 as mt5

# --- НАСТРОЙКИ ---
SYMBOL = "EURUSD"  # Одна пара, но на разных таймфреймах
TF_CONFIGS = [
    {
        "tf": mt5.TIMEFRAME_M30,       # Таймфрейм 30 минут
        "seq_len": 96,                  # Длина последовательности (4 дня)
        "model_file": "python/train_model_v7_EURUSD/EURUSD_M30/model_v7_EURUSD_30m_full.h5",
        "scaler_features_file": "python/train_model_v7_EURUSD/EURUSD_M30/scaler_features_v7_30m_EURUSD_full.pkl",
        "scaler_price_file": "python/train_model_v7_EURUSD/EURUSD_M30/scaler_price_v7_30m_EURUSD_full.pkl",
        "log_file": "hystory_log/prediction_log_30m_EURUSD_full.txt"
    },
    {
        "tf": mt5.TIMEFRAME_H1,        # Таймфрейм 1 час
        "seq_len": 240,                 # Длина последовательности (10 дней)
        "model_file": "python/train_model_v7_EURUSD/EURUSD_H1/model_v7.3_EURUSD_240.h5",
        "scaler_features_file": "python/train_model_v7_EURUSD/EURUSD_H1/scaler_features_v7.3_EURUSD.pkl",
        "scaler_price_file": "python/train_model_v7_EURUSD/EURUSD_H1/scaler_price_v7.3_EURUSD.pkl",
        "log_file": "hystory_log/prediction_log_H1_EURUSD.txt"
    },
    {
        "tf": mt5.TIMEFRAME_H4,        # Таймфрейм 4 часа
        "seq_len": 48,                  # Длина последовательности (4 суток)
        "model_file": "python/train_model_v7_EURUSD/EURUSD_H4/model_v7_EURUSD_4H.h5",
        "scaler_features_file": "python/train_model_v7_EURUSD/EURUSD_H4/scaler_features_v7_H4_EURUSD_full.pkl",
        "scaler_price_file": "python/train_model_v7_EURUSD/EURUSD_H4/scaler_price_v7_H4_EURUSD_full.pkl",
        "log_file": "hystory_log/prediction_log_H4_EURUSD.txt"
    }
]

NUM_CANDLES_TO_DOWNLOAD = 300  # Скачиваем с запасом

# Вспомогательные функции
def calculate_indicators(df):
    """Расчет технических индикаторов."""
    close_prices = df['close']
    
    df['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()
    
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    EMA_12 = close_prices.ewm(span=12, adjust=False).mean()
    EMA_26 = close_prices.ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA_12 - EMA_26 
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    return df

def generate_forecast(config):
    """Генерация прогноза для одного таймфрейма."""
    # 1. ПОДКЛЮЧЕНИЕ И СКАЧИВАНИЕ ДАННЫХ
    if not mt5.initialize():
        raise ConnectionError(f"Ошибка подключения к терминалу для {SYMBOL}/{config['tf']}.")
        
    rates = mt5.copy_rates_from_pos(SYMBOL, config["tf"], 0, NUM_CANDLES_TO_DOWNLOAD)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"Ошибка: Не удалось получить котировки для {SYMBOL}/{config['tf']}.")
        
    # 2. СОЗДАНИЕ DATAFRAME И РАСЧЕТ ИНДИКАТОРОВ
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Проверка на достаточное количество данных после очистки NaN
    if len(df) < config['seq_len'] + 10: # +10 на случай NaN от индикаторов
        raise ValueError(f"Мало данных ({len(df)}) для {SYMBOL}/{config['tf']}.")
    
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    if len(df) < config['seq_len']:
        raise ValueError(f"После очистки от NaN осталось всего {len(df)} строк для {SYMBOL}/{config['tf']}.")

    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
    features_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Histogram']
    
    try:
        scaler_X = joblib.load(config["scaler_features_file"])
        scaler_y = joblib.load(config["scaler_price_file"])
        model = tf.keras.models.load_model(config["model_file"], compile=False)
    except Exception as e:
        raise FileNotFoundError(f"Ошибка загрузки моделей для {SYMBOL}/{config['tf']}: {e}")

    X = df[features_cols].values 
    data_scaled = scaler_X.transform(X)
    input_data = data_scaled[-config['seq_len']:].reshape(1, config['seq_len'], len(features_cols))

    # 4. ПРОГНОЗ
    prediction_scaled = model.predict(input_data)
    
    predicted_price_normalized = prediction_scaled.flatten()[0]
    predicted_price_real = scaler_y.inverse_transform(np.array([[predicted_price_normalized]]))[0][0]
    
    current_price_real = df['close'].iloc[-1]
    price_diff = predicted_price_real - current_price_real

    # Определение сигнала
    if price_diff > 0.00005:
        signal_text = "🟢 BUY"
        signal_color = "#00FF00" # Зеленый
    elif price_diff < -0.00005:
        signal_text = "🔴 SELL"
        signal_color = "#FF0000" # Красный
    else:
        signal_text = "🟡 HOLD"
        signal_color = "#FFFF00" # Желтый

    # Возвращаем результат
    return {
        "current_price": current_price_real,
        "predicted_price": predicted_price_real,
        "price_diff_pips": int(price_diff * 100000),
        "signal": signal_text,
        "signal_color": signal_color,
        "log_file": config["log_file"]
    }

# Основной поток приложения
def main():
    st.title("📊 Прогноз EUR/USD на разных таймфреймах")

    # Кнопка для обновления данных
    if st.button("🔄🏼 Обновить данные"):
        st.info("🚀 Обновление данных...")

        # Генерируем прогнозы для всех таймфреймов
        for config in TF_CONFIGS:
            try:
                forecast_result = generate_forecast(config)
            except Exception as e:
                st.error(f"❌ Ошибка при получении прогноза для {SYMBOL}/{config['tf']}: {e}")
                continue

            # Печать красивого отчета
            st.markdown("=" * 30)
            st.markdown(f"📊 **ПРОГНОЗ ДЛЯ ПАРЫ:** {SYMBOL} ({config['tf']})")
            st.markdown("=" * 30)
            st.markdown(f"- 🕒 **Текущая цена (последняя свеча):** `{forecast_result['current_price']:.5f}`")
            st.markdown(f"- 🔮 **Прогноз цены через 1 свечу:** `{forecast_result['predicted_price']:.5f}`")
            st.markdown(f"- **Разница в пунктах:** `{forecast_result['price_diff_pips']:+d}`")
            st.markdown(f"- **Сигнал:** <span style='color:{forecast_result['signal_color']}'><b>{forecast_result['signal']}</b></span>", unsafe_allow_html=True)

            # Логирование в файл
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"[{timestamp}] | "
                f"Цена: {forecast_result['current_price']:.5f} | "
                f"Прогноз: {forecast_result['predicted_price']:.5f} | "
                f"Сигнал: {forecast_result['signal']} | "
                f"Пипсы: {forecast_result['price_diff_pips']:+d}\n"
            )

            try:
                with open(forecast_result["log_file"], "a", encoding="utf-8") as f:
                    f.write(log_entry)
                st.success(f"✅ Прогноз для {SYMBOL}/{config['tf']} успешно сохранен в файл: {forecast_result['log_file']}")
            except Exception as e:
                st.error(f"❌ Ошибка записи в файл: {e}")

if __name__ == "__main__":
    main()