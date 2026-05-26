import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import tensorflow as tf
import time
import os

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

# Функции остались теми же, просто добавили расчет индикаторов
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
    print(f"\n🚀 Прогноз для пары {SYMBOL} на таймфрейме {config['tf']}...")
    
    start_time = time.time()  # Время выполнения для отдельного таймфрейма

    # 1. ПОДКЛЮЧЕНИЕ И СКАЧИВАНИЕ ДАННЫХ
    if not mt5.initialize():
        print(f"❌ Ошибка подключения к терминалу для {SYMBOL}/{config['tf']}.")
        return
        
    rates = mt5.copy_rates_from_pos(SYMBOL, config["tf"], 0, NUM_CANDLES_TO_DOWNLOAD)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"❌ Ошибка: Не удалось получить котировки для {SYMBOL}/{config['tf']}.")
        return
        
    # 2. СОЗДАНИЕ DATAFRAME И РАСЧЕТ ИНДИКАТОРОВ
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Проверка на достаточное количество данных после очистки NaN
    if len(df) < config['seq_len'] + 10: # +10 на случай NaN от индикаторов
        print(f"⚠️ Внимание: Получено мало данных ({len(df)}) для {SYMBOL}/{config['tf']}.")
    
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    if len(df) < config['seq_len']:
        print(f"❌ Ошибка: После очистки от NaN осталось всего {len(df)} строк для {SYMBOL}/{config['tf']}.")
        return

    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
    features_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Histogram']
    
    try:
        scaler_X = joblib.load(config["scaler_features_file"])
        scaler_y = joblib.load(config["scaler_price_file"])
        model = tf.keras.models.load_model(config["model_file"], compile=False)
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей для {SYMBOL}/{config['tf']}: {e}")
        return

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
        signal_text = "BUY"
        signal_color_code = "GREEN"
    elif price_diff < -0.00005:
        signal_text = "SELL"
        signal_color_code = "RED"
    else:
        signal_text = "HOLD"
        signal_color_code = "YELLOW"

    # 5. ВЫВОД В КОНСОЛЬ
    print("\n" + "="*30)
    print(f"📊 ПРОГНОЗ ДЛЯ ПАРЫ: {SYMBOL} ({config['tf']})")
    print("="*30)
    
    # --- НОВЫЙ БЛОК: Расчет разницы в пунктах ---
    # 1 пункт (pip) для EURUSD = 0.00001
    # Чтобы получить целое число пунктов, умножаем разницу на 100000
    pips_diff = int(round((predicted_price_real - current_price_real) * 100000, 0))

    # Используем цветовые коды для консоли
    if signal_text == "BUY":
        color_start, color_end = "\033[92m", "\033[0m" # Зеленый
    elif signal_text == "SELL":
        color_start, color_end = "\033[91m", "\033[0m" # Красный
    else:
        color_start, color_end = "\033[93m", "\033[0m" # Желтый

    print(f"🕒 Текущая цена (последняя свеча): {current_price_real:.5f}")
    print(f"🔮 Прогноз цены через 1 свечу: {predicted_price_real:.5f}")
    
    # Выводим разницу с учетом знака (+ или -)
    pips_sign = "+" if pips_diff > 0 else ""
    print(f"Разница в пунктах: {pips_sign}{pips_diff}")
    
    print(f"{color_start}🚩 СИГНАЛ: {signal_text}{color_end}")
    
    # 6. СОХРАНЕНИЕ РЕЗУЛЬТАТА В ФАЙЛ
    timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = (
        f"[{timestamp}] | "
        f"Таймфрейм: {config['tf']} | "
        f"Цена: {current_price_real:.5f} | "
        f"Прогноз: {predicted_price_real:.5f} | "
        f"Сигнал: {signal_text} ({signal_color_code}) | "
        f"Время: {round(time.time() - start_time, 2)}s\n"
    )

    try:
        # Режим 'a' (append) добавляет данные в конец файла, не удаляя старые.
        with open(config["log_file"], 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"✅ Прогноз для {SYMBOL}/{config['tf']} успешно сохранен в файл: {config['log_file']}")
        
    except Exception as e:
        print(f"⚠️  Ошибка при записи в файл для {SYMBOL}/{config['tf']}: {e}")

def main():
    print("🚀 Запуск объединенного скрипта прогноза для EUR/USD на трех таймфреймах...")
    start_time = time.time()

    # Прогноз для каждого таймфрейма отдельно
    for config in TF_CONFIGS:
        generate_forecast(config)

    total_execution_time = round(time.time() - start_time, 2)
    print(f"\n⏱️  Общее время выполнения: {total_execution_time} секунд.")

if __name__ == "__main__":
    main()