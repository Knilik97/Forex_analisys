import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import tensorflow as tf
import time
import os

# --- НАСТРОЙКИ ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1 # Таймфрейм должен совпадать с обучением (V7)
SEQ_LEN = 240                 # Длина последовательности (из обучения)
NUM_CANDLES_TO_DOWNLOAD = 300 # Скачиваем с запасом
MODEL_FILE = "python/train_model_v7_EURUSD/EURUSD_H1/model_v7.3_EURUSD_240.h5"
SCALER_FEATURES_FILE = "python/train_model_v7_EURUSD/EURUSD_H1/scaler_features_v7.3_EURUSD.pkl"
SCALER_PRICE_FILE = "python/train_model_v7_EURUSD/EURUSD_H1/scaler_price_v7.3_EURUSD.pkl"
LOG_FILE = "hystory_log/prediction_log_H1_EURUSD.txt" 

def main():
    print("🚀 Запуск объединенного скрипта прогноза...")
    start_time = time.time()

    # 1. ПОДКЛЮЧЕНИЕ И СКАЧИВАНИЕ ДАННЫХ
    print("1/6: Подключение к MetaTrader 5...")
    if not mt5.initialize():
        print("❌ Ошибка подключения к терминалу.")
        return

    print(f"2/6: Скачивание {NUM_CANDLES_TO_DOWNLOAD} свечей для {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES_TO_DOWNLOAD)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("❌ Ошибка: Не удалось получить котировки.")
        return

    # 2. СОЗДАНИЕ DATAFRAME И РАСЧЕТ ИНДИКАТОРОВ
    print("3/6: Расчет технических индикаторов...")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Проверка на достаточное количество данных после очистки NaN
    if len(df) < SEQ_LEN + 10: # +10 на случай NaN от индикаторов
        print(f"⚠️ Внимание: Получено мало данных ({len(df)}). Прогноз может быть неточным.")
    
    close_prices = df['close']
    
    # Индикаторы (расчет как в data_processor)
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
    
    # Удаляем строки с NaN (где индикаторы еще не посчитались)
    df.dropna(inplace=True)
    
    if len(df) < SEQ_LEN:
        print(f"❌ Ошибка: После очистки от NaN осталось всего {len(df)} строк. Нужно минимум {SEQ_LEN}.")
        return

    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
    print("4/6: Подготовка данных для модели...")
    features_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Histogram']
    
    try:
        scaler_X = joblib.load(SCALER_FEATURES_FILE)
        scaler_y = joblib.load(SCALER_PRICE_FILE)
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели или файлов: {e}")
        return

    X = df[features_cols].values 
    data_scaled = scaler_X.transform(X)
    input_data = data_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features_cols))

    # 4. ПРОГНОЗ
    print("5/6: Генерация прогноза...")
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

    execution_time = time.time() - start_time

    # 6/6: СОХРАНЕНИЕ РЕЗУЛЬТАТА В ФАЙЛ
    timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = (
        f"[{timestamp}] | "
        f"Цена: {current_price_real:.5f} | "
        f"Прогноз: {predicted_price_real:.5f} | "
        f"Сигнал: {signal_text} ({signal_color_code}) | "
        f"Время: {execution_time:.2f}s\n"
    )

    try:
        # Режим 'a' (append) добавляет данные в конец файла, не удаляя старые.
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"✅ Прогноз успешно сохранен в файл: {LOG_FILE}")
        
    except Exception as e:
        print(f"⚠️  Ошибка при записи в файл: {e}")

    # 5. ВЫВОД В КОНСОЛЬ (обновленный блок)
    print("\n" + "="*30)
    print("📊 ИТОГОВЫЙ ПРОГНОЗ")
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
    print(f"🔮 Прогноз цены ? пока вопрос когда: {predicted_price_real:.5f}")
    
    # Выводим разницу с учетом знака (+ или -)
    pips_sign = "+" if pips_diff > 0 else ""
    print(f"Разница в пунктах: {pips_sign}{pips_diff}")
    
    print(f"{color_start}🚩 СИГНАЛ: {signal_text}{color_end}")
    
# Запуск скрипта
if __name__ == "__main__":
    main()