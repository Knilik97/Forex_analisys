import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import tensorflow as tf
import time
import os

# --- НАСТРОЙКИ (ИСПОЛЬЗУЕМ ТВОИ ПУТИ) ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1 # Таймфрейм H1 (как при обучении)
SEQ_LEN = 240                 # Длина последовательности (должна совпадать с обучением)
NUM_CANDLES_TO_DOWNLOAD = 500 # Скачиваем с запасом

# ПУТИ К ФАЙЛАМ (ИЗ ТВОЕГО СООБЩЕНИЯ)
MODEL_FILE = "python/train_model_v7_EURUSD/EURUSD_H1/model_regression_power_v1.h5"
SCALER_FEATURES_FILE = "python/train_model_v7_EURUSD/EURUSD_H1/scaler_features_power.pkl"
SCALER_PRICE_FILE = "python/train_model_v7_EURUSD/EURUSD_H1/scaler_price_power.pkl"

LOG_FILE = "prediction_log_EURUSD_H1.txt" # Файл для логирования прогнозов

def main():
    print("🚀 Запуск скрипта прогноза...")
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
    print("3/6: Расчет технических индикаторов и признаков силы...")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    if len(df) < SEQ_LEN + 10:
        print(f"⚠️ Внимание: Получено мало данных ({len(df)}). Прогноз может быть неточным.")
    
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    volume = df['tick_volume']

    # --- БАЗОВЫЕ ИНДИКАТОРЫ ---
    df['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()

    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = close_prices.ewm(span=12, adjust=False).mean() - close_prices.ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    tr1 = high_prices - low_prices
    tr2 = (high_prices - close_prices.shift(1)).abs()
    tr3 = (low_prices - close_prices.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    obv_values = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv_values.append(obv_values[-1] + volume.iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv_values.append(obv_values[-1] - volume.iloc[i])
        else:
            obv_values.append(obv_values[-1])
    df['OBV'] = obv_values

     # --- НОВЫЕ ПРИЗНАКИ СИЛЫ (POWER FEATURES) ---
    df['Trend_Strength_EMA20'] = df['close'] - df['EMA_20']
    df['Trend_Strength_EMA50'] = df['close'] - df['EMA_50']
     
    df['Volatility_Strength_ATR'] = df['ATR_14'] / df['ATR_14'].rolling(window=60).mean()
     
    df['OBV_Momentum'] = df['OBV'].diff()
     
    df['Volume_Confirmation'] = np.select(
         [
             (df['close'] > df['close'].shift(1)) & (df['tick_volume'] > df['tick_volume'].shift(1)),
             (df['close'] < df['close'].shift(1)) & (df['tick_volume'] > df['tick_volume'].shift(1))
         ],
         [1, -1],
         default=0
     )
     
    df['Price_Momentum'] = df['close'].diff()
     
     # --- ДОБАВЛЕНО: РАСЧЕТ BOLLINGER BANDS ДЛЯ НОВЫХ ДАННЫХ ---
    window_bb = 20
    std_dev_bb = 2.0
    rolling_mean_bb = close_prices.rolling(window=window_bb).mean()
    rolling_std_bb = close_prices.rolling(window=window_bb).std()
    df['BB_Middle'] = rolling_mean_bb
    df['BB_Upper'] = rolling_mean_bb + (rolling_std_bb * std_dev_bb)
    df['BB_Lower'] = rolling_mean_bb - (rolling_std_bb * std_dev_bb)


    # Удаляем строки с NaN (где индикаторы еще не посчитались)
    initial_len = len(df)
    df.dropna(inplace=True)
    
    if len(df) < SEQ_LEN:
        print(f"❌ Ошибка: После очистки от NaN осталось всего {len(df)} строк. Нужно минимум {SEQ_LEN}.")
        return

    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
    print("4/6: Подготовка данных для модели...")
    
    # --- ЯВНЫЙ СПИСОК ПРИЗНАКОВ ИЗ 22 ШТУК ---
    features_cols = [
        'open', 'high', 'low', 'close', 'tick_volume',
        'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Histogram',
        'ATR_14', 'OBV', 'BB_Middle', 'BB_Upper', 'BB_Lower',
        'Trend_Strength_EMA20', 'Trend_Strength_EMA50',
        'Volatility_Strength_ATR', 'OBV_Momentum',
        'Volume_Confirmation', 'Price_Momentum'
    ]
    
    try:
        scaler_X = joblib.load(SCALER_FEATURES_FILE)
        scaler_y = joblib.load(SCALER_PRICE_FILE)
        
        # Ключевой параметр для решения ошибки десериализации метрик
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели или файлов: {e}")
        print("Убедись, что файлы лежат по указанным путям.")
        return

    X = df[features_cols].values 
    
    # Масштабируем данные и формируем последовательность для LSTM
    data_scaled = scaler_X.transform(X)
    
    # Берем последние SEQ_LEN свечей для прогноза
    input_data = data_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features_cols))

    # 4. ПРОГНОЗ
    print("5/6: Генерация прогноза...")
    
    try:
        prediction_scaled = model.predict(input_data)
        
        predicted_price_normalized = prediction_scaled.flatten()[0]
        predicted_price_real = scaler_y.inverse_transform(np.array([[predicted_price_normalized]]))[0][0]
        
        current_price_real = df['close'].iloc[-1]
        
        execution_time = time.time() - start_time

        # 6/6: СОХРАНЕНИЕ РЕЗУЛЬТАТА В ФАЙЛ
        timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = (
            f"[{timestamp}] | "
            f"Текущая цена: {current_price_real:.5f} | "
            f"Прогноз цены: {predicted_price_real:.5f} | "
            f"Время выполнения: {execution_time:.2f}s\n"
        )
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        print(f"✅ Прогноз успешно сохранен в файл: {LOG_FILE}")
        
        # 5. ВЫВОД В КОНСОЛЬ
        print("\n" + "="*35)
        print("📊 ИТОГОВЫЙ ПРОГНОЗ")
        print("="*35)
        
        price_diff_pips = round((predicted_price_real - current_price_real) * 100000, 2) # Для EURUSD

        print(f"🕒 Текущая цена (последняя свеча): {current_price_real:.5f}")
        print(f"🔮 Прогноз цены на следующую свечу: {predicted_price_real:.5f}")
        
        if price_diff_pips > 0:
            color_start, color_end = "\033[92m", "\033[0m" # Зеленый
            signal_text = "BUY"
            pips_sign = "+"
        elif price_diff_pips < 0:
            color_start, color_end = "\033[91m", "\033[0m" # Красный
            signal_text = "SELL"
            pips_sign = ""
        else:
            color_start, color_end = "\033[93m", "\033[0m" # Желтый
            signal_text = "HOLD"
            pips_sign = ""
            
        print(f"{color_start}🚩 СИГНАЛ: {signal_text}{color_end}")
        
        # Исправленный вывод разницы в пунктах с учетом знака (+/-)
        print(f"Разница в пунктах (pips): {pips_sign}{price_diff_pips}")
        
    except Exception as e:
        print(f"❌ Ошибка при генерации прогноза: {e}")

# Запуск скрипта
if __name__ == "__main__":
   main()