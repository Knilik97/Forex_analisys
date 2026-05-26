import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import tensorflow as tf
import time
import os

# --- НАСТРОЙКИ ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_D1  # Таймфрейм должен совпадать с обучением
SEQ_LEN = 120                 # Длина последовательности (из обучения)
NUM_CANDLES_TO_DOWNLOAD = SEQ_LEN + 10  # Скачиваем с запасом для расчета индикаторов
MODEL_FILE = "model_classif_EURUSD_1D.h5"  # Имя файла обученной модели
SCALER_FEATURES_FILE = "scaler_features_classif.pkl"  # Файл масштабировщика
LOG_FILE = "prediction_log_D1_EURUSD.txt"

# Словари для красивого вывода
CLASS_NAMES = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
COLORS = {'SELL': 'RED', 'HOLD': 'YELLOW', 'BUY': 'GREEN'}
REASONS = {0: 'Цена слишком высока', 1: 'Рынок в балансе', 2: 'Цена привлекательна'}


def main():
    start_time = time.time()
    print("🚀 Запуск скрипта прогноза...")

    # 1. ПОДКЛЮЧЕНИЕ И СКАЧИВАНИЕ ДАННЫХ
    print("1/6: Подключение к MetaTrader 5...")
    if not mt5.initialize():
        print("❌ Ошибка подключения к терминалу.")
        return

    print(f"2/6: Скачивание {NUM_CANDLES_TO_DOWNLOAD} свечей для {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES_TO_DOWNLOAD)
    mt5.shutdown()  # Обязательно закрываем соединение

    if rates is None or len(rates) == 0:
        print("❌ Ошибка: Не удалось получить котировки.")
        return

    # 2. СОЗДАНИЕ DATAFRAME И РАСЧЕТ ИНДИКАТОРОВ
    print("3/6: Обработка данных и расчет индикаторов...")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Проверка на достаточное количество данных после очистки NaN
    if len(df) < SEQ_LEN:
        print(f"❌ Ошибка: После загрузки осталось всего {len(df)} строк. Нужно минимум {SEQ_LEN}.")
        return

    close_prices = df['close']

    # --- РАСЧЕТ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ ---
    df['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()

    # RSI
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    EMA_12 = close_prices.ewm(span=12, adjust=False).mean()
    EMA_26 = close_prices.ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA_12 - EMA_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Удаляем строки с NaN (где индикаторы еще не посчитались)
    df.dropna(inplace=True)

    if len(df) < SEQ_LEN:
        print(f"❌ Ошибка: После очистки от NaN осталось всего {len(df)} строк. Нужно минимум {SEQ_LEN}.")
        return

    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
    print("4/6: Подготовка данных для модели...")
    features_cols = [
        'open', 'high', 'low', 'close', 'tick_volume',
        'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal'
    ]

    try:
        scaler_X = joblib.load(SCALER_FEATURES_FILE)
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели или файлов масштабирования: {e}")
        return

    X = df[features_cols].values

    # Масштабируем данные так же, как при обучении
    data_scaled = scaler_X.transform(X)

    # Берем последние SEQ_LEN свечей для прогноза
    input_data = data_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features_cols))

    # 4. ПРОГНОЗ
    print("5/6: Генерация прогноза...")

    prediction_probabilities = model.predict(input_data)[0]  # Массив вероятностей [SELL, HOLD, BUY]
    predicted_class_index = np.argmax(prediction_probabilities)  # Индекс максимального значения

    current_price_real = df['close'].iloc[-1]

    signal_text = CLASS_NAMES[predicted_class_index]
    signal_color_code = COLORS[signal_text]
    reason_text = REASONS[predicted_class_index]

    execution_time = time.time() - start_time

    # 6/6: СОХРАНЕНИЕ РЕЗУЛЬТАТА В ФАЙЛ
    timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

    log_entry = (
        f"[{timestamp}] | "
        f"Текущая цена: {current_price_real:.5f} | "
        f"Сигнал: {signal_text} ({signal_color_code}) | "
        f"Причина: {reason_text} | "
        f"Вероятности [SELL,HOLD,BUY]: {[round(p,3) for p in prediction_probabilities.tolist()]} | "
        f"Время выполнения: {execution_time:.2f}s\n"
    )

    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # Создаем папку, если нет
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"✅ Прогноз успешно сохранен в файл: {LOG_FILE}")

    except Exception as e:
        print(f"⚠️ Ошибка при записи в файл: {e}")

    # 5. ВЫВОД В КОНСОЛЬ (обновленный блок)
    print("\n" + "=" * 30)
    print("📊 ИТОГОВЫЙ ПРОГНОЗ")
    print("=" * 30)

     # Используем цветовые коды для консоли
    color_start, color_end = {
         'SELL': ("\033[91m", "\033[0m"),   # Красный
         'BUY': ("\033[92m", "\033[0m"),   # Зеленый
         'HOLD': ("\033[93m", "\033[0m")   # Желтый
     }[signal_text]
     
    print(f"🕒 Текущая цена (последняя свеча): {current_price_real:.5f}")
    print(f"{color_start}🚩 СИГНАЛ: {signal_text}{color_end}")
    print(f"📝 Рекомендация: {reason_text}")
    print(f"⚡ Время выполнения скрипта: {execution_time:.2f} секунд")
     
# Запуск скрипта
if __name__ == "__main__":
   main()