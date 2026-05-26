import pandas as pd
import numpy as np
import ta  # Библиотека для технического анализа

# --- НАСТРОЙКИ ---
INPUT_FILE = "data/EUR_USD/H1/EURUSD_H1.csv"
OUTPUT_FILE = "data/EUR_USD/H1/EURUSD_H1_with_features.csv"
# ------------------

print("🔧 Запуск расчета технических индикаторов...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Данные загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить data_collector.py")
    quit()

# --- РАСЧЕТ ИНДИКАТОРОВ ---
# Используем колонку 'close' как базу для большинства индикаторов
close_prices = df['close']
high_prices = df['high']
low_prices = df['low']
volume = df['tick_volume']

# 1. Скользящие средние (MA)
df['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()

# 2. Индекс относительной силы (RSI) - используем библиотеку ta для точности
rsi_indicator = ta.momentum.RSIIndicator(close=close_prices, window=14)
df['RSI'] = rsi_indicator.rsi()

# 3. Схождение/расхождение скользящих средних (MACD)
ema_12 = close_prices.ewm(span=12, adjust=False).mean()
ema_26 = close_prices.ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Histogram'] = df['MACD'] - df['Signal']

# 4. ATR (Average True Range) - Индикатор волатильности
# Используем библиотеку ta для корректного расчета True Range
atr_indicator = ta.volatility.AverageTrueRange(high=high_prices, low=low_prices, close=close_prices, window=14)
df['ATR_14'] = atr_indicator.average_true_range()

# 5. OBV (On-Balance Volume) - Индикатор накопления/распределения объема
obv_values = [0] # Начинаем с 0
for i in range(1, len(df)):
    if df['close'].iloc[i] > df['close'].iloc[i-1]:
        obv_values.append(obv_values[-1] + df['tick_volume'].iloc[i])
    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
        obv_values.append(obv_values[-1] - df['tick_volume'].iloc[i])
    else:
        obv_values.append(obv_values[-1])
df['OBV'] = obv_values

# 6. Bollinger Bands (Полосы Боллинджера) - Дополнительный индикатор структуры
bollinger = ta.volatility.BollingerBands(close=close_prices, window=20, window_dev=2)
df['BB_Middle'] = bollinger.bollinger_mavg()
df['BB_Upper'] = bollinger.bollinger_hband()
df['BB_Lower'] = bollinger.bollinger_lband()

# --- ОЧИСТКА ДАННЫХ ---
# Индикаторы в начале DataFrame будут содержать NaN, так как им не хватает данных для расчета.
# Удаляем эти строки.
initial_len = len(df)
df = df.dropna()
dropped_rows = initial_len - len(df)

print(f"✅ Индикаторы рассчитаны. Удалено строк с NaN: {dropped_rows}. Осталось: {len(df)}")
print(f"Добавлены новые колонки: ATR_14, OBV, BB_Middle, BB_Upper, BB_Lower")

# --- СОХРАНЕНИЕ ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ УСПЕХ! Данные с индикаторами сохранены в {OUTPUT_FILE}")
print(f"Итоговая форма данных: {df.shape}")