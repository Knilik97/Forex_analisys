import pandas as pd
import numpy as np
import ta  # Убедись, что библиотека установлена: pip install ta

# --- НАСТРОЙКИ ---
# ВХОД: Файл, который был обработан предыдущим скриптом (с EMA, RSI, ATR, OBV)
INPUT_FILE = "data/EUR_USD/H1/EURUSD_H1_with_features.csv"
# ВЫХОД: Новый файл с добавленными признаками "силы"
OUTPUT_FILE = "data/EUR_USD/H1/EURUSD_H1_with_power_features.csv"
# ------------------

print("⚡ Запуск расчета признаков 'Силы' (Power Features)...")

try:
    # Загружаем данные с уже рассчитанными индикаторами
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Данные с базовыми индикаторами загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить feature_engineering.py")
    quit()

# --- РАСЧЕТ ПРИЗНАКОВ "СИЛЫ" (POWER FEATURES) ---
# Используем уже существующие колонки для расчета новых признаков

# 1. СИЛА ТРЕНДА (Momentum Strength)
# Насколько цена отклонилась от скользящих средних?
df['Trend_Strength_EMA20'] = df['close'] - df['EMA_20'] # Положительное = Сильный бычий тренд
df['Trend_Strength_EMA50'] = df['close'] - df['EMA_50']

# 2. СИЛА ВОЛАТИЛЬНОСТИ (Volatility Strength)
# Является ли текущая волатильность (ATR) аномально высокой или низкой?
# Сравниваем текущее значение ATR со средним за последние 60 периодов (12.5 дней на H1)
df['Volatility_Strength_ATR'] = df['ATR_14'] / df['ATR_14'].rolling(window=60).mean()

# 3. СИЛА ОБЪЕМА (Volume Strength / Confirmation)
# Подтверждает ли объем текущее движение цены?

# Изменение OBV (накопление/распределение)
df['OBV_Momentum'] = df['OBV'].diff()

# Подтверждение объемом (1 - бычье, -1 - медвежье, 0 - нет подтверждения)
df['Volume_Confirmation'] = np.select(
    [
        (df['close'] > df['close'].shift(1)) & (df['tick_volume'] > df['tick_volume'].shift(1)),
        (df['close'] < df['close'].shift(1)) & (df['tick_volume'] > df['tick_volume'].shift(1))
    ],
    [1, -1],
    default=0
)

# 4. СИЛА ИМПУЛЬСА (Momentum)
# Насколько сильно изменилась цена за последний период?
df['Price_Momentum'] = df['close'].diff()

print(f"⚡ Признаки 'Силы' рассчитаны: Trend_Strength, Volatility_Strength, Volume_Confirmation.")

# --- ОЧИСТКА ДАННЫХ ---
# Новые признаки в начале DataFrame будут содержать NaN.
initial_len = len(df)
df = df.dropna()
dropped_rows = initial_len - len(df)

print(f"✅ Очистка завершена. Удалено строк с NaN: {dropped_rows}. Осталось: {len(df)}")

# --- СОХРАНЕНИЕ ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"🚀 УСПЕХ! Данные с признаками силы сохранены в {OUTPUT_FILE}")
print(f"Итоговая форма данных: {df.shape}")