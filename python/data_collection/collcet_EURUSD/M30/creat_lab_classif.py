import pandas as pd

# --- НАСТРОЙКИ ---
INPUT_FILE = "data/EUR_USD/M30/EURUSD_M30_with_features.csv"
OUTPUT_FILE = "data/EUR_USD/M30/ready_for_training_EURUSD_M30_class.csv"
THRESHOLD = 0.00010 # Порог для определения значимого движения цены
# ------------------

print("🏷️ Создание целевой переменной (Label)...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Данные с индикаторами загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить скрипт расчета индикаторов.")
    quit()

# --- СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (КЛАССА) ---
# Берем текущую цену и будущую (со сдвигом -1)
current_close = df['close']
future_close = df['close'].shift(-1)

# Считаем разницу между будущей и текущей ценой
price_diff = future_close - current_close

# Функция определения класса: BUY, SELL или HOLD
def get_class(diff):
    if diff > THRESHOLD:
        return 2 # BUY (Цена вырастет)
    elif diff < -THRESHOLD:
        return 0 # SELL (Цена упадет)
    else:
        return 1 # HOLD (Флэт, движения нет)

# Применяем функцию ко всем значениям разницы цен
df['target_class'] = price_diff.apply(get_class)

# --- ОЧИСТКА ДАННЫХ ---
# Убираем последнюю строку, где нет будущей цены (там NaN)
df = df.dropna()

print(f"✅ Целевая переменная 'target_class' создана. Удалена последняя строка с NaN. Осталось: {len(df)}")

# --- СОХРАНЕНИЕ ---
# Сохраняем DataFrame с новой колонкой 'target_class'
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ УСПЕХ! Файл для обучения готов: {OUTPUT_FILE}")
print(f"Итоговая форма данных: {df.shape}")