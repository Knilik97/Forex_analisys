import pandas as pd

# --- НАСТРОЙКИ ---
INPUT_FILE = "data/EUR_USD/D1/EURUSD_D1_with_features.csv"
OUTPUT_FILE = "data/EUR_USD/D1/ready_for_training_EURUSD_D1.csv"
# ------------------

print("🏷️ Создание целевой переменной (Label)...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Данные с индикаторами загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить feature_engineering.py")
    quit()

# --- СОЗДАНИЕ ЛАБЕЛА (ЦЕЛЕВОЙ ПЕРЕМЕННОЙ) ---
# Модель должна предсказать цену 'close' на следующем шаге (на следующей 15-минутной свече).
# Мы просто сдвигаем колонку 'close' на 1 шаг ВНИЗ.
# Текущая цена станет признаком, а будущая - целью.
df['target'] = df['close'].shift(-1)

# --- ОЧИСТКА ДАННЫХ ---
# После сдвига последняя строка будет содержать NaN в колонке 'target',
# так как для неё нет "будущей" цены. Удаляем её.
df = df.dropna()

print(f"✅ Целевая переменная создана. Удалена последняя строка с NaN. Осталось: {len(df)}")

# --- СОХРАНЕНИЕ ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ УСПЕХ! Файл для обучения готов: {OUTPUT_FILE}")
print(f"Итоговая форма данных: {df.shape}")