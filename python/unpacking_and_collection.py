# Прежде собери данные в консле, файл будет в папке env1 через проводник 
#     mt4forexparser --mt4-history "C:/Users/Knilik977/AppData/Roaming/MetaQuotes/Terminal/50CA3DFB510CC5A8F28B48D1BF2A5702/history/MetaQuotes-Demo/EURUSD30.hst" --output "EURUSD_M30.csv" --parse


import pandas as pd

# --- НАСТРОЙКИ ---
SOURCE_FILE = 'data/EUR_USD/M30/EURUSD_M30_full.csv'
TARGET_FILE = 'data/EUR_USD/M30/EURUSD_M30_labeled.csv'
COLUMN_NAMES = ['time', 'open', 'high', 'low', 'close', 'volume']
# ----------------------

# 1. Загрузка данных без заголовков
df = pd.read_csv(SOURCE_FILE, header=None)

# 2. Назначение заголовков колонок
df.columns = COLUMN_NAMES

# 3. Преобразование времени в datetime
df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d,%H:%M')

# 4. Сохранение с заголовками
df.to_csv(TARGET_FILE, index=False)

# 5. Вывод информации
print("\n" + "="*30)
print("📊 Информация о данных:")
print("="*30)

# Размеры датафрейма
print(f"Количество записей: {len(df)}")

# Первые 5 строк
print("\nПервые 5 записей:")
print(df.head())

# Последние 5 строк
print("\nПоследние 5 записей:")
print(df.tail())

# Типы данных
print("\nТипы данных:")
print(df.dtypes)

# Статистика
print("\nСтатистические показатели:")
print(df.describe())

print(f"\n✅ Заголовки добавлены и сохранены в {TARGET_FILE}")