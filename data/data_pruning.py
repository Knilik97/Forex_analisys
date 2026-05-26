import pandas as pd

# --- НАСТРОЙКИ ---
input_file = 'data/EUR_USD/M30/EURUSD_M30_markup.csv'
output_file = 'data/EUR_USD/M30/EURUSD_M30_tobe.csv'

# --- КОД ---
# 1. Загружаем уже размеченный файл
#    Указываем, что первая колонка ('datetime') — это индекс по времени (для скорости)
df = pd.read_csv(input_file, parse_dates=['datetime'])

# 2. Определяем список лет, которые нужно УДАЛИТЬ
years_to_remove = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009] # Укажи свои года здесь

# 3. Фильтрация
#    .dt.year извлекает год из колонки datetime.
#    ~ (тильда) означает "НЕ ВХОДИТ В СПИСОК".
filtered_df = df[~df['datetime'].dt.year.isin(years_to_remove)]

# 4. Сохраняем результат
filtered_df.to_csv(output_file, index=False)

print(f"✅ Готово! Файл сохранён: {output_file}")
print(f"🗑️ Удалены строки за годы: {years_to_remove}")
print(f"📊 Осталось строк: {len(filtered_df)}")