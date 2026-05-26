import pandas as pd
import numpy as np
import random
import time
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import config # Импортируем наш конфиг

# --- НАСТРОЙКИ (ГЛОБАЛЬНЫЕ) ---
SEQ_LEN = 96            # Длина последовательности (память модели)
FUTURE_PERIOD_PREDICT = 2 # На сколько свечей вперед предсказываем
# BATCH_SIZE теперь будет браться из конфига для каждой пары

# --- НАСТРОЙКИ GPU ---
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("💻 GPU найден и настроен.")
    else:
        print("⚠️ GPU не найден, будет использоваться CPU.")
except Exception as e:
    print(f"⚠️ Ошибка при настройке GPU: {e}. Будет использоваться CPU.")
# ------------------

def preprocess_regression_data(df):
    """Функция для подготовки данных к задаче регрессии."""
    df = df.copy()
    
    # 1. Удаляем ненужные колонки
    df = df.drop(['time', 'target'], axis=1, errors='ignore')
    
    # 2. Создаем метку (Target) - цену через FUTURE_PERIOD_PREDICT свечей
    df['future_close'] = df['close'].shift(-FUTURE_PERIOD_PREDICT)
    df.dropna(inplace=True)
    
    # Разделяем на признаки (X) и целевую переменную (y)
    feature_columns = [col for col in df.columns if col != 'future_close']
    X = df[feature_columns].values
    y = df['future_close'].values

    # 3. Масштабирование
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 4. Создание последовательностей для LSTM
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in range(len(X_scaled)):
        features = X_scaled[i]
        target = y_scaled[i]
        prev_days.append(features)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), target])
    
    random.shuffle(sequential_data)

    X_final, y_final = [], []
    for seq, target in sequential_data:
        X_final.append(seq)
        y_final.append(target)
        
    return np.array(X_final), np.array(y_final), scaler_y, scaler_X

def build_model(input_shape, pair_config):
    """
    Строит модель динамически на основе параметров из конфига.
    """
    model = Sequential()
    
    # Первый LSTM-слой (возвращает последовательности для следующего LSTM)
    model.add(LSTM(
        pair_config['lstm_units_1'], 
        input_shape=input_shape, 
        return_sequences=True,
        activation='tanh'
    ))
    model.add(Dropout(pair_config['dropout_rate']))
    
    # Второй LSTM-слой
    model.add(LSTM(
        pair_config['lstm_units_2'], 
        return_sequences=False # Здесь возвращаем только последний выход
    ))
    model.add(Dropout(pair_config['dropout_rate']))
    
    # Первый Dense-слой
    model.add(Dense(
        pair_config['dense_units_1'], 
        activation='relu'
    ))
    model.add(Dropout(pair_config['dropout_rate']))
    
    # Второй Dense-слой
    model.add(Dense(
        pair_config['dense_units_2'], 
        activation='relu'
    ))
    
    # --- УСЛОВИЕ ДЛЯ CHF/USD: Третий Dense-слой ---
    # Проверяем, есть ли в конфиге параметр для третьего слоя
    if 'dense_units_3' in pair_config:
        model.add(Dropout(pair_config['dropout_rate'])) # Добавляем еще один Dropout перед третьим слоем
        model.add(Dense(
            pair_config['dense_units_3'], 
            activation='relu'
        ))
    
    # Выходной слой для регрессии (предсказание цены)
    model.add(Dense(1))
    
    return model

def main():
    # --- ВЫБОР ВАЛЮТНОЙ ПАРЫ ДЛЯ ОБУЧЕНИЯ ---
    # Меняй эту строку, чтобы обучить другую пару!
    pair_key = "EURUSD"  # <--- МОЖЕШЬ ПОМЕНЯТЬ НА "GBPUSD" или "CHFUSD"
    
    print(f"🚀 Запуск обучения для пары: {pair_key}...")
    
    # Получаем настройки для выбранной пары из конфига
    pair_config = config.CONFIGS[pair_key]
    
    INPUT_FILE = f"data/ready_for_training.csv" # Путь к данным

    try:
        main_df = pd.read_csv(INPUT_FILE)
        print(f"✅ Данные загружены. Форма: {main_df.shape}")
        main_df.dropna(inplace=True)
        
        if len(main_df) < SEQ_LEN:
            print(f"❌ Ошибка: Недостаточно данных. Нужно минимум {SEQ_LEN} свечей.")
            return

    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
        print("Сначала нужно запустить create_labels.py")
        return

    print("🔧 Предобработка данных для регрессии...")
    X, y, scaler_y, scaler_X = preprocess_regression_data(main_df)

    print("📊 Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📚 Размер обучающей выборки: {len(X_train)}")
    print(f"🧪 Размер тестовой выборки: {len(X_test)}")

    # --- ОПРЕДЕЛЕНИЕ ВХОДА МОДЕЛИ ---
    # input_shape = (кол-во свечей в последовательности, кол-во признаков)
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"🧠 Входная форма модели: {input_shape}")

    # --- СОЗДАНИЕ МОДЕЛИ (ДИНАМИЧЕСКИ ПО КОНФИГУ) ---
    model = build_model(input_shape, pair_config)

    # КОМПИЛЯЦИЯ МОДЕЛИ
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)

    model.summary() # Выведем архитектуру в консоль для проверки

    # --- ОБУЧЕНИЕ: EarlyStopping и ModelCheckpoint ---
    # Имена файлов для сохранения будут браться из конфига
    checkpoint = ModelCheckpoint(
        pair_config['model_name'],
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min'
    )

    print(f"🚀 Начало обучения модели {pair_key}...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=pair_config['batch_size'], # Берем размер батча из конфига
        epochs=pair_config['epochs'],         # Берем кол-во эпох из конфига
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО для {pair_key}!")
    
    # Сохраняем масштабировщики с именами из конфига
    joblib.dump(scaler_y, pair_config['scaler_name']) 
    joblib.dump(scaler_X, f"scaler_features_{pair_key}.pkl") # Сохраняем и масштабер признаков
    
    print(f"💾 Финальные файлы сохранены для {pair_key}:")
    print(f"- Модель: {pair_config['model_name']}")
    print(f"- Масштабер: {pair_config['scaler_name']}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    main()