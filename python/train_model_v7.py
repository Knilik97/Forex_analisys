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
import joblib # Для сохранения масштабира

# --- НАСТРОЙКИ ---
SEQ_LEN = 120            # Сколько прошлых свечей модель будет "помнить"
FUTURE_PERIOD_PREDICT = 3 # <-- ИЗМЕНЕНИЕ: Предсказываем через n свечей
BATCH_SIZE = 64
MODEL_NAME = "model_v7.h5" # Новое имя для новой модели
# ------------------

# --- НАСТРОЙКИ GPU ---
# Сообщаем TensorFlow, что у нас есть GPU и мы хотим его использовать.
# Это нужно прописать один раз в начале скрипта.
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("💻 GPU найден и настроен. Будет использоваться для обучения.")
    else:
        print("⚠️ GPU не найден, будет использоваться CPU. Обучение может быть медленнее.")
except Exception as e:
    print(f"⚠️ Ошибка при настройке GPU: {e}. Будет использоваться CPU.")
# ------------------

def preprocess_regression_data(df):
    """Функция для подготовки данных к задаче регрессии."""
    df = df.copy()
    
    # 1. УДАЛЯЕМ ВСЕ НЕНУЖНЫЕ КОЛОНКИ (ИСПРАВЛЕНО)
    # Явно удаляем колонку 'time' (дату) и 'target' (цель), 
    # так как они не являются числовыми признаками для обучения.
    # 'target' мы удалим позже вручную, но лучше убрать её сразу, чтобы не мешала масштабированию.
    df = df.drop(['time', 'target'], axis=1, errors='ignore')
    
    # 2. СОЗДАЕМ МЕТКУ ДЛЯ РЕГРЕССИИ (Target)
    # Мы предсказываем цену 'close' на следующей свече
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    
    # Разделяем на признаки (X) и целевую переменную (y)
    # Мы используем ВСЕ колонки, кроме 'future_close', как признаки
    feature_columns = [col for col in df.columns if col != 'future_close']
    
    X = df[feature_columns].values
    y = df['future_close'].values
    
    # 3. МАСШТАБИРОВАНИЕ
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 4. СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ДЛЯ LSTM
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in range(len(X_scaled)):
        features = X_scaled[i]
        target = y_scaled[i]
        prev_days.append(features)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), target])
    
    # Перемешиваем данные для лучшего обучения
    random.shuffle(sequential_data)

    X_final = []
    y_final = []
    
    for seq, target in sequential_data:
        X_final.append(seq)
        y_final.append(target)
        
    return np.array(X_final), np.array(y_final), scaler_y, scaler_X


def main():
    print("🚀 Загрузка данных для V7...")
    
    INPUT_FILE = "data/ready_for_training_1H.csv"
    
    try:
        main_df = pd.read_csv(INPUT_FILE)
        print(f"✅ Данные загружены. Форма: {main_df.shape}")
        main_df.dropna(inplace=True) # На всякий случай уберем оставшиеся NaN
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
        print("Сначала нужно запустить create_labels.py")
        return

    print("🔧 Предобработка данных для регрессии...")
    X, y, scaler_y, scaler_X = preprocess_regression_data(main_df)

    print("📊 Разделение на обучающую и тестовую выборки...")
    # Разделяем на train/test. shuffle=False важно для временных рядов!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📚 Размер обучающей выборки: {len(X_train)}")
    print(f"🧪 Размер тестовой выборки: {len(X_test)}")

    # --- АРХИТЕКТУРА МОДЕЛИ (ТВОЯ ВЕРСИЯ V4) ---
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.5))
    
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.4))
    
    model.add(Dense(64))
    model.add(Dropout(0.3))
    
    model.add(Dense(32))
    
    # ВЫХОДНОЙ СЛОЙ ДЛЯ РЕГРЕССИИ (предсказание цены)
    model.add(Dense(1))

    # КОМПИЛЯЦИЯ МОДЕЛИ (СРЕДНЕКВАДРАТИЧНАЯ ОШИБКА для регрессии)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)

    # --- НОВИНКА: РАННЯЯ ОСТАНОВКА И СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
    checkpoint = ModelCheckpoint(
        MODEL_NAME,
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

    print("🧠 Начало обучения модели V7...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    
    # Сохраняем масштабировщики цены и признаков
    joblib.dump(scaler_y, 'scaler_price_v7.pkl') 
    joblib.dump(scaler_X, 'scaler_features_v7.pkl')
    
    print(f"💾 Финальные файлы сохранены:")
    print(f"- Модель: {MODEL_NAME}")
    print(f"- Масштабер цены: scaler_price_v7.pkl")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    main()