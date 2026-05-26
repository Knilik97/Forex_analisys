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
import os

# --- НАСТРОЙКИ ---
SYMBOLS = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY"] # Список валютных пар
TIMEFRAME = "1H" # Таймфрейм данных в файлах
SEQ_LEN = 96
FUTURE_PERIOD_PREDICT = 1 # Предсказываем на 1 свечу вперед
BATCH_SIZE = 64
MODEL_NAME = "model_multi_v1.h5"
DATASETS_PATH = 'ml_ready_data' # Папка с файлами *_features.csv
# ------------------

def preprocess_multi_data(main_df):
    """Функция для подготовки объединенных данных."""
    df = main_df.copy()
    
    # Удаляем колонки, которые не нужны для обучения
    df = df.drop(['time', 'pair', 'target'], axis=1, errors='ignore')
    
    # Создаем цель (target) - предсказываем цену закрытия текущей свечи на шаг вперед
    df['future_close'] = df['close'].shift(-FUTURE_PERIOD_PREDICT)
    df.dropna(inplace=True)
    
    # Разделяем на признаки и цель
    feature_columns = [col for col in df.columns if col != 'future_close']
    X = df[feature_columns].values
    y = df['future_close'].values

    # Масштабирование (ВАЖНО: один скалер на все пары!)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Создание последовательностей для LSTM
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


def main():
    print("🚀 Загрузка и объединение данных для мульти-парной модели...")
    
    main_df = pd.DataFrame()
    
    # Собираем данные по всем парам из папки
    for filename in os.listdir(DATASETS_PATH):
        if filename.endswith(f'_{TIMEFRAME}_features.csv'):
            pair = filename.split('_')[0]
            if pair in SYMBOLS:
                print(f"📥 Загружаю пару: {pair}")
                df = pd.read_csv(os.path.join(DATASETS_PATH, filename))
                df['pair'] = pair # Добавляем колонку с именем пары
                main_df = pd.concat([main_df, df])
    
    if main_df.empty:
        print("❌ Ошибка: Не найдены файлы данных.")
        return

    main_df.dropna(inplace=True)
    print(f"✅ Данные объединены. Общая форма: {main_df.shape}")

    print("🔧 Предобработка данных...")
    X, y, scaler_y, scaler_X = preprocess_multi_data(main_df)

    print("📊 Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📚 Размер обучающей выборки: {len(X_train)}")
    print(f"🧪 Размер тестовой выборки: {len(X_test)}")

    # --- АРХИТЕКТУРА МОДЕЛИ ---
    # Архитектура остается прежней! Ей не нужно "знать", что пар много.
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)

    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    print("🧠 Начало обучения мульти-парной модели...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    
    joblib.dump(scaler_y, 'scaler_price_multi.pkl') 
    joblib.dump(scaler_X, 'scaler_features_multi.pkl')
    
    print(f"💾 Финальные файлы сохранены:")
    print(f"- Модель: {MODEL_NAME}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    main()