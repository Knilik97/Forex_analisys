# config.py

# Мы будем использовать архитектуру из v5 как эталон:
# LSTM(256) -> Dropout -> LSTM(128) -> Dropout -> Dense(64) -> Dropout -> Dense(32)

CONFIGS = {
    "EURUSD": {
        "symbol": "EURUSD",
        "model_name": "model_EURUSD.h5",
        "scaler_name": "scaler_EURUSD.pkl",
        "history_name": "history_EURUSD.txt",
        # --- АРХИТЕКТУРА V5 (МОЩНАЯ) ---
        # Оставляем как есть, это наш эталон
        "lstm_units_1": 256,
        "lstm_units_2": 128,
        "dense_units_1": 64,
        "dense_units_2": 32,
        "dropout_rate": 0.3,
        "batch_size": 64,
        "epochs": 30
    },
    "GBPUSD": {
        "symbol": "GBPUSD",
        "model_name": "model_GBPUSD.h5",
        "scaler_name": "scaler_GBPUSD.pkl",
        "history_name": "history_GBPUSD.txt",
        # --- АРХИТЕКТУРА (СЛАБЕЕ, ЧЕМ EUR/USD) ---
        # Уменьшаем количество нейронов в LSTM и Dense слоях
        # GBPUSD может быть менее волатильным или нам не нужна такая сложность
        "lstm_units_1": 128, # Было 256
        "lstm_units_2": 64,  # Было 128
        "dense_units_1": 32, # Было 64
        "dense_units_2": 16, # Было 32
        "dropout_rate": 0.3,
        "batch_size": 64,
        "epochs": 30
    },
    "CHFUSD": {
        "symbol": "CHFUSD", 
        "model_name": "model_CHFUSD.h5",
        "scaler_name": "scaler_CHFUSD.pkl",
        "history_name": "history_CHFUSD.txt",
        # --- АРХИТЕКТУРА (СИЛЬНЕЕ, ЧЕМ EUR/USD) ---
        # Добавляем еще один Dense-слой и увеличиваем количество нейронов.
        # Это сделает сеть глубже и мощнее для улавливания сложных паттернов.
        # Также немного увеличим dropout для регуляризации такой большой сети.
        "lstm_units_1": 512, # Увеличили первый LSTM-слой
        "lstm_units_2": 256, # Увеличили второй LSTM-слой
        "dense_units_1": 128,
        "dense_units_2": 64,
        # Добавляем ТРЕТИЙ Dense-слой для большей глубины
        # (В коде обучения это нужно будет учесть)
        "dense_units_3": 32,
        "dropout_rate": 0.4, # Увеличили dropout
        "batch_size": 128,   # Увеличили batch size для стабильности
        "epochs": 40         # Увеличили количество эпох для качественного обучения
    }
}