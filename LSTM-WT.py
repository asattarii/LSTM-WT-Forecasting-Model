import numpy as np
import pandas as pd
import pywt
import itertools
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback


# ===== Hyperparameter Grid =====
param_grid = {
    'hidden_size': [25, 50, 100, 150],
    'num_layers': [1, 2],
    'dropout': [0.1, 0.2, 0.4, 0.5],
    'batch_size': [50, 128, 256, 512],
    'learning_rate': [0.0001, 0.005, 0.001]
}

# ===== Load Data =====
data = pd.read_csv('data.csv')
discharge = data['observation'].values
# ==== Wavelet Transform ====
train_indices = # user_defined slice(start, end)
val_indices = #user_defined slice(, )
test_indices = #user_defined slice(, )

train_discharge = discharge[train_indices]
val_discharge = discharge[val_indices]
test_discharge = discharge[test_indices]

def wavelet_features(discharge_segment):
    n_levels = "User_defined"
    coeffs = pywt.wavedec(discharge_segment, 'wavelet_name', level=n_levels)
    reconstructed_signals = []
    coeff_for_approximation = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approximation = pywt.waverec(coeff_for_approximation, 'wavelet_name')
    reconstructed_signals.append(approximation[:len(discharge_segment)])
    for i in range(1, n_levels + 1):
        coeff_for_detail = [np.zeros_like(c) if j != i else coeffs[i] for j, c in enumerate(coeffs)]
        detail = pywt.waverec(coeff_for_detail, 'wavelet_name')
        reconstructed_signals.append(detail[:len(discharge_segment)])
    features = np.vstack(reconstructed_signals).T
    feature_names = ['approximation'] + [f'detail_level_{i}' for i in range(1, n_levels + 1)]
    features_df = pd.DataFrame(features, columns=feature_names)
    return features_df

# ===== Apply on dataset =====
train_features_df = wavelet_features(train_discharge)
train_df = pd.concat([train_features_df, data.iloc[train_indices].reset_index(drop=True)], axis=1)
val_features_df = wavelet_features(val_discharge)
val_df = pd.concat([val_features_df, data.iloc[val_indices].reset_index(drop=True)], axis=1)
test_features_df = wavelet_features(test_discharge)
test_df = pd.concat([test_features_df, data.iloc[test_indices].reset_index(drop=True)], axis=1)


df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

# ===== Normalization =====
scaler = MinMaxScaler()
scaler_target = MinMaxScaler()

indices = "user_defined"  # Replace with actual indices when ready
X_train = df.drop(columns=['observation']).values[indices]
y_train = df['observation'].values[indices].reshape(-1, 1)

scaler.fit(X_train)
scaler_target.fit(y_train)

df_scaled = scaler.transform(df.drop(columns=['observation']).values)
target_scaled = scaler_target.transform(df['observation'].values.reshape(-1, 1))
df_scaled = np.hstack([df_scaled, target_scaled])


def split_sequences_multistep(sequences, n_steps_in, n_steps_out, target_index=-1):
    X, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x = sequences[i:end_ix, :-1]
        seq_y = sequences[end_ix:out_end_ix, target_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps_in = "User_defined"
n_steps_ahead = "User_defined"
X, y = split_sequences_multistep(df_scaled, n_steps_in, n_steps_ahead)

test_split = "User_defined"
X_train, y_train = X[:test_split], y[:test_split]
X_test, y_test = X[test_split:], y[test_split:]
validation_split_ratio = "User_defined"
val_split_index = int((1 - validation_split_ratio) * len(X_train))
X_train_main, X_val = X_train[:val_split_index], X_train[val_split_index:]
y_train_main, y_val = y_train[:val_split_index], y_train[val_split_index:]
n_features = X.shape[2]

# ===== NSE Callback =====
class NSECallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val, scaler_target, n_steps_ahead):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_target = scaler_target
        self.n_steps_ahead = n_steps_ahead

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_val_pred = self.model.predict(self.X_val, verbose=0)

        y_train_pred_rescaled = self.scaler_target.inverse_transform(y_train_pred.reshape(-1, 1)).reshape(y_train_pred.shape)
        y_train_rescaled = self.scaler_target.inverse_transform(self.y_train.reshape(-1, 1)).reshape(self.y_train.shape)
        y_val_pred_rescaled = self.scaler_target.inverse_transform(y_val_pred.reshape(-1, 1)).reshape(y_val_pred.shape)
        y_val_rescaled = self.scaler_target.inverse_transform(self.y_val.reshape(-1, 1)).reshape(self.y_val.shape)

        nse_train, nse_val = [], []
        for step in range(self.n_steps_ahead):
            train_actual = y_train_rescaled[:, step]
            train_predicted = y_train_pred_rescaled[:, step]
            val_actual = y_val_rescaled[:, step]
            val_predicted = y_val_pred_rescaled[:, step]
            train_nse_step = 1 - (np.sum((train_actual - train_predicted) ** 2) /
                                  np.sum((train_actual - np.mean(train_actual)) ** 2))
            val_nse_step = 1 - (np.sum((val_actual - val_predicted) ** 2) /
                                np.sum((val_actual - np.mean(val_actual)) ** 2))
            nse_train.append(train_nse_step)
            nse_val.append(val_nse_step)
        print(f"Epoch {epoch + 1} - NSE Train: {[f'{nse:.4f}' for nse in nse_train]} | Val: {[f'{nse:.4f}' for nse in nse_val]}")

# ====== Grid Search ======
combinations = list(itertools.product(*param_grid.values()))
for i, (hidden_size, num_layers, dropout, batch_size, lr) in enumerate(combinations):
    print(f"\nGrid Search {i + 1}/{len(combinations)}: hidden_size={hidden_size}, layers={num_layers}, dropout={dropout}, batch={batch_size}, lr={lr}")

    # ==== Model ====
    model = Sequential()
    for layer in range(num_layers):
        return_seq = True if layer < num_layers - 1 else False
        if layer == 0:
            model.add(LSTM(hidden_size, activation='relu', return_sequences=return_seq, input_shape=(n_steps_in, n_features)))
        else:
            model.add(LSTM(hidden_size, activation='relu', return_sequences=return_seq))
        model.add(Dropout(dropout))

    model.add(Dense(n_steps_ahead))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae')

    # ==== Callbacks ====
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    nse_callback = NSECallback(X_train_main, y_train_main, X_val, y_val, scaler_target, n_steps_ahead)

    # ==== Train ====
    model.fit(X_train_main, y_train_main, epochs=400, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=[early_stop, nse_callback], verbose=0)

    # ==== Evaluate ====
    y_test_preds = model.predict(X_test, verbose=0)
    y_test_preds_rescaled = scaler_target.inverse_transform(y_test_preds.reshape(-1, 1)).reshape(y_test_preds.shape)
    y_test_actuals_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    metrics = {'nse': []}
    forecasted_data = []
    for step in range(n_steps_ahead):
        actual = y_test_actuals_rescaled[:, step]
        predicted = y_test_preds_rescaled[:, step]
        nse = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        metrics['nse'].append(nse)
        for j in range(len(actual)):
            forecasted_data.append({
                'lead_time': f't+{step + 1}',
                'time_step': j + 1,
                'observed': actual[j],
                'forecasted': predicted[j],
                'nse': nse if j == 0 else ''
            })

    # ==== Save ====
    label = f"hs{hidden_size}_nl{num_layers}_do{int(dropout*10)}_bs{batch_size}_lr{lr}"
    forecasted_df = pd.DataFrame(forecasted_data)
    forecasted_df.to_csv(f'forecasted_streamflow_{label}.csv', index=False)
    metrics_df = pd.DataFrame({'step': [f't+{i+1}' for i in range(n_steps_ahead)], 'nse': metrics['nse']})
    metrics_df.to_csv(f'nse_metrics_{label}.csv', index=False)
    print(f"✅ Results saved for {label}")
