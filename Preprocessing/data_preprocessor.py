import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Battery SOH preprocessing pipeline:
    - One-hot encode categorical variables
    - Remove NaN / Inf rows
    - Remove physically invalid IC values
    - Train/test split
    - Scale IC bins and cycle number
    - Build context vector
    - Normalize target
    - Reshape IC features for CNN/LSTM
    """

    def __init__(
        self,
        train_size=0.8,
        random_state=42,
        ic_low=0.0,
        ic_high=20.0
    ):
        self.train_size = train_size
        self.random_state = random_state
        self.ic_low = ic_low
        self.ic_high = ic_high

        # Scalers
        self.ic_scaler = StandardScaler()
        self.cycle_scaler = StandardScaler()

        # Target normalization params
        self.y_mean = None
        self.y_std = None

        # Column trackers
        self.ic_cols = None
        self.one_hot_cols = None

    def preprocess(self, df: pd.DataFrame):
        # --------------------------------------------------
        # 0. One-hot encode categorical features
        # --------------------------------------------------
        df = pd.get_dummies(df, columns=['temp', 'charge_rate'])

        # --------------------------------------------------
        # 1. Drop unused columns
        # --------------------------------------------------
        df = df.drop(columns=['filename', 'cell_id', 'discharge_rate'])

        # --------------------------------------------------
        # 2. Identify columns
        # --------------------------------------------------
        self.ic_cols = [c for c in df.columns if 'IC' in c]
        self.one_hot_cols = [
            c for c in df.columns
            if c.startswith('temp_') or c.startswith('charge_rate_')
        ]

        X_ic = df[self.ic_cols].values
        X_cycle = df[['cycle']].values
        X_onehot = df[self.one_hot_cols].values
        y = df['SOH'].values

        # --------------------------------------------------
        # 3. Remove NaN / Inf rows
        # --------------------------------------------------
        mask_clean = (
            ~np.isnan(X_ic).any(axis=1) &
            ~np.isnan(X_cycle).any(axis=1) &
            ~np.isnan(X_onehot).any(axis=1) &
            ~np.isnan(y) &
            ~np.isinf(X_ic).any(axis=1) &
            ~np.isinf(X_cycle).any(axis=1) &
            ~np.isinf(X_onehot).any(axis=1) &
            ~np.isinf(y)
        )

        X_ic = X_ic[mask_clean]
        X_cycle = X_cycle[mask_clean]
        X_onehot = X_onehot[mask_clean]
        y = y[mask_clean]

        # --------------------------------------------------
        # 4. Remove physically invalid IC values
        # --------------------------------------------------
        mask_ic = np.all(
            (X_ic >= self.ic_low) & (X_ic <= self.ic_high),
            axis=1
        )

        X_ic = X_ic[mask_ic]
        X_cycle = X_cycle[mask_ic]
        X_onehot = X_onehot[mask_ic]
        y = y[mask_ic]

        # --------------------------------------------------
        # 5. Train-test split
        # --------------------------------------------------
        X_ic_tr, X_ic_te, \
        X_cycle_tr, X_cycle_te, \
        X_onehot_tr, X_onehot_te, \
        y_tr, y_te = train_test_split(
            X_ic, X_cycle, X_onehot, y,
            train_size=self.train_size,
            random_state=self.random_state
        )

        # --------------------------------------------------
        # 6. Standardize IC bins
        # --------------------------------------------------
        X_ic_tr = self.ic_scaler.fit_transform(X_ic_tr)
        X_ic_te = self.ic_scaler.transform(X_ic_te)

        # --------------------------------------------------
        # 7. Standardize cycle number
        # --------------------------------------------------
        X_cycle_tr = self.cycle_scaler.fit_transform(X_cycle_tr)
        X_cycle_te = self.cycle_scaler.transform(X_cycle_te)

        # --------------------------------------------------
        # 8. Build context vector (cycle + one-hot)
        # --------------------------------------------------
        X_ctx_tr = np.concatenate([X_cycle_tr, X_onehot_tr], axis=1)
        X_ctx_te = np.concatenate([X_cycle_te, X_onehot_te], axis=1)

        # --------------------------------------------------
        # 9. Normalize SOH target
        # --------------------------------------------------
        self.y_mean = y_tr.mean()
        self.y_std = y_tr.std()

        y_tr = (y_tr - self.y_mean) / self.y_std
        y_te = (y_te - self.y_mean) / self.y_std

        # --------------------------------------------------
        # 10. Expand dims
        # --------------------------------------------------
        X_ic_tr = X_ic_tr[..., np.newaxis]
        X_ic_te = X_ic_te[..., np.newaxis]

        return {
            "X_ic_train": X_ic_tr,
            "X_ic_test": X_ic_te,
            "X_context_train": X_ctx_tr,
            "X_context_test": X_ctx_te,
            "y_train": y_tr,
            "y_test": y_te,
            "y_mean":self.y_mean,
            "y_std":self.y_std
        }
 