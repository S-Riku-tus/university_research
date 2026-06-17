import gc

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import SGD


class LightweightHistory(Callback):
    """Keep only the small loss history needed for plotting."""

    def __init__(self):
        super().__init__()
        self.history = {"loss": []}
        self.params = {}
        self.epochs_completed = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_completed = epoch + 1
        self.history["loss"].append(float(logs.get("loss", np.nan)))


def _is_memory_error(exc):
    return (
        isinstance(exc, (tf.errors.ResourceExhaustedError, MemoryError))
        or exc.__class__.__name__ == "_ArrayMemoryError"
    )


class ModelTrainer:
    """
    1 モデルの学習・予測と、非深層モデル用の PCA 前処理をまとめたクラス。
    MODEL_SPECS の各 spec (key/label/kind/builder ...) を受け取り、
    kind ("keras" / "sklearn") に応じて入力形態を切り替える。
    """

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def make_pca(self, x_fit, x_other_list, n_components):
        """sklearn 系モデル用に平坦化 + PCA。学習データのみで fit する。"""
        x_fit_flat = x_fit.reshape(x_fit.shape[0], -1)
        pca = PCA(n_components=min(n_components, x_fit_flat.shape[0], x_fit_flat.shape[1]),
                  random_state=self.random_seed)
        x_fit_pca = pca.fit_transform(x_fit_flat)
        others = []
        for x_other in x_other_list:
            if x_other is None:
                others.append(None)
            else:
                others.append(pca.transform(x_other.reshape(x_other.shape[0], -1)))
        return x_fit_pca, others

    def train_one_model(self, spec, mm, x_fit, y_fit_scaled,
                        x_fit_pca, epochs):
        """1 モデルを学習して返す。kind に応じて入力形態を変える。"""
        if spec["kind"] == "keras":
            if "lr" not in spec or "batch_size" not in spec:
                raise ValueError(
                    f"Keras model '{spec.get('key', spec.get('label'))}' needs "
                    "resolved 'lr' and 'batch_size'. Check PARAMETER_SETS."
                )
            lr = spec["lr"]
            requested_batch_size = int(spec["batch_size"])
            use_early_stopping = bool(spec.get("early_stopping", True))
            min_batch_size = int(spec.get("min_batch_size", 8))
            accept_partial_min_epochs = int(spec.get("accept_partial_min_epochs", 100))
            batch_size = requested_batch_size
            last_oom = None

            while batch_size >= min_batch_size:
                K.clear_session()
                gc.collect()
                model = spec["builder"](mm, **spec.get("builder_params", {}))
                model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0),
                              loss='mean_squared_error')
                lightweight_history = LightweightHistory()
                callbacks = [lightweight_history]
                if use_early_stopping:
                    callbacks.append(
                        EarlyStopping(
                            monitor="loss",
                            min_delta=float(spec.get("early_stopping_min_delta", 1e-4)),
                            patience=int(spec.get("early_stopping_patience", 20)),
                            restore_best_weights=False,
                            verbose=1,
                        )
                    )
                try:
                    history = model.fit(
                        x_fit, y_fit_scaled,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                    )
                    history.history = lightweight_history.history
                    history.params["requested_batch_size"] = requested_batch_size
                    history.params["actual_batch_size"] = batch_size
                    history.params["epochs_completed"] = lightweight_history.epochs_completed
                    history.params["early_stopping"] = use_early_stopping
                    return model, history
                except Exception as exc:
                    if not _is_memory_error(exc):
                        raise
                    last_oom = exc
                    if lightweight_history.epochs_completed >= accept_partial_min_epochs:
                        lightweight_history.params["requested_batch_size"] = requested_batch_size
                        lightweight_history.params["actual_batch_size"] = batch_size
                        lightweight_history.params["epochs_completed"] = lightweight_history.epochs_completed
                        lightweight_history.params["early_stopping"] = use_early_stopping
                        lightweight_history.params["stopped_by_memory_error"] = True
                        print(
                            f"[OOM accepted] {spec.get('key', spec.get('label'))}: "
                            f"{lightweight_history.epochs_completed} epochs completed; "
                            "current weights will be used."
                        )
                        return model, lightweight_history
                    print(
                        f"[OOM retry] {spec.get('key', spec.get('label'))}: "
                        f"batch_size={batch_size} でメモリ不足。"
                    )
                    del model
                    K.clear_session()
                    gc.collect()
                    next_batch_size = batch_size // 2
                    if next_batch_size < min_batch_size:
                        break
                    batch_size = next_batch_size

            raise last_oom
        else:  # sklearn / xgboost
            model = spec["builder"](mm, **spec.get("builder_params", {}))
            model.fit(x_fit_pca, y_fit_scaled.ravel())
            return model, None

    def predict_one_model(self, spec, model, x, x_pca, scaler):
        """学習済みモデルで予測し、元スケールの熱流束に戻して返す。"""
        if spec["kind"] == "keras":
            pred_scaled = model.predict(x)
        else:
            pred_scaled = model.predict(x_pca).reshape(-1, 1)
        return scaler.inverse_transform(pred_scaled).ravel()
