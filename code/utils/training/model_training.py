import gc

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD

from utils.experiment.run_helpers import current_global_seed, reapply_global_seed


# XGBoost trees may change branches when a PCA feature lands extremely close
# to a split threshold. BLAS can otherwise introduce a few 1e-9 of variation
# when the same transform is evaluated in a different process or batch.
PCA_FEATURE_DECIMALS = 6


def stable_pca_transform(pca, x):
    """Flatten x and quantize transform-level numerical BLAS noise."""
    array = np.asarray(x)
    flat = np.ascontiguousarray(array.reshape(array.shape[0], -1))
    return np.round(pca.transform(flat), decimals=PCA_FEATURE_DECIMALS)


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
        if "val_loss" in logs:
            self.history.setdefault("val_loss", []).append(float(logs["val_loss"]))


class EpochPredictionRecorder(Callback):
    """Record held-out predictions at predeclared epochs in original units."""

    def __init__(self, x_validation, scaler, checkpoint_epochs):
        super().__init__()
        self.x_validation = x_validation
        self.scaler = scaler
        self.checkpoint_epochs = {int(value) for value in checkpoint_epochs}
        self.predictions = {}

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1
        if completed not in self.checkpoint_epochs:
            return
        scaled = self.model.predict(self.x_validation, verbose=0)
        self.predictions[completed] = self.scaler.inverse_transform(
            np.asarray(scaled).reshape(-1, 1)
        ).ravel()


class TrainingProgress(Callback):
    """Emit a log-safe progress bar without relying on an interactive TTY.

    Keras's built-in ``verbose=1`` bar redraws a single line and can be
    suppressed by IDE consoles, redirected output, or log capture.  This
    callback instead writes completed checkpoints as ordinary lines, so the
    same progress is visible from a terminal, VS Code, and saved logs.
    """

    def __init__(self, label, total_epochs, interval_epochs=10):
        super().__init__()
        self.label = str(label)
        self.total_epochs = max(1, int(total_epochs))
        self.interval_epochs = max(1, int(interval_epochs))

    def on_train_begin(self, logs=None):
        print(f"[training] {self.label}: [--------------------] 0/{self.total_epochs}", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1
        if completed % self.interval_epochs and completed != self.total_epochs:
            return
        width = 20
        filled = round(width * completed / self.total_epochs)
        bar = "#" * filled + "-" * (width - filled)
        loss = (logs or {}).get("loss")
        loss_text = "" if loss is None else f", loss={float(loss):.6g}"
        print(
            f"[training] {self.label}: [{bar}] {completed}/{self.total_epochs}"
            f" ({completed / self.total_epochs:.0%}){loss_text}",
            flush=True,
        )


def _is_memory_error(exc):
    text = " ".join(
        str(part).lower()
        for part in (exc, repr(exc), getattr(exc, "message", ""))
        if part
    )
    return (
        isinstance(exc, (tf.errors.ResourceExhaustedError, MemoryError))
        or exc.__class__.__name__ == "_ArrayMemoryError"
        or "out of memory" in text
        or "cuda_error_out_of_memory" in text
        or "failed to allocate" in text
        or "could not create cudnn handle" in text
        or "cudnn_status_internal_error" in text
        or "dnn library is not found" in text
    )


class ModelTrainer:
    """
    1 モデルの学習・予測と、非深層モデル用の PCA 前処理をまとめたクラス。
    MODEL_SPECS の各 spec (key/label/kind/builder ...) を受け取り、
    kind ("keras" / "sklearn") に応じて入力形態を切り替える。
    """

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def make_pca(self, x_fit, x_other_list, n_components, return_pca=False):
        """sklearn 系モデル用に平坦化 + PCA。学習データのみで fit する。"""
        x_fit_flat = np.ascontiguousarray(
            np.asarray(x_fit).reshape(x_fit.shape[0], -1)
        )
        pca = PCA(n_components=min(n_components, x_fit_flat.shape[0], x_fit_flat.shape[1]),
                  random_state=self.random_seed)
        # Keep fitting and inference on the same numerical transform path.
        pca.fit(x_fit_flat)
        x_fit_pca = stable_pca_transform(pca, x_fit)
        others = []
        for x_other in x_other_list:
            if x_other is None:
                others.append(None)
            else:
                others.append(stable_pca_transform(pca, x_other))
        if return_pca:
            return x_fit_pca, others, pca
        return x_fit_pca, others

    def transform_pca(self, pca, x):
        return stable_pca_transform(pca, x)

    def _train_keras_model(
        self,
        spec,
        mm,
        x_fit,
        y_fit_scaled,
        epochs,
        validation_data=None,
        callback_factory=None,
    ):
        if "lr" not in spec or "batch_size" not in spec:
            raise ValueError(
                f"Keras model '{spec.get('key', spec.get('label'))}' needs "
                "resolved 'lr' and 'batch_size'. Check PARAMETER_SETS."
            )
        lr = spec["lr"]
        requested_batch_size = int(spec["batch_size"])
        fit_verbose = int(spec.get("fit_verbose", 0))
        progress_interval = int(spec.get("progress_interval_epochs", 10))
        min_batch_size = int(spec.get("min_batch_size", 1))
        accept_partial_min_epochs = int(spec.get("accept_partial_min_epochs", 100))
        batch_size = requested_batch_size
        last_oom = None

        while batch_size >= min_batch_size:
            K.clear_session()
            reapply_global_seed()
            gc.collect()
            model = spec["builder"](mm, **spec.get("builder_params", {}))
            model.compile(
                optimizer=SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0),
                loss="mean_squared_error",
            )
            lightweight_history = LightweightHistory()
            extra_callbacks = list(callback_factory() if callback_factory else [])
            callbacks = [
                lightweight_history,
                TrainingProgress(
                    spec.get("label", spec.get("key", "Keras model")),
                    epochs,
                    progress_interval,
                ),
                *extra_callbacks,
            ]
            try:
                history = model.fit(
                    x_fit,
                    y_fit_scaled,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=fit_verbose,
                    callbacks=callbacks,
                    validation_data=validation_data,
                )
                history.history = lightweight_history.history
                history.params["requested_batch_size"] = requested_batch_size
                history.params["actual_batch_size"] = batch_size
                history.params["epochs_completed"] = lightweight_history.epochs_completed
                return model, history, extra_callbacks
            except Exception as exc:
                if not _is_memory_error(exc):
                    raise
                last_oom = exc
                if lightweight_history.epochs_completed >= accept_partial_min_epochs:
                    lightweight_history.params["requested_batch_size"] = requested_batch_size
                    lightweight_history.params["actual_batch_size"] = batch_size
                    lightweight_history.params["epochs_completed"] = lightweight_history.epochs_completed
                    lightweight_history.params["stopped_by_memory_error"] = True
                    print(
                        f"[OOM accepted] {spec.get('key', spec.get('label'))}: "
                        f"{lightweight_history.epochs_completed} epochs completed; "
                        "current weights will be used."
                    )
                    return model, lightweight_history, extra_callbacks
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

    def train_one_model(self, spec, mm, x_fit, y_fit_scaled,
                        x_fit_pca, epochs):
        """1 モデルを学習して返す。kind に応じて入力形態を変える。"""
        if spec["kind"] == "keras":
            model, history, _ = self._train_keras_model(
                spec, mm, x_fit, y_fit_scaled, epochs
            )
            return model, history
        else:  # sklearn / xgboost
            builder_params = dict(spec.get("builder_params", {}))
            if spec.get("random_state_from_run", False):
                builder_params.setdefault(
                    "random_state", current_global_seed(self.random_seed)
                )
            model = spec["builder"](mm, **builder_params)
            label = spec.get("label", spec.get("key", "sklearn model"))
            print(f"[training] {label}: [--------------------] 0/1", flush=True)
            model.fit(x_fit_pca, y_fit_scaled.ravel())
            print(f"[training] {label}: [####################] 1/1 (100%)", flush=True)
            return model, None

    def train_one_model_with_epoch_validation(
        self,
        spec,
        mm,
        x_fit,
        y_fit_scaled,
        x_fit_pca,
        epochs,
        x_validation,
        _x_validation_pca,
        y_validation,
        scaler,
        checkpoints,
    ):
        """Train a Keras model and retain held-out predictions at checkpoints."""
        if spec["kind"] != "keras":
            raise ValueError("Epoch checkpoint validation is only defined for Keras models.")
        y_validation_scaled = scaler.transform(
            np.asarray(y_validation, dtype=float).reshape(-1, 1)
        )

        def callback_factory():
            return [EpochPredictionRecorder(x_validation, scaler, checkpoints)]

        model, history, callbacks = self._train_keras_model(
            spec,
            mm,
            x_fit,
            y_fit_scaled,
            epochs,
            validation_data=(x_validation, y_validation_scaled),
            callback_factory=callback_factory,
        )
        recorder, = callbacks
        return model, history, dict(recorder.predictions)

    def predict_one_model(self, spec, model, x, x_pca, scaler):
        """学習済みモデルで予測し、元スケールの熱流束に戻して返す。"""
        if spec["kind"] == "keras":
            pred_scaled = model.predict(x, verbose=0)
        else:
            pred_scaled = model.predict(x_pca).reshape(-1, 1)
        return scaler.inverse_transform(pred_scaled).ravel()
