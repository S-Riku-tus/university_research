import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import SGD


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
        model = spec["builder"](mm, **spec.get("builder_params", {}))
        if spec["kind"] == "keras":
            if "lr" not in spec or "batch_size" not in spec:
                raise ValueError(
                    f"Keras model '{spec.get('key', spec.get('label'))}' needs "
                    "resolved 'lr' and 'batch_size'. Check PARAMETER_SETS."
                )
            lr = spec["lr"]
            batch_size = spec["batch_size"]
            model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0),
                          loss='mean_squared_error')
            history = model.fit(x_fit, y_fit_scaled,
                                batch_size=batch_size,
                                epochs=epochs, verbose=1)
            return model, history
        else:  # sklearn / xgboost
            model.fit(x_fit_pca, y_fit_scaled.ravel())
            return model, None

    def predict_one_model(self, spec, model, x, x_pca, scaler):
        """学習済みモデルで予測し、元スケールの熱流束に戻して返す。"""
        if spec["kind"] == "keras":
            pred_scaled = model.predict(x)
        else:
            pred_scaled = model.predict(x_pca).reshape(-1, 1)
        return scaler.inverse_transform(pred_scaled).ravel()
