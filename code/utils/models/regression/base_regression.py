from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Conv2D,
                                     BatchNormalization, MaxPooling2D, Dropout, 
                                     Flatten, Dense, Activation, Input, Permute, Reshape, 
                                     Bidirectional, GRU, GlobalAveragePooling1D,
                                     MultiHeadAttention, LayerNormalization, Layer, Embedding,
                                     TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16, EfficientNetB0
from tensorflow.keras.regularizers import l2
from xgboost import XGBRFRegressor
import numpy as np
import tensorflow as tf


#######################################################################

#                             transformer系

#######################################################################

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Self-Attention
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)

    # Feed Forward Network
    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dropout(dropout)(ff_out)
    ff_out = Dense(inputs.shape[-1])(ff_out)
    x = LayerNormalization(epsilon=1e-6)(ff_out + x)
    return x


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

class LogPowerCompression(Layer):
    """Compress raw non-negative spectrogram power without per-sample scaling."""

    def __init__(self, scale=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("LogPowerCompression scale must be positive.")

    def call(self, inputs):
        return tf.math.log1p(tf.maximum(inputs, 0.0) / self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config
    
#######################################################################

#                               CNN系

#######################################################################

class RegressionModelMaker:
    def __init__(self, input_shape):
        self.input_shape = input_shape  # インスタンス変数を定義(ここで指定した変数は、class内ならどこからでもアクセス可能)
        pass

    def alexnet(self):
        """Selected AlexNet: log-power input and the legacy regression head.

        The discarded architecture variants and their comparison results are
        documented in experiments/2026-08-30_log_power_architecture_study.md.
        """
        x_in = Input(shape=self.input_shape, name="spec_input")
        x = LogPowerCompression(scale=1e-12, name="log_power")(x_in)

        x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                   padding="valid", name="alex_conv1")(x)
        x = BatchNormalization(name="alex_bn1")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding="valid", name="alex_pool1")(x)
        x = Conv2D(256, (5, 5), activation='relu', padding="same",
                   name="alex_conv2")(x)
        x = BatchNormalization(name="alex_bn2")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding="valid", name="alex_pool2")(x)
        x = Conv2D(384, (3, 3), activation='relu', padding="same",
                   name="alex_conv3")(x)
        x = Conv2D(384, (3, 3), activation='relu', padding="same",
                   name="alex_conv4")(x)
        x = Conv2D(256, (3, 3), activation='relu', padding="same",
                   name="alex_conv5")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding="valid", name="alex_pool3")(x)

        x = Flatten(name="alex_flatten")(x)
        x = Dense(4096, activation="relu", name="alex_dense1")(x)
        x = Dropout(0.5, name="alex_dropout1")(x)
        x = Dense(4096, activation="relu", name="alex_dense2")(x)
        x = Dropout(0.2, name="alex_dropout2")(x)
        output = Dense(1, activation="linear", name="output")(x)
        return Model(x_in, output, name="alexnet_legacy_log")

    def vgg16(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))  # 回帰のための線形活性化関数

        return model

    def resnet50(self):
        base_model = ResNet50(weights=None, include_top=False, input_shape=self.input_shape)
        x = base_model.output
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear')(x)  # 回帰のための線形活性化関数
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def random_forest(self, **params):
        """
        XGBoostを使用したランダムフォレスト回帰モデル
        入力は (Batch, Feature_Size) の2次元配列である必要があります。
        """
        model = XGBRFRegressor(
            n_estimators=100,       # 木の数
            subsample=0.8,          # 各木で使うデータの割合
            colsample_bynode=0.8,   # 各分岐で使う特徴量の割合
            learning_rate=1.0,      # RFは学習率1.0が基本
            max_depth=10,           # 木の深さ（深すぎると過学習、浅すぎると精度不足）
            random_state=42,
            n_jobs=-1,              # CPU並列処理
            # tree_method='hist',     
            # device='cuda',          # GPU指定
            tree_method='auto',
            device='cpu',
            objective='reg:squarederror' # 回帰問題（二乗誤差最小化）
        )
        if params:
            model.set_params(**params)
        return model


    def mobilenet_v2(self):
        # 1. ImageNetで学習済みのMobileNetV2をロード (weights='imagenet')
        #    include_top=False で、最終の全結合層は含めない
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=self.input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)

        # 3. 回帰のためのヘッド（出力層）を追加
        #    Dropoutで正則化し、最後のDense層で1つの連続値を出力

        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)

        x = Dropout(0.5)(x)  # Dropout率は0.2〜0.5で調整するのが一般的です
        outputs = Dense(1, activation='linear')(x)  # 回帰なので活性化関数は linear

        # 4. 新しいモデルを定義
        model = Model(inputs=base_model.input, outputs=outputs)

        return model
    
    def seldnet_regressor(self, pool_sizes=(8, 8, 2), rnn_units=(128, 128)):
        """
        2D-CNN + BiGRUを回帰ヘッドにつないだ簡易SELDnet。
        入力 : (T, F, C) = (224, 224, 1)を想定（既存のspectrogram画像）
        出力 : 熱流束1値
        """
        x_in = Input(shape=self.input_shape, name='spec_input')  # (T,F,C)

        # --- CNNブロック ---
        x = x_in
        for p in pool_sizes:
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(1, p))(x)  # 時間軸はプールしない
            x = Dropout(0.2)(x)

        # --- 2D->3D reshape & BiGRU ---
        T = self.input_shape[0]
        x = Permute((2, 1, 3))(x)      # (F',T,C')
        x = Reshape((T, -1))(x)        # (T, features)
        for units in rnn_units:
            x = Bidirectional(
                GRU(units, return_sequences=True,
                    dropout=0.2, recurrent_dropout=0.2),
                merge_mode='mul')(x)

        # --- 時間平均 → 回帰線形出力 ---
        x = GlobalAveragePooling1D()(x)  # (features,)
        output = Dense(1, activation='linear', name='heatflux')(x)

        return Model(inputs=x_in, outputs=output)

    def efficientnet_b0(self):
            """
            EfficientNetB0をベースにした回帰モデル。
            少ないパラメータで高い性能が期待できる。
            """
            # 1. ベースモデルのロード
            base_model = EfficientNetB0(
                weights=None,          # ゼロから学習
                include_top=False,     # 全結合層は含めない
                input_shape=self.input_shape
            )

            # 2. 回帰ヘッドの構築
            x = base_model.output
            x = GlobalAveragePooling2D()(x) # Flattenの代わりにGAPを使うとパラメータを削減でき、過学習に強い傾向がある
            # x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='linear')(x)

            # 3. 新しいモデルを定義
            model = Model(inputs=base_model.input, outputs=outputs)
            
            return model

    def cnn_transformer_v2(self):
            """Selected CNN+Transformer: balanced time-axis tokens with log power.

            Architecture search results and the reason for fixing these values
            are documented in
            experiments/2026-08-30_log_power_architecture_study.md.
            """
            num_transformer_blocks = 2
            num_heads = 4
            ff_dim = 256
            model_dim = 64
            attention_key_dim = 16
            dropout = 0.1

            x_in = Input(shape=self.input_shape, name='spec_input')
            x = LogPowerCompression(scale=1e-12, name="log_power")(x_in)

            # --- AlexNetベースの特徴抽出器 ---
            # 元のalexnetのConv層とPooling層をFunctional APIで記述
            x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding="same")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            
            x = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            
            x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same")(x)
            x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same")(x)
            x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same")(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

            # --- 2D->3D tokenization ---
            shape = x.shape
            x = Reshape(
                (shape[1], shape[2] * shape[3]),
                name="v2_time_axis_tokens",
            )(x)

            x = TimeDistributed(Dense(model_dim, activation='relu'),
                                name='v2_projection')(x)

            x = PositionalEmbedding(sequence_length=x.shape[1],
                                    output_dim=x.shape[-1],
                                    name='v2_positional_embedding')(x)

            # --- Transformer Encoder ブロック ---
            # Transformer Encoderを複数重ねる
            for _ in range(num_transformer_blocks):
                x = transformer_encoder(x, attention_key_dim, num_heads,
                                        ff_dim, dropout=dropout)

            # --- 時間平均 → 回帰線形出力 (seldnet_regressorと同じ) ---
            x = GlobalAveragePooling1D(name='v2_global_average_pooling')(x)
            output = Dense(1, activation='linear', name='output')(x)

            return Model(
                inputs=x_in,
                outputs=output,
                name="cnn_transformer_v2_gap_balanced_axis_log",
            )

    #### EfficientNet + Tfのアーキテクチャ

    def efficientnet_transformer_v1(self,
                                        model_dim=128,
                                        num_transformer_blocks=4, 
                                        num_heads=4, 
                                        ff_dim=512):
        """
        EfficientNetB0をCNNベースとして利用し、Transformer Encoderに接続する回帰モデル。
        """
        # ---入力層---
        inputs = Input(shape=self.input_shape, name='spec_input')

        # --- 1. EfficientNetベースの特徴抽出器 ---
        # weights=Noneでランダムな初期値から学習を開始（転移学習しない）
        base_model = EfficientNetB0(
            include_top=False,  # include_top=Falseで全結合層を除いた特徴抽出部のみをロード
            weights=None, # スペクトログラムをゼロから学習
            input_tensor=inputs,
            pooling=None # 後段でプーリングするため、ここでは行わない
        )
        
        # EfficientNetからの出力特徴マップ
        x = base_model.output
        
        # --- 2. 2D特徴マップを3Dシーケンスに変換（Transformerへの橋渡し） ---
        # 出力形状を確認 (例: (None, 7, 7, 1280))
        shape = x.shape 
        
        # 時間軸(shape[1])をシーケンス長として、周波数軸(shape[2])とチャネル軸(shape[3])を特徴量次元にまとめる
        x = Reshape((shape[1], shape[2] * shape[3]))(x)
        
        # Transformerのモデル次元に合わせるために、Dense層で次元削減
        x = TimeDistributed(Dense(model_dim, activation='relu'))(x)

        # --- 3. Transformer Encoder ブロック ---
        # 位置情報を付与
        x = PositionalEmbedding(sequence_length=shape[1], output_dim=model_dim)(x) # 独自レイヤーを想定
        
        # Transformer Encoderを複数重ねる
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size=model_dim, num_heads=num_heads, ff_dim=ff_dim) # 独自関数を想定

        # --- 4. 時間軸の情報を集約し、最終的な回帰出力を得る ---
        # GlobalAveragePooling1Dでシーケンス全体の情報を平均化
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.5)(x)
        
        # 最終的な熱流束を予測
        outputs = Dense(1, activation='linear', name='output')(x)

        # モデルを構築
        return Model(inputs=inputs, outputs=outputs)


    #### wavenetは全然精度が出ないのでやめた

    # def wavenet_2d(self, num_blocks=2, num_layers_per_block=8, 
    #                filters=32, kernel_size=(3, 3)):
    #     """
    #     WaveNetのアーキテクチャを2次元入力に適用した回帰モデル。
        
    #     Args:
    #         num_blocks (int): WaveNetブロック（dilationをリセットするサイクル）の数。
    #         num_layers_per_block (int): 1ブロックあたりのレイヤー数。
    #         filters (int): 畳み込み層のフィルタ数。
    #         kernel_size (tuple): 畳み込みのカーネルサイズ。
    #     """
    #     x_in = Input(shape=self.input_shape, name='spec_input')

    #     # --- WaveNet Body ---
    #     skip_connections = []
    #     # 最初の畳み込みでチャンネル数を調整
    #     x = Conv2D(filters, 1, padding='same')(x_in)

    #     for block in range(num_blocks):
    #         for i in range(num_layers_per_block):
    #             dilation_rate = 2**i
                
    #             # Gated Activation Unit
    #             # 2つの並列した拡張畳み込み層
    #             tanh_out = Conv2D(filters, kernel_size, padding='same', 
    #                               dilation_rate=dilation_rate, activation='tanh')(x)
    #             sigmoid_out = Conv2D(filters, kernel_size, padding='same', 
    #                                  dilation_rate=dilation_rate, activation='sigmoid')(x)
                
    #             # ゲートを適用
    #             gated_out = multiply([tanh_out, sigmoid_out])

    #             # Skip Connection & Residual Connection
    #             # スキップ接続用の1x1畳み込み
    #             skip_conv = Conv2D(filters, 1, padding='same')(gated_out)
    #             skip_connections.append(skip_conv)

    #             # 残差接続用の1x1畳み込み
    #             residual_conv = Conv2D(filters, 1, padding='same')(gated_out)
    #             x = add([x, residual_conv])

    #     # --- 後処理 & 回帰ヘッド ---
    #     # 全てのスキップ接続を合計
    #     x = add(skip_connections)
    #     x = Activation('relu')(x)
        
    #     # 特徴マップをベクトル化して回帰値を出力
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(128, activation='relu')(x)
    #     x = Dropout(0.5)(x)
    #     output = Dense(1, activation='linear', name='output')(x)

    #     return Model(inputs=x_in, outputs=output)
