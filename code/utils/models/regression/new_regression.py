"""
regression_models.py
沸騰検知のための回帰モデル集
- CNN+LSTM (推奨)
- CNN+Transformer v2 (改良版)
- CNN+Transformer v1 (現行モデルの改善版)
"""
# ============================================================================
# インポート
# ============================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Reshape,
    LSTM, Bidirectional, TimeDistributed, LayerNormalization,
    MultiHeadAttention, Add, Layer
)
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling2D, GlobalAveragePooling1D,
    Lambda, Add, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
import warnings
warnings.filterwarnings('ignore')
# ============================================================================
# カスタムレイヤー: Positional Embedding
# ============================================================================
class PositionalEmbedding(Layer):
    """
    Transformerのための位置エンコーディング
    """
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embeddings = None
    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.sequence_length, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)
    def call(self, inputs):
        # inputs: (batch, sequence_length, output_dim)
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = tf.nn.embedding_lookup(
            self.position_embeddings, positions
        )
        return inputs + embedded_positions
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config
# ============================================================================
# カスタムレイヤー: Attention Pooling
# ============================================================================
class AttentionPooling(Layer):
    """
    系列の重み付き平均プーリング
    """
    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.attention_weights = None
    def build(self, input_shape):
        # input_shape: (batch, sequence_length, features)
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)
    def call(self, inputs):
        # inputs: (batch, sequence_length, features)
        # 各トークンの重要度を計算
        attention_scores = tf.matmul(inputs, self.attention_weights)
        # (batch, sequence_length, 1)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        # 重み付き和
        weighted = inputs * attention_scores
        output = tf.reduce_sum(weighted, axis=1)
        # (batch, features)
        return output
    def get_config(self):
        return super().get_config()
# ============================================================================
# Transformer Encoder Block
# ============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Transformer Encoderブロック
    Args:
        inputs: 入力テンソル (batch, seq_len, model_dim)
        head_size: 各Attentionヘッドの次元
        num_heads: Attentionヘッド数
        ff_dim: Feed-forward層の中間次元
        dropout: Dropout率
    Returns:
        出力テンソル (batch, seq_len, model_dim)
    """
    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    # Residual Connection & Layer Normalization
    x = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    # Feed-Forward Network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    # Residual Connection & Layer Normalization
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x
# ============================================================================
# カスタム損失関数: Weighted MSE Loss
# ============================================================================
class WeightedMSELoss(Loss):
    """
    沸騰開始点近傍を重視する損失関数
    """
    def __init__(self, critical_min=0.25, critical_max=0.45,
                 critical_weight=20.0, name="weighted_mse_loss"):
        super().__init__(name=name)
        self.critical_min = critical_min
        self.critical_max = critical_max
        self.critical_weight = critical_weight
    def call(self, y_true, y_pred):
        # MSE計算
        mse = tf.square(y_true - y_pred)
        # 遷移期（沸騰開始点近傍）の判定
        is_critical = tf.logical_and(
            y_true >= self.critical_min,
            y_true <= self.critical_max
        )
        # 重み適用
        weights = tf.where(is_critical, self.critical_weight, 1.0)
        weighted_mse = weights * mse
        return tf.reduce_mean(weighted_mse)
    def get_config(self):
        config = super().get_config()
        config.update({
            "critical_min": self.critical_min,
            "critical_max": self.critical_max,
            "critical_weight": self.critical_weight,
        })
        return config
# ============================================================================
# メインクラス: RegressionModelMaker
# ============================================================================
class RegressionModelMaker1106:
    """
    各種回帰モデルを構築するクラス
    """
    def __init__(self, input_shape):
        """
        Args:
            input_shape: 単一フレームの形状 (height, width, channels)
        """
        self.input_shape = input_shape
        print(f"RegressionModelMaker initialized with input_shape: {input_shape}")
    # ========================================================================
    # モデル1: CNN+LSTM (推奨★★★)
    # ========================================================================
    def cnn_lstm_regressor(self, num_frames=10, lstm_units=256,
                          dropout=0.3, use_bidirectional=False):
        """
        CNN+LSTMによる真の時系列回帰モデル（最推奨）
        Args:
            num_frames: 入力する連続フレーム数
            lstm_units: LSTMのユニット数
            dropout: Dropout率
            use_bidirectional: 双方向LSTMを使用するか
        Returns:
            Kerasモデル
        入力形状: (batch, num_frames, height, width, channels)
        出力形状: (batch, 1)
        """
        print(f"\n{'='*60}")
        print(f"Building CNN+LSTM Regressor")
        print(f"{'='*60}")
        print(f"  - Input: ({num_frames}, {self.input_shape})")
        print(f"  - LSTM units: {lstm_units}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Bidirectional: {use_bidirectional}")
        # 入力: 連続フレーム系列
        sequence_input = Input(
            shape=(num_frames, *self.input_shape),
            name='sequence_input'
        )
        # 各フレーム用の特徴抽出CNN
        cnn_base = self._create_feature_extractor_cnn(
            output_dim=512,
            dropout=dropout
        )
        # 各フレームに同じCNNを適用
        features = TimeDistributed(cnn_base, name='time_distributed_cnn')(
            sequence_input
        )
        # shape: (batch, num_frames, 512)
        # LSTM層
        if use_bidirectional:
            lstm_out = Bidirectional(
                LSTM(lstm_units // 2, return_sequences=True, dropout=dropout),
                name='bidirectional_lstm_1'
            )(features)
            lstm_out = Bidirectional(
                LSTM(lstm_units // 4, return_sequences=False, dropout=dropout),
                name='bidirectional_lstm_2'
            )(lstm_out)
        else:
            lstm_out = LSTM(
                lstm_units, return_sequences=True, dropout=dropout,
                name='lstm_1'
            )(features)
            lstm_out = LSTM(
                lstm_units // 2, return_sequences=False, dropout=dropout,
                name='lstm_2'
            )(lstm_out)
        # shape: (batch, lstm_units or lstm_units//2)
        # 全結合層
        x = Dense(128, activation='relu', name='fc1')(lstm_out)
        x = Dropout(dropout, name='dropout_fc')(x)
        # 回帰出力
        output = Dense(1, activation='linear', name='output')(x)
        model = Model(inputs=sequence_input, outputs=output,
                     name='CNN_LSTM_Regressor')
        print(f"\nModel created successfully!")
        print(f"Total parameters: {model.count_params():,}")
        return model

# ========================================================================
    # モデル2: 改良版CNN+Transformer (中推奨★★)
    # ========================================================================
    def cnn_transformer_v2_time_series_corrected(
        self,
        num_time_patches=10,      # 時間軸を何分割するか
        num_transformer_blocks=2,
        model_dim=256,
        num_heads=4,
        ff_dim=512,
        dropout=0.3
    ):
        """
        真の時系列モデル: 横軸(時間)を分割してTransformer処理
        データ構造:
        - 入力: (224, 224, 1)
            周波数 × 時間 × チャンネル
        処理フロー:
        1. 横軸(時間)を num_time_patches 個に分割
        例: 224 ÷ 10 = 22.4 → (10, 224, 22, 1)
        2. 各時間パッチを独立したCNNで処理
        (224, 22, 1) → feature vector
        3. 時系列: (10, feature_dim)
        4. Transformerで時間方向の依存関係を学習
        - 沸騰前 → 沸騰開始 → 沸騰後 の遷移
        物理的意味:
        - 各時刻のスペクトル特徴を抽出
        - 時間変化パターンから熱流束を予測
        """
        print("="*70)
        print("Building CNN+Transformer v2 (Time Series - CORRECTED)")
        print("="*70)
        print(f"  - Input: {self.input_shape}")
        print(f"  - Time axis: horizontal (width = {self.input_shape[1]})")
        print(f"  - Frequency axis: vertical (height = {self.input_shape[0]})")
        print(f"  - Time patches: {num_time_patches}")
        print(f"  - Patch width: {self.input_shape[1] // num_time_patches}")
        print(f"  - Model dim: {model_dim}")
        print(f"  - Transformer blocks: {num_transformer_blocks}")
        print(f"  - Attention heads: {num_heads}")
        print("  ✓ This model processes TEMPORAL evolution (correct!)")
        print("="*70)
        inputs = Input(shape=self.input_shape)  # (224, 224, 1)
        # 時間軸の分割幅を計算
        patch_width = self.input_shape[1] // num_time_patches
        # ============================================
        # 時間軸に沿ってパッチ分割
        # ============================================
        time_patches = []
        for i in range(num_time_patches):
            start_col = i * patch_width
            end_col = start_col + patch_width
            # スライス: [:, start:end, :] で横方向を切り出す
            patch = Lambda(
                lambda x, s=start_col, e=end_col: x[:, :, s:e, :],
                name=f'time_patch_{i}'
            )(inputs)
            # patch.shape = (224, patch_width, 1)
            time_patches.append(patch)
        # ============================================
        # 各時間パッチをCNNで処理（共有重み）
        # ============================================
        def create_patch_cnn(model_dim, name_prefix=''):
            """
            各時間パッチ用のCNN
            入力: (224, patch_width, 1)
            出力: (model_dim,)
            """
            def cnn_block(x):
                # Block 1
                x = Conv2D(64, (3, 3), padding='same', name=f'{name_prefix}_conv1')(x)
                x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
                x = Activation('relu')(x)
                x = MaxPooling2D((2, 2), name=f'{name_prefix}_pool1')(x)
                # Block 2
                x = Conv2D(128, (3, 3), padding='same', name=f'{name_prefix}_conv2')(x)
                x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
                x = Activation('relu')(x)
                x = MaxPooling2D((2, 2), name=f'{name_prefix}_pool2')(x)
                # Block 3
                x = Conv2D(256, (3, 3), padding='same', name=f'{name_prefix}_conv3')(x)
                x = BatchNormalization(name=f'{name_prefix}_bn3')(x)
                x = Activation('relu')(x)
                x = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
                # Project to model_dim
                x = Dense(model_dim, name=f'{name_prefix}_project')(x)
                return x
            return cnn_block
        # 共有CNNを作成
        shared_cnn = create_patch_cnn(model_dim, name_prefix='shared')
        # 各パッチに適用
        patch_features = []
        for i, patch in enumerate(time_patches):
            feature = shared_cnn(patch)
            patch_features.append(feature)
        # ============================================
        # 時系列に変換: (num_time_patches, model_dim)
        # ============================================
        x = Lambda(
            lambda features: tf.stack(features, axis=1),
            name='stack_time_series'
        )(patch_features)
        # x.shape = (batch, num_time_patches, model_dim)
        # ============================================
        # Positional Encoding（時刻情報を追加）
        # ============================================
        x = self._add_positional_encoding(x)
        # ============================================
        # Transformer Blocks（時間方向の依存関係を学習）
        # ============================================
        for block_idx in range(num_transformer_blocks):
            x = self._transformer_block(
                x,
                model_dim,
                num_heads,
                ff_dim,
                dropout,
                name_prefix=f'transformer_block_{block_idx}'
            )
        # ============================================
        # 時系列全体を集約して回帰
        # ============================================
        x = GlobalAveragePooling1D(name='temporal_pooling')(x)
        # Dense layers
        x = Dense(128, activation='relu', name='dense1')(x)
        x = Dropout(dropout, name='dropout1')(x)
        x = Dense(64, activation='relu', name='dense2')(x)
        x = Dropout(dropout, name='dropout2')(x)
        # Output
        outputs = Dense(1, name='output')(x)
        model = Model(inputs=inputs, outputs=outputs, name="cnn_transformer_v2_time_corrected")
        print(f"✓ Model created successfully!")
        print(f"  Total parameters: {model.count_params():,}")
        print("="*70)
        return model
    # ========================================================================
    # モデル3: 現行モデルの改善版 (短期対策★)
    # ========================================================================
    def cnn_transformer_v1_improved(self, num_transformer_blocks=2,
                                   model_dim=256, num_heads=4,
                                   ff_dim=512, dropout=0.4):
        """
        現行モデル(v1)の改善版（単一フレーム入力、パラメータ調整版）
        注意: これは周波数間の関係を学習するモデルであり、
             時系列を見ていない。短期的な改善策としてのみ使用。
        Args:
            num_transformer_blocks: Transformer層の数
            model_dim: Transformerのモデル次元（32→256に改善）
            num_heads: Attentionヘッド数
            ff_dim: Feed-forward中間次元（2048→512に改善）
            dropout: Dropout率（0.2→0.4に改善）
        Returns:
            Kerasモデル
        入力形状: (batch, height, width, channels)
        出力形状: (batch, 1)
        """
        print(f"\n{'='*60}")
        print(f"Building CNN+Transformer v1 (Improved)")
        print(f"{'='*60}")
        print(f":警告:  WARNING: This model processes frequency direction,")
        print(f"   not time series. Use only as a short-term solution.")
        print(f"{'='*60}")
        print(f"  - Input: {self.input_shape}")
        print(f"  - Model dim: {model_dim} (improved from 32)")
        print(f"  - Transformer blocks: {num_transformer_blocks} (reduced from 4)")
        print(f"  - FF dim: {ff_dim} (reduced from 2048)")
        print(f"  - Dropout: {dropout} (increased from 0.2)")
        # head_sizeの自動計算
        head_size = model_dim // num_heads
        assert model_dim % num_heads == 0, \
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        print(f"  - Head size: {head_size}")
        # 入力
        x_in = Input(shape=self.input_shape, name='spec_input')
        # AlexNet風のCNN特徴抽出
        x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                  padding="same", name='conv1')(x_in)
        x = BatchNormalization(name='bn1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                        padding="same", name='pool1')(x)
        x = Conv2D(256, (5, 5), strides=(1, 1), activation='relu',
                  padding="same", name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                        padding="same", name='pool2')(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu',
                  padding="same", name='conv3')(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu',
                  padding="same", name='conv4')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu',
                  padding="same", name='conv5')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                        padding="same", name='pool3')(x)
        # 2D→3D Reshape（周波数方向を系列とする）
        shape = x.shape
        sequence_length = shape[1]  # 周波数方向
        x = Reshape((sequence_length, shape[2] * shape[3]),
                   name='reshape_to_sequence')(x)
        print(f"  - Sequence length: {sequence_length} (frequency bins)")
        # model_dimへの射影（★改善: 32→256）
        x = TimeDistributed(
            Dense(model_dim, activation='relu'),
            name='projection'
        )(x)
        x = Dropout(dropout, name='dropout_projection')(x)
        # Positional Embedding
        x = PositionalEmbedding(
            sequence_length=sequence_length,
            output_dim=model_dim,
            name='positional_embedding'
        )(x)
        # Transformer Encoderブロック（★改善: 4→2層）
        for i in range(num_transformer_blocks):
            x = transformer_encoder(
                x,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
        # Attention Pooling
        x = AttentionPooling(name='attention_pooling')(x)
        # 回帰出力
        output = Dense(1, activation='linear', name='output')(x)
        model = Model(inputs=x_in, outputs=output,
                     name='CNN_Transformer_v1_Improved')
        print(f"\nModel created successfully!")
        print(f"Total parameters: {model.count_params():,}")
        return model
    # ========================================================================
    # 補助メソッド: 特徴抽出CNN
    # ========================================================================
    def _create_feature_extractor_cnn(self, output_dim=512, dropout=0.3):
        """
        各フレーム用の軽量CNN（共通）
        Args:
            output_dim: 出力特徴次元
            dropout: Dropout率
        Returns:
            Kerasモデル（フレーム→特徴ベクトル）
        """
        frame_input = Input(shape=self.input_shape, name='frame_input')
        # 浅めのCNN（過学習防止）
        x = Conv2D(64, (7, 7), strides=2, activation='relu',
                  padding='same', name='cnn_conv1')(frame_input)
        x = BatchNormalization(name='cnn_bn1')(x)
        x = MaxPooling2D(2, name='cnn_pool1')(x)
        x = Conv2D(128, (5, 5), activation='relu',
                  padding='same', name='cnn_conv2')(x)
        x = BatchNormalization(name='cnn_bn2')(x)
        x = MaxPooling2D(2, name='cnn_pool2')(x)
        x = Conv2D(256, (3, 3), activation='relu',
                  padding='same', name='cnn_conv3')(x)
        x = BatchNormalization(name='cnn_bn3')(x)
        x = MaxPooling2D(2, name='cnn_pool3')(x)
        # Global Average Pooling
        x = GlobalAveragePooling2D(name='cnn_gap')(x)
        # shape: (256,)
        # 全結合層
        x = Dense(output_dim, activation='relu', name='cnn_fc')(x)
        x = Dropout(dropout, name='cnn_dropout')(x)
        # shape: (output_dim,)
        return Model(inputs=frame_input, outputs=x,
                    name='Feature_Extractor_CNN')
    
    def _add_positional_encoding(self, x):
        """
        Add positional encoding to sequence
        x.shape = (batch, seq_len, model_dim)
        """
        def positional_encoding_fn(inputs):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            model_dim = tf.shape(inputs)[2]
            # Position indices
            position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
            # Dimension indices
            div_term = tf.exp(
                tf.range(0, model_dim, 2, dtype=tf.float32) *
                -(tf.math.log(10000.0) / tf.cast(model_dim, tf.float32))
            )
            # Positional encoding
            pos_encoding = tf.zeros((seq_len, model_dim))
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.stack([tf.repeat(tf.range(seq_len), model_dim // 2),
                        tf.tile(tf.range(0, model_dim, 2), [seq_len])], axis=1),
                tf.reshape(tf.sin(position * div_term), [-1])
            )
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.stack([tf.repeat(tf.range(seq_len), model_dim // 2),
                        tf.tile(tf.range(1, model_dim, 2), [seq_len])], axis=1),
                tf.reshape(tf.cos(position * div_term), [-1])
            )
            # Add to inputs
            return inputs + pos_encoding[tf.newaxis, :, :]
        return Lambda(positional_encoding_fn, name='positional_encoding')(x)

    def _transformer_block(self, x, model_dim, num_heads, ff_dim, dropout, name_prefix=''):
        """
        Transformer Block with proper naming
        """
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            name=f'{name_prefix}_mha'
        )(x, x)
        attn_output = Dropout(dropout, name=f'{name_prefix}_attn_dropout')(attn_output)
        x = Add(name=f'{name_prefix}_attn_add')([x, attn_output])
        x = LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_attn_ln')(x)
        # Feed Forward Network
        ffn = Dense(ff_dim, activation='relu', name=f'{name_prefix}_ffn1')(x)
        ffn = Dropout(dropout, name=f'{name_prefix}_ffn_dropout')(ffn)
        ffn = Dense(model_dim, name=f'{name_prefix}_ffn2')(ffn)
        x = Add(name=f'{name_prefix}_ffn_add')([x, ffn])
        x = LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ffn_ln')(x)
        return x


# ============================================================================
# データ準備用ユーティリティ
# ============================================================================
def create_sequence_dataset(spectrograms, labels, num_frames=10,
                           stride=1, verbose=True):
    """
    単一フレームから連続フレーム系列データセットを作成
    Args:
        spectrograms: スペクトログラム配列 (N, H, W, C)
        labels: 熱流束ラベル (N,)
        num_frames: 系列長
        stride: スライディングウィンドウのストライド
        verbose: 進捗表示
    Returns:
        sequences: (N', num_frames, H, W, C)
        seq_labels: (N',) - 系列の最後のラベル
    注意: 同じ実験条件のフレームを連続させる必要があります。
         実際の使用時は、実験IDや電圧でグループ化してください。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Creating sequence dataset")
        print(f"{'='*60}")
        print(f"  - Input frames: {len(spectrograms)}")
        print(f"  - Sequence length: {num_frames}")
        print(f"  - Stride: {stride}")
    sequences = []
    seq_labels = []
    # 簡易版: 全データを1つのグループとして扱う
    # 実際の使用時は実験ごとにグループ化する
    total_sequences = (len(spectrograms) - num_frames) // stride + 1
    for i in range(0, len(spectrograms) - num_frames + 1, stride):
        seq = spectrograms[i:i + num_frames]
        label = labels[i + num_frames - 1]  # 最後のフレームのラベル
        sequences.append(seq)
        seq_labels.append(label)
    sequences = np.array(sequences)
    seq_labels = np.array(seq_labels)
    if verbose:
        print(f"  - Output sequences: {len(sequences)}")
        print(f"  - Sequence shape: {sequences.shape}")
        print(f"  - Labels shape: {seq_labels.shape}")
        print(f"{'='*60}\n")
    return sequences, seq_labels
def create_grouped_sequence_dataset(spectrograms, labels, group_ids,
                                   num_frames=10, stride=1, verbose=True):
    """
    実験グループごとに連続フレーム系列を作成（推奨）
    Args:
        spectrograms: スペクトログラム配列 (N, H, W, C)
        labels: 熱流束ラベル (N,)
        group_ids: グループID（実験ID、電圧など） (N,)
        num_frames: 系列長
        stride: スライディングウィンドウのストライド
        verbose: 進捗表示
    Returns:
        sequences: (N', num_frames, H, W, C)
        seq_labels: (N',)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Creating grouped sequence dataset")
        print(f"{'='*60}")
        print(f"  - Input frames: {len(spectrograms)}")
        print(f"  - Unique groups: {len(np.unique(group_ids))}")
        print(f"  - Sequence length: {num_frames}")
        print(f"  - Stride: {stride}")
    sequences = []
    seq_labels = []
    # グループごとに処理
    unique_groups = np.unique(group_ids)
    for group_id in unique_groups:
        # このグループのデータを取得
        mask = group_ids == group_id
        group_specs = spectrograms[mask]
        group_labels = labels[mask]
        # 連続系列を作成
        for i in range(0, len(group_specs) - num_frames + 1, stride):
            seq = group_specs[i:i + num_frames]
            label = group_labels[i + num_frames - 1]
            sequences.append(seq)
            seq_labels.append(label)
    sequences = np.array(sequences)
    seq_labels = np.array(seq_labels)
    if verbose:
        print(f"  - Output sequences: {len(sequences)}")
        print(f"  - Sequence shape: {sequences.shape}")
        print(f"  - Labels shape: {seq_labels.shape}")
        print(f"{'='*60}\n")
    return sequences, seq_labels





