import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def window_partition(x, window_size):
    shape = tf.shape(x)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W):
    shape = tf.shape(windows)
    C = shape[-1]
    denominator = float(H * W / window_size / window_size)
    B_float = tf.cast(shape[0], dtype=tf.float32) / denominator
    B = tf.cast(B_float, dtype=tf.int32)
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x

# ★★★ ここからが修正箇所 ★★★
class WindowAttention(layers.Layer):
    """ウィンドウ内でのMulti-Head Self-Attentionを計算するレイヤー"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout)
        self.proj = layers.Dense(dim)
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=[(2 * window_size - 1) * (2 * window_size - 1), num_heads],
            initializer=tf.initializers.zeros(),
            trainable=True,
        )
        coords_h = tf.range(self.window_size)
        coords_w = tf.range(self.window_size)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)
        coords_flatten = tf.reshape(coords, [-1, 2])
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords = relative_coords + (self.window_size - 1)
        relative_coords = tf.reduce_sum(relative_coords * [2 * self.window_size - 1, 1], axis=-1)
        self.relative_position_index = tf.Variable(
            relative_coords, trainable=False, name="relative_position_index"
        )

    def call(self, x, mask=None):
        shape = tf.shape(x)
        B_, N, C = shape[0], shape[1], shape[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ tf.transpose(k, [0, 1, 3, 2])
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, self.relative_position_index
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, [self.window_size * self.window_size, self.window_size * self.window_size, -1]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + relative_position_bias[None, ...]
        
        if mask is not None:
            # マスクの形状(nW, N, N)をアテンションスコア(B_, num_heads, N, N)に加算できるようにする
            # B_ = B * nW なので、マスクをB回繰り返す
            nW = tf.shape(mask)[0]
            batch_size = B_ // nW
            
            # マスクを(nW, 1, N, N)に拡張し、バッチサイズ分リピート
            mask_repeated = tf.repeat(mask[:, None, :, :], repeats=batch_size, axis=0)
            # attnの形状(B_, num_heads, N, N)に合わせるためにreshape
            attn = tf.reshape(attn, [B_, self.num_heads, N, N]) + mask_repeated
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
        
        attn = self.dropout(attn)
        x = (attn @ v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        x = self.proj(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    def call(self, x):
        x = self.proj(x)
        static_shape = x.shape
        B = tf.shape(x)[0]
        H, W, C = static_shape[1], static_shape[2], static_shape[3]
        x = tf.reshape(x, [B, H * W, C])
        x = self.norm(x)
        return x

class PatchMerging(layers.Layer):
    def __init__(self, num_patch, dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    def call(self, x):
        H, W = self.num_patch
        shape = tf.shape(x)
        B, L, C = shape[0], shape[1], shape[2]
        x = tf.reshape(x, [B, H, W, C])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, [B, (H // 2) * (W // 2), 4 * C])
        x = self.norm(x)
        x = self.reduction(x)
        return x
        
class SwinTransformerBlock(layers.Layer):
    """Swin Transformerの基本ブロック"""
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, dropout)
        self.drop_path = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = models.Sequential([
            layers.Dense(int(dim * mlp_ratio)),
            layers.Activation("gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
        self.attn_mask = None

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.num_patch
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(tf.convert_to_tensor(img_mask, dtype=tf.float32), self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            self.attn_mask = tf.cast(tf.where(attn_mask != 0, -100.0, 0.0), dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        H, W = self.num_patch
        shape = tf.shape(x)
        B, L, C = shape[0], shape[1], shape[2]
        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H * W, C])
        x = self.drop_path(x)
        x = shortcut + x
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = shortcut2 + x
        return x


class SwinTransformerModelMaker:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        pass
    def swin_transformer_regressor(
        self,
        patch_size=4,
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        dropout_rate=0.1
    ):
        """全ての部品を組み立ててSwin Transformer回帰モデルを構築する"""
        inputs = layers.Input(shape=self.input_shape)
        x = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)(inputs)
        patch_resolution = (self.input_shape[0] // patch_size, self.input_shape[1] // patch_size)
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            for j in range(depth):
                x = SwinTransformerBlock(
                    dim=int(embed_dim * 2**i),
                    num_patch=patch_resolution,
                    num_heads=num_head,
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout_rate,
                )(x)
            if i < len(depths) - 1:
                x = PatchMerging(patch_resolution, dim=int(embed_dim * 2**i))(x)
                patch_resolution = (patch_resolution[0] // 2, patch_resolution[1] // 2)
        x = layers.LayerNormalization(epsilon=1e-5)(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = models.Model(inputs, outputs)
        return model