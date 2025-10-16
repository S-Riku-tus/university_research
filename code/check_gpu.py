import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("✅ GPUが正常に認識されています。")
    for device in gpu_devices:
        print("- ", device)
else:
    print("❌【問題】GPUがTensorFlowから認識されていません。")