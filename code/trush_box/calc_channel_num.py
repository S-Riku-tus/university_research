import wave
# WAVファイルを読み込みモードで開く
with wave.open(r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav", 'rb') as wf:
    channels = wf.getnchannels()
    print(f'チャンネル数: {channels}')