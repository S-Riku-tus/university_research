from tensorflow.python.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utils.models.regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.calc_r2_auc import AUCorR2Calculation

#modelへ保存データを読み込み
regressionmodelmaker = RegressionModelMaker((224, 224, 1))
model = regressionmodelmaker.alexnet()
model.compile(optimizer=Adam(), loss='mean_squared_error')
model.load_weights(r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\ensemble\channel=1\weight_average_highpass\100%\all_weights\no_noise\AlexNet_fold1_no_noise.h5")

model.summary()
