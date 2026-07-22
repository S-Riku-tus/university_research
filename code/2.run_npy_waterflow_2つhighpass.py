import os

from utils.dataloading.waterflow_preprocessing import (
    build_experiment_context,
    get_sorted_wav_files,
    save_spectrogram_chunks_with_snr,
)


fp = 500
fs = 400
gpass = 0.00001
gstop = 0.0001

sample_number = 672
MAX_FREQ_HZ = [2000, 3000, 5000, 10000, 15000, 22050]

SAVE_DATE = 20260722
DATASET_VERSION = "y_power"
CHUNK_SECONDS_LIST = [0.5, 1]

AUDIO_SAMPLES_USED = 2646000
SAVE_NPY = True
SAVE_SPECTROGRAM_PNG = True
PNG_WITH_AXES = True

BASE_EXPERIMENT_DIR = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3"
EXPERIMENT_NAMES = [
    "2025.06.18_0.3_3",
    "2025.07.09_0.3_1",
    "2025.06.11_0.3_2",
]
RECORDING_DIR_NAME = "録音データ_熱流束"
WATERFLOW_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"

SNR_list = [None, 0, -4, -8, -12, -16, -20]

PREPROCESS_CONFIG = {
    "fp": fp,
    "fs": fs,
    "gpass": gpass,
    "gstop": gstop,
    "sample_number": sample_number,
    "audio_samples_used": AUDIO_SAMPLES_USED,
    "save_npy": SAVE_NPY,
    "save_spectrogram_png": SAVE_SPECTROGRAM_PNG,
    "png_with_axes": PNG_WITH_AXES,
}


def run_all_experiments():
    for chunk_i, chunk_seconds in enumerate(CHUNK_SECONDS_LIST, start=1):
        print(
            f"========== chunk {chunk_i}/{len(CHUNK_SECONDS_LIST)}: "
            f"{chunk_seconds:g}s =========="
        )
        for exp_i, experiment_name in enumerate(EXPERIMENT_NAMES, start=1):
            context = build_experiment_context(
                experiment_name=experiment_name,
                base_experiment_dir=BASE_EXPERIMENT_DIR,
                recording_dir_name=RECORDING_DIR_NAME,
                waterflow_path=WATERFLOW_PATH,
                save_date=SAVE_DATE,
                dataset_version=DATASET_VERSION,
                chunk_seconds=chunk_seconds,
                script_path=__file__,
            )
            folder_path = context["folder_path"]

            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"recording folder not found: {folder_path}")

            if SAVE_NPY:
                os.makedirs(context["base_npy_save_folder_path"], exist_ok=True)
            if SAVE_SPECTROGRAM_PNG:
                os.makedirs(context["base_png_save_folder_path"], exist_ok=True)

            wav_files = get_sorted_wav_files(folder_path)

            label_count = len(context["heat_flux_label_by_index"])
            if label_count and label_count != len(wav_files):
                print(
                    f"Warning: heat flux labels={label_count}, "
                    f"wav files={len(wav_files)} in {experiment_name}"
                )

            print(
                f"========== experiment {exp_i}/{len(EXPERIMENT_NAMES)}: "
                f"{experiment_name} | wav={len(wav_files)} =========="
            )
            for max_freq_hz in MAX_FREQ_HZ:
                print(f"----- max_freq_hz={max_freq_hz} -----")
                for i, filename in enumerate(wav_files, start=1):
                    file_path = os.path.join(folder_path, filename)
                    save_spectrogram_chunks_with_snr(
                        file_path=file_path,
                        max_freq_hz=max_freq_hz,
                        chunk_seconds=chunk_seconds,
                        context=context,
                        config=PREPROCESS_CONFIG,
                        snr_db=SNR_list,
                    )
                    print(
                        f"---------- {i} / {len(wav_files)} ({filename}) done! ----------"
                    )

    print("---------- All experiments done! ----------")


if __name__ == "__main__":
    run_all_experiments()
