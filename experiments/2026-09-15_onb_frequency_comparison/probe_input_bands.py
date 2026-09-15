"""Bounded waveform diagnostic: 18 WAVs x chunks 0/30/59; no model fitting.

Reconstruct clean and -20 dB saved arrays, then compare signal/noise STFT power
on the original common frequency grid, before bandwidth-dependent resizing.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from analyze_3khz import ROOT, longpath, relative, NOISES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise SystemExit('Use a new empty output directory.')
    sys.path.insert(0,str(ROOT/'code'))
    source=ROOT/'code/2.run_npy_waterflow_2つhighpass.py'
    spec=importlib.util.spec_from_file_location('preprocessing_reference',source)
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    data=longpath(ROOT/'Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/data/npy/waterflow_20260817_1s')
    hashes={relative(source):hashlib.sha256(source.read_bytes()).hexdigest()}

    def read(p):
        raw=p.read_bytes();hashes[relative(p)]=hashlib.sha256(raw).hexdigest();return raw

    def csv(p):return pd.read_csv(io.BytesIO(read(p)),keep_default_na=False)

    manifests={}
    paired=[]
    for freq in ['3','22']:
        manifests[freq]=json.loads(read(data/f'maxfreq={freq}kHz/preprocess_manifest.json'))
    for snr in NOISES:
        folder='heatflux_no_noise' if snr=='no_noise' else f'heatflux_reference_SNR={snr}'
        a=csv(data/f'maxfreq=3kHz/{folder}/chunk_manifest.csv')
        b=csv(data/f'maxfreq=22kHz/{folder}/chunk_manifest.csv')
        pd.testing.assert_frame_equal(a,b)
        paired.append({'snr':snr,'rows':len(a),'identical_metadata_between_frequencies':True})
        if snr=='-20':meta=a
    m=manifests['3']
    nfft=m['stft']['n_fft'];nframes=m['stft']['frames'];hop=m['stft']['hop_length_samples']
    sr=m['samplerate_hz'];window=np.hanning(nfft)
    frequencies=np.arange(nfft//2)*sr/nfft
    band_edges=[(500,1000),(1000,2000),(2000,3000),(3000,5000),(5000,10000),(10000,15000),(15000,22050),(500,3000),(3000,22050),(0,22050)]
    bands=[(lo,hi,f'{lo}-{hi}') for lo,hi in band_edges]

    def stft(x):
        inds=np.arange(nframes)[:,None]*hop+np.arange(nfft)[None,:]
        fft=np.fft.fft(x[inds]*window,axis=1)[:,:nfft//2]
        return (np.abs(fft)*2/window.sum())**2

    noise_path=Path(m['waterflow_path'])
    hashes[relative(noise_path)]=hashlib.sha256(noise_path.read_bytes()).hexdigest()
    # Production uses first channel for the water-flow record.
    noise=prep._load_and_filter(str(noise_path),first_channel=True)
    rows=[];checks=[];spectra=[]
    for wav,g in meta[meta.chunk_index.isin([0,30,59])].groupby('source_wav_name',sort=True):
        wavpath=Path(m['source_audio_folder'])/wav
        hashes[relative(wavpath)]=hashlib.sha256(wavpath.read_bytes()).hexdigest()
        signal=prep._load_and_filter(str(wavpath))
        for _,r in g.iterrows():
            offset=int(r.noise_offset_samples);start=int(r.chunk_start_sample)
            clean=signal[start:start+sr]
            raw=noise[offset:offset+sr]
            assert len(raw)==sr
            added=raw*float(r.noise_scale)
            assert np.isclose(np.mean(clean**2),r.signal_chunk_power,rtol=1e-8)
            assert np.isclose(np.mean(added**2),r.scaled_noise_chunk_power,rtol=1e-8)
            sp=stft(clean);npow=stft(added)
            spectra.append((sp.mean(axis=0),npow.mean(axis=0)))
            for lo,hi,label in bands:
                mask=(frequencies>=lo)&(frequencies<hi)
                s=sp[:,mask].sum(axis=1).mean();n=npow[:,mask].sum(axis=1).mean()
                rows.append(dict(source_wav_id=r.source_wav_id,chunk_index=r.chunk_index,heat_flux=r.heat_flux,
                    band_hz=label,signal_stft_power=s,noise20_stft_power=n,
                    realized_band_snr_at_ref_minus20=10*np.log10(s/n),
                    realized_band_snr_at_ref_zero=10*np.log10(s/n)+20))
            for snr,power in [('no_noise',sp),('-20',stft(clean+added))]:
                folder='heatflux_no_noise' if snr=='no_noise' else 'heatflux_reference_SNR=-20'
                for freq in ['3','22']:
                    maxk=manifests[freq]['max_frequency_bin_index']
                    made=resize(power[:,:maxk+1],(224,224)).astype('float32')
                    stored=np.load(io.BytesIO(read(data/f'maxfreq={freq}kHz/{folder}'/r.sample_filename)))
                    difference=np.max(np.abs(made-stored))
                    assert np.allclose(made,stored,rtol=3e-5,atol=1e-17)
                    checks.append(dict(source_wav_id=r.source_wav_id,chunk_index=r.chunk_index,
                        snr=snr,maxfreq_khz=freq,max_absolute_difference=float(difference)))
        print(f'Checked {wav}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'band_snr_samples.csv',index=False,encoding='utf-8-sig')
    summary=df.groupby('band_hz',sort=False).agg(n=('heat_flux','size'),
        median_snr=('realized_band_snr_at_ref_minus20','median'),min_snr=('realized_band_snr_at_ref_minus20','min'),
        max_snr=('realized_band_snr_at_ref_minus20','max')).reset_index()
    summary.to_csv(out/'band_snr_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(out/'saved_array_reconstruction.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(paired).to_csv(out/'paired_metadata_verification.csv',index=False,encoding='utf-8-sig')
    # A descriptive association across 18 recordings, never a predictive validation result.
    associations=[]
    for band,g in df.groupby('band_hz',sort=False):
        wav=g.groupby('source_wav_id').agg(y=('heat_flux','first'),power=('signal_stft_power','median'))
        associations.append(dict(band_hz=band,n_wavs=len(wav),pearson_log_band_power_with_heatflux=
                                 np.corrcoef(np.log10(wav.power),wav.y)[0,1]))
    pd.DataFrame(associations).to_csv(out/'clean_band_associations.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    s=np.median(np.array([a for a,b in spectra]),axis=0)
    n=np.median(np.array([b for a,b in spectra]),axis=0)
    axs[0].plot(frequencies/1000,10*np.log10(s+1e-30),label='Clean signal')
    axs[0].plot(frequencies/1000,10*np.log10(n+1e-30),label='Added water-flow noise (ref -20 dB)')
    axs[0].set(xlabel='Frequency (kHz)',ylabel='Median STFT bin power (dB, arbitrary reference)')
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    display=summary.iloc[:7]
    axs[1].plot(range(7),display.median_snr,marker='o')
    axs[1].set_xticks(range(7),display.band_hz,rotation=40)
    axs[1].axhline(0,color='black',lw=.8)
    axs[1].set(xlabel='Physical band (Hz)',ylabel='Median in-band SNR at reference -20 dB')
    axs[1].grid(alpha=.2)
    fig.suptitle('54 paired chunks: 18 source WAVs x chunk 0/30/59; original 32.8125 Hz STFT grid\nSignal/noise inspected separately before resizing; no model fit or physical-event attribution')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'band_signal_noise.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    (out/'verification.json').write_text(json.dumps(dict(created_at=datetime.now().astimezone().isoformat(),
        selected_chunks=[0,30,59],n_wavs=18,n_chunks=54,n_arrays_checked=len(checks),
        sampling='Prespecified first, middle and last chunks per WAV; descriptive, not 54 independent recordings',
        source_sha256=hashes),ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
