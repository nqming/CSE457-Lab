"""Offline Vietnamese synthetic speech generator for CSE457 Lab 3.
Requires eSpeak (classic) executable; output uses only standard Python, NumPy, SciPy and SoundFile.
All data are synthetic TTS and are not a substitute for real recorded speech.
"""
from pathlib import Path
import os, subprocess, tempfile, shutil, hashlib, csv, json
import numpy as np
import librosa
from scipy.signal import resample_poly, lfilter
import soundfile as sf

ROOT=Path(__file__).resolve().parent
WORDS={'mo':'mở','dong':'đóng','len':'lên','xuong':'xuống','dung':'dừng'}
FS=16000
SEED=20261019
RNG=np.random.default_rng(SEED)
# Different eSpeak VN voice variants, prosody and timing. All utterances are fully synthesized.
CONFIGS=[
('vi', 130, 48, 0.29, 0.34),
('vi-hue', 146, 55, 0.32, 0.26),
('vi-sgn', 154, 63, 0.25, 0.32),
('vi', 163, 58, 0.30, 0.30),
('vi-hue', 121, 44, 0.34, 0.27),
('vi-sgn', 138, 69, 0.28, 0.35),
('vi', 174, 65, 0.27, 0.36),
('vi-sgn', 119, 51, 0.33, 0.29),
]
ROBUST_CONFIGS=[('vi-hue',159,60,0.29,0.31),('vi',113,40,0.31,0.29)]

def synth(word,voice,speed,pitch):
    with tempfile.NamedTemporaryFile(suffix='.wav',delete=False) as f:
        tmp=Path(f.name)
    try:
        subprocess.run(['espeak','-v',voice,'-s',str(speed),'-p',str(pitch),'-a','135',
                        '-w',str(tmp),word],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        x,sr=sf.read(tmp,dtype='float64',always_2d=False)
    finally:
        tmp.unlink(missing_ok=True)
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=FS:
        import math
        div=math.gcd(sr,FS)
        x=resample_poly(x,FS//div,sr//div)
    # Remove encoder lead/tail silence except short enough cushion.
    threshold=max(0.003,np.max(np.abs(x))*0.008)
    active=np.flatnonzero(np.abs(x)>threshold)
    if len(active)>0:
        x=x[max(active[0]-int(.03*FS),0):min(active[-1]+int(.03*FS),len(x)-1)+1]
    # Very short monosyllables (especially "đóng") yield too few HMM frames.
    # Lengthen short speech with a pitch-preserving phase vocoder, keeping each
    # token as ONE isolated word. Other words keep their original durations.
    min_speech_duration = 0.31
    if len(x) < int(min_speech_duration*FS):
        rate = len(x) / (min_speech_duration*FS)
        x = librosa.effects.time_stretch(x.astype(np.float32), rate=rate,
                                         n_fft=512, hop_length=128).astype(np.float64)
    return np.asarray(x,np.float64)

def produce(x, lead,tail, noise=False):
    # Normalize clean speech to fixed RMS for consistent training dynamics.
    rms=np.sqrt(np.mean(x*x)+1e-12)
    x=x * (0.075/max(rms,1e-8))
    peak=np.max(np.abs(x))
    if peak>0.82:x=x*(0.82/peak)
    pad1=np.zeros(int(lead*FS));pad2=np.zeros(int(tail*FS))
    y=np.concatenate([pad1,x,pad2])
    if noise:
        # Distinct, noise-degraded renders for a controlled synthetic robustness test.
        noise_raw=RNG.normal(size=y.size)
        smooth=lfilter([1],[1,-.89],noise_raw)
        smooth=(smooth/np.std(smooth))*0.012
        # Simulated short reverberation and light environmental hum.
        delayed=np.pad(y[:-int(.033*FS)],(int(.033*FS),0))*.13
        hum=.003*np.sin(2*np.pi*110*np.arange(len(y))/FS)
        y=y+delayed+smooth+hum
    else:
        # Quiet synthesized recording bed, separate random waveform for each utterance.
        y=y+RNG.normal(0,0.00018,size=len(y))
    y=np.clip(y,-.98,.98)
    return y

rows=[]
for lab,word in WORDS.items():
    for ix,(voice,speed,pitch,lead,tail) in enumerate(CONFIGS,1):
        part='train' if ix<=6 else 'test'
        path=ROOT/'dataset'/part/lab/f'{lab}_{ix:02d}.wav'
        path.parent.mkdir(parents=True,exist_ok=True)
        y=produce(synth(word,voice,speed,pitch),lead,tail,False)
        sf.write(path,y,FS,subtype='PCM_16')
        rows.append({'split':part,'label':lab,'word_vi':word,'filename':str(path.relative_to(ROOT)),
                     'engine':'espeak (offline synthetic)','voice':voice,'speed_wpm':speed,'pitch':pitch,
                     'noise_condition':'quiet_synthetic','duration_s':round(len(y)/FS,3),
                     'sample_rate':FS,'channels':1,'subtype':'PCM_16',
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for ix,(voice,speed,pitch,lead,tail) in enumerate(ROBUST_CONFIGS,1):
        path=ROOT/'dataset'/'robustness'/lab/f'{lab}_noise_{ix:02d}.wav'
        path.parent.mkdir(parents=True,exist_ok=True)
        y=produce(synth(word,voice,speed,pitch),lead,tail,True)
        sf.write(path,y,FS,subtype='PCM_16')
        rows.append({'split':'robustness','label':lab,'word_vi':word,'filename':str(path.relative_to(ROOT)),
                     'engine':'espeak (offline synthetic)','voice':voice,'speed_wpm':speed,'pitch':pitch,
                     'noise_condition':'simulated_noise_reverb','duration_s':round(len(y)/FS,3),
                     'sample_rate':FS,'channels':1,'subtype':'PCM_16',
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
manifest=ROOT/'dataset'/'audio_manifest.csv'
with open(manifest,'w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
print('Generated',len(rows),'synthetic WAV files; total bytes',sum((ROOT/x['filename']).stat().st_size for x in rows))
for p in sorted((ROOT/'dataset'/'train').rglob('*.wav'))[:3]:
 x,sr=sf.read(p);print(str(p.relative_to(ROOT)),len(x)/sr,round(float(np.max(np.abs(x))),3))
