
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
try:
    from numba import njit
except ImportError:
    njit = None
import csv

FS = 16000
FRAME_MS, HOP_MS = 25, 10
WIN, HOP = int(FS*FRAME_MS/1000), int(FS*HOP_MS/1000)
ALPHA = 0.97
NFFT, N_MELS, N_MFCC = 512, 24, 13
TOP_DB, MARGIN_MS = 35, 50
LABELS = ['khong', 'mot', 'hai', 'ba', 'bon']
VN = {'khong':'không','mot':'một','hai':'hai','ba':'ba','bon':'bốn'}

def load_audio(path):
    y, sr = librosa.load(path, sr=FS, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def frame_signal(y):
    return librosa.util.frame(y, frame_length=WIN, hop_length=HOP).T

def energy_zcr(y):
    frames = frame_signal(y)
    window = np.hamming(WIN)
    f = frames * window
    energy = np.sum(f**2, axis=1)
    rms = np.sqrt(np.mean(f**2, axis=1))
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=WIN, hop_length=HOP, center=False
    )[0][:len(frames)]
    times = (np.arange(len(frames))*HOP + WIN/2)/FS
    return times, energy, rms, zcr

def trim_energy(y, top_db=TOP_DB, margin_ms=MARGIN_MS):
    _, idx = librosa.effects.trim(y, top_db=top_db,
                                  frame_length=WIN, hop_length=HOP)
    margin = int(FS*margin_ms/1000)
    s = max(0, idx[0]-margin)
    e = min(len(y), idx[1]+margin)
    return y[s:e], (s, e)

def mfcc_feature(y, use_cmn=True, use_delta=False):
    y = lfilter([1.0, -ALPHA], [1.0], y)
    M = librosa.feature.mfcc(
        y=y, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS,
        n_fft=NFFT, win_length=WIN, hop_length=HOP,
        window='hamming', center=False
    )
    if use_delta:
        D = librosa.feature.delta(M)
        M = np.vstack([M, D])
    if use_cmn:
        M = M - np.mean(M, axis=1, keepdims=True)
    return M.T

def _dtw_core(X, Y):
    N, M = len(X), len(Y)
    D = np.full((N+1, M+1), np.inf)
    D[0,0] = 0.0
    back = np.zeros((N+1, M+1, 2), dtype=np.int32)
    for i in range(1, N+1):
        for j in range(1, M+1):
            s = 0.0
            for q in range(X.shape[1]):
                z = X[i-1,q] - Y[j-1,q]
                s += z*z
            local = np.sqrt(s)
            a, b, c = D[i-1,j], D[i,j-1], D[i-1,j-1]
            if a <= b and a <= c:
                D[i,j] = local + a; back[i,j,0] = i-1; back[i,j,1] = j
            elif b <= c:
                D[i,j] = local + b; back[i,j,0] = i; back[i,j,1] = j-1
            else:
                D[i,j] = local + c; back[i,j,0] = i-1; back[i,j,1] = j-1
    return D, back

if njit is not None:
    _dtw_core = njit(cache=True)(_dtw_core)

def dtw_distance(X, Y):
    D, back = _dtw_core(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64))
    path=[]
    i,j=len(X),len(Y)
    while i>0 or j>0:
        path.append((i-1,j-1))
        i,j=int(back[i,j,0]), int(back[i,j,1])
    path.reverse()
    return D[len(X),len(Y)]/max(len(path),1), path, D[1:,1:]

def plot_wave_energy_zcr(path, out):
    y=load_audio(path)
    t,e,r,z=energy_zcr(y)
    fig, ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    tt=np.arange(len(y))/FS
    ax[0].plot(tt,y); ax[0].set_ylabel('Amplitude'); ax[0].set_title(f'Waveform - {path.stem}')
    ax[1].plot(t,10*np.log10(e+1e-12)); ax[1].set_ylabel('Log-energy (dB)')
    ax[2].plot(t,z); ax[2].set_ylabel('ZCR'); ax[2].set_xlabel('Time (s)')
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)

def plot_mfcc(path, out, trim=True):
    y=load_audio(path)
    if trim: y,_=trim_energy(y)
    M=mfcc_feature(y)
    fig,ax=plt.subplots(figsize=(10,4.5))
    im=ax.imshow(M.T,origin='lower',aspect='auto',cmap='magma')
    ax.set_xlabel('Frame'); ax.set_ylabel('MFCC coefficient')
    ax.set_title(f'MFCC - {path.stem}' + (' (trimmed)' if trim else ''))
    fig.colorbar(im,ax=ax,label='Coefficient')
    fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)

def plot_dtw(X,Y,title,out):
    cost,path,C=dtw_distance(X,Y)
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(C,origin='lower',aspect='auto',cmap='viridis')
    px=[p[1] for p in path]; py=[p[0] for p in path]
    ax.plot(px,py,linewidth=2)
    ax.set_xlabel('Template frame'); ax.set_ylabel('Test frame')
    ax.set_title(f'{title}\nDTW_norm = {cost:.4f}')
    fig.colorbar(im,ax=ax,label='Euclidean distance')
    fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)
    return cost

def build_templates(root, trim=True, use_delta=False, n_train=3):
    templates={lab:[] for lab in LABELS}
    for lab in LABELS:
        files=sorted((Path(root)/lab).glob('*.wav'))[:n_train]
        for f in files:
            y=load_audio(f)
            if trim: y,_=trim_energy(y)
            templates[lab].append((f.name,mfcc_feature(y,use_delta=use_delta)))
    return templates

def recognize(path, templates, trim=True, use_delta=False):
    y=load_audio(path)
    if trim: y,_=trim_energy(y)
    X=mfcc_feature(y,use_delta=use_delta)
    scores=[]
    for lab, refs in templates.items():
        for name,R in refs:
            d=dtw_distance(X,R)[0]
            scores.append((d,lab,name))
    scores.sort()
    return scores[0][1], scores

def evaluate(root, trim=True, use_delta=False):
    templates=build_templates(root,trim,use_delta)
    y_true=[]; y_pred=[]; rows=[]
    for lab in LABELS:
        for f in sorted((Path(root)/lab).glob('*.wav'))[3:]:
            pred,scores=recognize(f,templates,trim,use_delta)
            y_true.append(lab); y_pred.append(pred)
            rows.append([f.name, VN[lab], VN[pred], pred==lab,
                         scores[0][0], scores[1][0], scores[2][0]])
    acc=accuracy_score(y_true,y_pred)
    cm=confusion_matrix(y_true,y_pred,labels=LABELS)
    return acc,cm,rows,templates

def save_results(rows,out):
    with open(out,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['test_file','true_label','pred_label','correct','top1_dtw','top2_dtw','top3_dtw'])
        w.writerows(rows)

def main():
    root=Path(__file__).resolve().parent
    fig=root/'figures'; res=root/'results'
    data=root/'dataset'
    # A/B/C: time-domain features and endpoint
    sample=[data/'ba'/'ba_01.wav',data/'mot'/'mot_01.wav',data/'khong'/'khong_01.wav']
    for p in sample: plot_wave_energy_zcr(p,fig/f'{p.parent.name}_energy_zcr.png')
    durations=[]
    for p in sorted(data.glob('*/*.wav')):
        y=load_audio(p); yt,(s,e)=trim_energy(y)
        durations.append([p.parent.name,p.name,len(y)/FS,len(yt)/FS])
    with open(res/'endpoint_durations.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['label','file','duration_before_s','duration_after_s']); w.writerows(durations)
    # D: MFCC
    for p in [sample[0],sample[1]]:
        plot_mfcc(p,fig/f'{p.parent.name}_mfcc.png')
    # E: same word and different word
    def feat(p):
        y,_=trim_energy(load_audio(p)); return mfcc_feature(y)
    X1=feat(data/'ba'/'ba_04.wav'); Y1=feat(data/'ba'/'ba_05.wav')
    X2=feat(data/'ba'/'ba_04.wav'); Y2=feat(data/'bon'/'bon_04.wav')
    same=plot_dtw(X1,Y1,'Same word: ba_04 vs ba_05',fig/'dtw_same_word.png')
    diff=plot_dtw(X2,Y2,'Different words: ba_04 vs bon_04',fig/'dtw_different_word.png')
    # F/G baseline
    acc,cm,rows,_=evaluate(data,trim=True,use_delta=False)
    save_results(rows,res/'results_baseline.csv')
    figcm,ax=plt.subplots(figsize=(6,5))
    disp=ConfusionMatrixDisplay(cm,display_labels=[VN[x] for x in LABELS])
    disp.plot(ax=ax,cmap='Blues',values_format='d',colorbar=False)
    ax.set_title(f'Confusion matrix - MFCC 13 + DTW (accuracy={acc*100:.1f}%)')
    figcm.tight_layout(); figcm.savefig(fig/'confusion_matrix_baseline.png',dpi=160); plt.close(figcm)
    # E1 no trim
    acc_nt,cm_nt,rows_nt,_=evaluate(data,trim=False,use_delta=False)
    save_results(rows_nt,res/'results_no_trim.csv')
    # E2 delta
    acc_d,cm_d,rows_d,_=evaluate(data,trim=True,use_delta=True)
    save_results(rows_d,res/'results_mfcc_delta.csv')
    summary=[
        ['experiment','accuracy_percent'],
        ['E1 no endpoint trim',f'{acc_nt*100:.2f}'],
        ['Baseline trim + MFCC13',f'{acc*100:.2f}'],
        ['E2 trim + MFCC13+Delta',f'{acc_d*100:.2f}'],
        ['DTW same word (ba)',f'{same:.4f}'],
        ['DTW different word (ba vs bon)',f'{diff:.4f}'],
    ]
    with open(res/'summary.csv','w',newline='',encoding='utf-8-sig') as f:
        csv.writer(f).writerows(summary)
    print('\n'.join(' | '.join(r) for r in summary))

if __name__=='__main__':
    main()
