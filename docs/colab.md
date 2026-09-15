# Running s2s-lab on Colab

Colab is the development GPU: a free, shared T4. Its numbers are development
numbers -- the GPU is shared and the runtime changes without notice. Reportable
GPU numbers come from Modal.

## On the laptop, after each commit you want on Colab

    mkdir -p ../colab-upload
    git bundle create ../colab-upload/s2s-lab.bundle main
    tar -czf ../colab-upload/probe-audio.tgz -C probes/endpointing/audio .

Upload both files to Google Drive at `My Drive/s2s-lab/colab-upload/`,
replacing the old ones. They stay private to your Drive.

## Every Colab session

Runtime -> Change runtime type -> **T4 GPU**. A TPU will not work: faster-whisper
needs CUDA. Then run these cells in order. Everything in `/content` is wiped
when a session ends; only the Drive files persist.

**1. Is there a GPU?**

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| CUDA available:", torch.cuda.is_available())
```

**2. Connect Drive** (approve the pop-up)

```python
from google.colab import drive
drive.mount("/content/drive")
```

**3. Code and audio**

A fresh clone has no audio folder (`*.wav` is gitignored), so it is created
before unpacking.

```python
DRIVE = "/content/drive/MyDrive/s2s-lab/colab-upload"
import os
for f in ("s2s-lab.bundle", "probe-audio.tgz"):
    assert os.path.exists(f"{DRIVE}/{f}"), f"missing: {DRIVE}/{f}"
%cd /content
!rm -rf s2s-lab
!git clone -q -b main {DRIVE}/s2s-lab.bundle s2s-lab
%cd /content/s2s-lab
!git log --oneline -1
!mkdir -p probes/endpointing/audio
!tar -xzf {DRIVE}/probe-audio.tgz -C probes/endpointing/audio
!ls probes/endpointing/audio | wc -l
```

**4. Install**

Colab already has torch with CUDA, numpy, scipy and soundfile; do not install
`requirements/base.txt` or the CPU torch here. Colab runs Python 3.13, and
kokoro 0.9.4 and misaki 0.9.4 declare `Requires-Python <3.13`. They work on
3.13 (all tests pass, same output as 3.10), so the check is overridden. spaCy is
pinned to 3.8.16 because it has a 3.13 wheel; without the pin, pip tries to
build spaCy from source and fails.

```python
!pip install -q --ignore-requires-python \
    --only-binary=spacy,thinc,blis,cymem,preshed,murmurhash,srsly \
    spacy==3.8.16 praat-parselmouth==0.4.7 -r requirements/r0.txt
```

**5. Let faster-whisper find the CUDA libraries**

A precaution: CTranslate2 loads cuDNN and cuBLAS by name, and Colab's torch
keeps them inside its own package folders.

```python
import glob, os, site
nvidia_libs = [p for sp in site.getsitepackages()
               for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))]
os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs + [os.environ.get("LD_LIBRARY_PATH", "")])
print(len(nvidia_libs), "NVIDIA library folders added")
```

**6. Same audio, same tests**

```python
!PYTHONPATH=src python -m s2slab.probes verify
!python -m pytest -q
```

**7. One turn, and listen**

```python
!PYTHONPATH=src python -m s2slab.r0 probes/endpointing/audio/f02.wav r0-gpu
from IPython.display import Audio
Audio("results/r0-gpu/f02.reply.wav")
```

**8. All 30 probe clips**

```python
!PYTHONPATH=src python -m s2slab.rungs r0-gpu
!PYTHONPATH=src python -m s2slab.rungs r0-gpu-bigger
```

Results are recorded on Notion page 20, not here.