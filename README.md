# 🎵 MusicGen Fine-Tuning: Generating Novel Music Compositions

Fine-tune Meta's open-source **MusicGen** (a Transformer "LLM for audio") with **LoRA** on a small genre-labelled dataset, then generate new instrumental clips from plain-text prompts.

![MusicGen Studio](assets/musicgen-studio.png)

*MusicGen Studio: type a prompt, pick a duration, generate, play and download the WAV (model: `facebook/musicgen-small`).*

---

## Overview

| | |
|---|---|
| **Model** | [`facebook/musicgen-small`](https://huggingface.co/facebook/musicgen-small): decoder-only Transformer over EnCodec audio tokens, conditioned on a T5 text encoder |
| **Fine-tuning** | LoRA (rank 16) via [`musicgen-dreamboothing`](https://github.com/ylacombe/musicgen-dreamboothing) and Hugging Face `Trainer` |
| **Dataset** | [`lewtun/music_genres_small`](https://huggingface.co/datasets/lewtun/music_genres_small): 1,000 clips, 10 genres (FMA-derived) |
| **Output** | Short 32 kHz mono clips conditioned on a text prompt |
| **Hardware** | Google Colab, single T4 GPU |

**Genres:** Chiptune / Glitch, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Punk, Rock.

### How it works

1. Audio is converted to discrete tokens by a frozen **EnCodec** codec (32 kHz, 4 codebooks, 50 Hz).
2. A **causal Transformer decoder** predicts the next audio token, exactly like a text LLM predicts the next word.
3. Generation is **text-conditioned**: a frozen T5 encoder embeds the prompt and the decoder cross-attends to it.
4. Only small **LoRA adapters** are trained; the pretrained weights stay frozen. This is cheap and limits catastrophic forgetting.

> **Note on TensorFlow:** MusicGen's official weights are released for PyTorch (via `transformers`), so this project uses the PyTorch backend.

---

## Repository contents

```
.
├── MusicGen_Finetuning_Novel_Compositions_FIXED.ipynb   # full pipeline (Colab)
├── assets/
│   └── musicgen-studio.png                              # UI screenshot
└── README.md
```

## Pipeline (notebook)

| Step | What happens |
|---|---|
| **Setup** | Install libraries, clone the training script, log in to Hugging Face |
| **0. Dataset** | Load `lewtun/music_genres_small`, inspect genre distribution, listen to a clip |
| **1. Preprocess** | Build captions from genre labels (e.g. `an electronic music track`), keep 1–30 s clips, take a 300-clip subset, 90/10 train/test split, push to your own Hub repo |
| **2. Inputs** | Show how text and audio become model inputs (T5 token ids, EnCodec codes) |
| **3. Model** | Load MusicGen, inspect its three components, preview the LoRA setup |
| **4. Config** | Assemble training arguments (fp16, gradient accumulation and checkpointing, LoRA) |
| **4b. Patch** | Make the 2024 training script compatible with current `transformers` |
| **5. Train** | LoRA fine-tuning, 3 epochs, eval loss every 25 steps |
| **6. Generate** | Generate clips for seen genres and one unseen genre (Jazz), compare base vs. fine-tuned, export adapters |

### Key training settings

| Setting | Value |
|---|---|
| LoRA | r = 16, α = 16, dropout 0.05 |
| Epochs | 3 |
| Batch | 2 per device × 8 gradient accumulation |
| Learning rate | 2e-4 |
| Precision | fp16 + gradient checkpointing |
| Guidance scale (training) | 1.0 |
| Pad / decoder-start token | 2048 |

---

## Getting started

### Requirements

- A Google account and Colab with a **GPU runtime** (`Runtime → Change runtime type → T4 GPU`)
- A free [Hugging Face account](https://huggingface.co/join) and an **access token with write permission** ([create one](https://huggingface.co/settings/tokens)); it is needed to download MusicGen and to push the captioned dataset to your own namespace

### Run

1. Open `MusicGen_Finetuning_Novel_Compositions_FIXED.ipynb` in Colab.
2. Run the cells in order. Paste your Hugging Face token when the login widget appears.
3. Training writes LoRA adapters to `./musicgen-small-lora-genres`.
4. Step 6 saves generated clips to `generated_music/` and zips the adapters as `final_lora_adapters.zip`.

### Load the fine-tuned adapters

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenForConditionalGeneration

base = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small", torch_dtype=torch.float16
).to("cuda")
model = PeftModel.from_pretrained(base, "final_lora_adapters").to("cuda").eval()
processor = AutoProcessor.from_pretrained("final_lora_adapters")

inputs = processor(text=["a punk music track"], padding=True, return_tensors="pt").to("cuda")
audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=512)
```

`max_new_tokens=512` is about 10 seconds of audio (50 tokens per second).

### Prompting tips

Training captions follow the pattern `a/an <genre> music track`, so prompts in that format match what the model saw. The Studio UI also accepts free-form descriptions such as *"a calm classical piano piece"*, which rely on the base model's pretrained knowledge.

---

## MusicGen Studio (web UI)

The screenshot above shows the demo interface: a prompt box, one-click example prompts, a duration slider, a **Generate music** button, an audio player and a **Download WAV** link.

<!-- TODO: add the command to launch the Studio app (e.g. `python app.py`) and its dependencies -->

---

## Known limitations

- Only 300 clips and genre-only captions are used, so the model learns coarse genre style, not fine control over mood or instruments.
- Output is short, mono, 32 kHz audio; long-form structure is not modelled.
- Prompts for genres outside the training set (for example Jazz) fall back on the base model's pretrained behaviour.
- Judge results by listening; eval loss alone is a weak indicator of musical quality.

## Acknowledgements

- [MusicGen](https://arxiv.org/abs/2306.05284), Copet et al., Meta AI
- [`musicgen-dreamboothing`](https://github.com/ylacombe/musicgen-dreamboothing) by Yoach Lacombe
- [`lewtun/music_genres_small`](https://huggingface.co/datasets/lewtun/music_genres_small) and the Free Music Archive
- Hugging Face `transformers`, `datasets`, `peft`

## License

Add your license here (for example MIT). Note that MusicGen's pretrained weights are released under **CC-BY-NC 4.0**, which restricts commercial use.
