# VITS Fast Tuning Docker

VITS finetunining docker for fast speaker adaption TTS.


## Quick start

### Step 1: Download the latest VITS Fast Tuning Docker image

```bash
docker pull ljxha471758/vits-fast-tuning
```

Or build it up from source:

```bash
git clone https://github.com/fengwang/VITS-fast-fine-tuning-docker.git
cd VITS-fast-fine-tuning-docker
mkdir ./pretrained_models
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
docker build --file ./Dockerfile . -t vits
```

### Step 2: Prepare audios for finetunining

Rename all audios in a folder with the speaker name and a random number in the file name. For example, if there are two audios for speaker A, rename them as speaker_A_0.wav and speaker_A_1.wav.
A typical audio folder structure looks like:

```
/data/vits/trainingset_1/audios
├── yqt_102.mp3
├── yqt_214.mp3
├── yqt_23.mp3
└── yqt_9.mp3
```

in which there is only one speaker whose name is `yqt`, and there are four audios from her. Note that the audio file format can be arbitrary, but should be preprocessiable by ffmpeg.


### Step 3: Start finetunining

Suppose we will save the fine-tuned model to `/data/vits/model_yqt`, the following command is supposed to be:

```bash
ocker run --rm -it --gpus all --shm-size=16g -v /data/vits/trainingset_1/audios:/input -v /data/vits/model_yqt:/output vits-fast-tuning sh /workspace/VITS-fast-fine-tuning/scripts/run.sh
```

After finetuning, the model will be saved to `/data/vits/model_yqt/model` (the `/output/model` directory in the docker).

```
/data/vits/model_yqt/model
├── config.json
├── D_320.pth
├── D_330.pth
├── D_340.pth
├── D_350.pth
├── D_latest.pth
├── eval
│   └── events.out.tfevents.1709492024.055cbe3d6178.400.1
├── events.out.tfevents.1709492024.055cbe3d6178.400.0
├── G_320.pth
├── G_330.pth
├── G_340.pth
├── G_350.pth
├── githash
├── G_latest.onnx
├── G_latest.pth
├── lexicon.txt
├── rule.fst
├── tokens.txt
└── train.log

2 directories, 19 files
```


Note that:
0. a decent nvidia gpu is required;
1. if `G_latest.pth` and `D_latest.pth` exist in the `/output/model` director in the docker, the model will be continuously finetuned from that checkpoint.
2. the `config.json` file can be re-mapped to `/config/config.json` to adjust your custom configuration.


### Step 4: Start inference

The exported ONNX model can be inference using [sherpa framework](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#csukuangfj-vits-zh-hf-fanchen-c-chinese-1-female). Inferencing method using PyTorch will be added in the future.




## Acknowledgements
+ [Vits-Fast-Fine-Tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)


## License

+ [Apache License 2.0](https://github.com/fengwang/VITS-fast-fine-tuning-docker/blob/main/LICENSE)






