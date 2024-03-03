FROM ubuntu:jammy
ENV DEBIAN_FRONTEND=noninteractive

# install system packages
RUN apt-get clean && apt update -y && apt-get -y upgrade --fix-missing
RUN apt install -y software-properties-common  lsb-release
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt install -y build-essential wget curl git screen ffmpeg cmake vim screen python3.11 python3.11-dev

# install torch
WORKDIR /tmp
RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt update -y && apt install -y libcudnn8 libcudnn8-dev

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 -
RUN rm -rf /usr/bin/python && ln -s /usr/bin/python3.11 /usr/bin/python
RUN rm -rf /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3

# install python packages
RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3.11 -m pip install imageio moviepy Cython librosa==0.9.2 scikit-learn scipy tensorboard unidecode pyopenjtalk jamo pypinyin jieba protobuf cn2an inflect eng_to_ipa ko_pron indic_transliteration num_thai opencc demucs openai-whisper gradio
RUN python3.11 -m pip install ffmpeg-python
RUN python3.11 -m pip install onnx


# copy repo to workspace
ADD .  /workspace/VITS-fast-fine-tuning
RUN mkdir -p /workspace/VITS-fast-fine-tuning/monotonic_align/monotonic_align

# build monotonic align
WORKDIR /workspace/VITS-fast-fine-tuning/monotonic_align
RUN python setup.py build_ext --inplace

# download auxiliary data for training
WORKDIR /workspace/VITS-fast-fine-tuning

RUN ln -s /workspace/VITS-fast-fine-tuning/pretrained_models /pretrained_models
RUN ln -s /pretrained_models/D_trilingual.pth /pretrained_models/D_0.pth
RUN ln -s /pretrained_models/G_trilingual.pth /pretrained_models/G_0.pth
RUN ln -s /workspace/VITS-fast-fine-tuning/configs /configs
RUN ln -s /configs/uma_trilingual.json /configs/finetune_speaker.json

# pipeline:
# 0. suppose all input files are in /input and output model is in /output
#    - input: /input/说话人名称_xxxxx.wav
#    - input: /input/说话人名称_yyyyy.wav
#    in range [0, 999999]
# 1. convert long audio to wave, example command: `python3 /app/convert_long_audio_to_wave.py /input /workspace/VITS-fast-fine-tuning/denoised_audio/`
#### /workspace/VITS-fast-fine-tuning/denoised_audio -> /output/denoised_audio
# 2. process all wav files with whipser model, example command: `cd /workspace/VITS-fast-fine-tuning && python scripts/long_audio_transcribe.py --whisper_size large` or large-v2 or large-v3?
# 3. process all text data, example command: `python preprocess_v2.py`
# 3.5. change batch size in config
# 4. start training, example command: `python finetune_speaker_v2.py -m /output --max_epochs "{Maximum_epochs}" --drop_speaker_embed True`
#                                  or `python finetune_speaker_v2.py -m /output --max_epochs "{Maximum_epochs}" --drop_speaker_embed False --cont True` to continue the training
# 5. vc inference, example command: `python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth --share True`
# 6. command inference, example command: `python cmd_inference.py -m 模型路径/output/xxx -c 配置文件路径/output/config.json -o 输出文件路径 -l 输入的语言 -t 输入文本 -s 合成目标说话人名称`



#ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["bash", "/workspace/VITS-fast-fine-tuning/scripts/run.sh"]

