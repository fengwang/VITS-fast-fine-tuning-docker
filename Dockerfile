FROM ubuntu:jammy
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt update -y && apt-get -y upgrade --fix-missing
RUN apt install -y software-properties-common  lsb-release
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt install -y build-essential wget curl git screen ffmpeg cmake vim screen python3.11 python3.11-dev

RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt update -y && apt install -y libcudnn8 libcudnn8-dev

WORKDIR /workspace
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 -
RUN rm -rf /usr/bin/python && ln -s /usr/bin/python3.11 /usr/bin/python
RUN rm -rf /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3

# install python packages
RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3.11 -m pip install imageio moviepy Cython librosa scikit-learn scipy tensorboard unidecode pyopenjtalk jamo pypinyin jieba protobuf cn2an inflect eng_to_ipa ko_pron indic_transliteration num_thai opencc demucs openai-whisper gradio

RUN git clone https://github.com/Plachtaa/VITS-fast-fine-tuning.git --depth 1 /workspace/VITS-fast-fine-tuning
RUN mkdir -p /workspace/VITS-fast-fine-tuning/monotonic_align/monotonic_align

# build monotonic align
WORKDIR /workspace/VITS-fast-fine-tuning/monotonic_align
RUN python setup.py build_ext --inplace

# download auxiliary data for training
WORKDIR /workspace/VITS-fast-fine-tuning
RUN mkdir -p video_data raw_audio denoised_audio custom_character_voice segmented_character_voice configs pretrained_models
RUN wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
RUN apt install -y unzip
RUN unzip sampled_audio4ft_v2.zip

# download pretrained CJE model
RUN wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
RUN wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
RUN wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json

# add script
WORKDIR /app
ADD scripts/convert_long_audio_to_wave.py .





ENTRYPOINT ["tail", "-f", "/dev/null"]

