#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: export2onnx.py
# Description: Convert Generator to ONNX format
# Author: Feng Wang
# Date: 2024-03-02

import sys
import os
from typing import Any, Dict
import logging
import onnx
import torch
import utils
from models import SynthesizerTrn


class OnnxModel(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        x_lengths,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        sid=0,
        max_len=None,
    ):
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            max_len=max_len,
        )[0]


device = "cpu"


@torch.no_grad()
def main():

    model_path: str = '/output/model/G_latest.pth'
    config_path:str = '/output/model/config.json'
    onnx_path:str = '/output/model/G_latest.onnx'
    pt_path:str = '/output/model/G_latest_optimized.pt'
    opset_version:int = 13

    if not os.path.exists(model_path):
        logging.error( f"model path {model_path} doesn't exist'" )
        assert False, f"model path {model_path} does not exist"

    if not os.path.exists(config_path):
        logging.error( f"model config {config_path} doesn't exist'" )
        assert False, f"model config {config_path} does not exist"

    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    logging.info( f'Generator checkpoint loaded from {model_path}' )

    x = torch.randint(low=1, high=50, size=(50,), dtype=torch.int64)
    x = x.unsqueeze(0)

    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    length_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_w = torch.tensor([1], dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.int64)

    model = OnnxModel(net_g)

    torch.onnx.export(
        model,
        (x, x_length, noise_scale, length_scale, noise_scale_w, sid),
        onnx_path,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_length",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
            "sid",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},  # n_audio is also known as batch_size
            "x_length": {0: "N"},
            "y": {0: "N", 2: "L"},
        },
    )
    logging.info( f'Model exported to {onnx_path}' )


if __name__ == '__main__':
    logging.basicConfig( format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='/output/export2onnx.log', encoding='utf-8', level=logging.DEBUG )
    main()

