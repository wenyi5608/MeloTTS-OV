
import argparse
from pathlib import Path

from melo.api import TTS
from pathlib import Path
import time
import openvino as ov
import nncf
import string
from torch.utils.data import Dataset, DataLoader
from melo import utils
import torch
import re
from transformers import AutoTokenizer
import random
import numpy as np
import librosa
import soundfile
from melo.utils import load_wav_to_torch_librosa as load_wav_to_torch
from melo.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from torch.nn import functional as F

#constant parameter
device = 'cpu' 
lang = "ZH"# ZH for Chinese
sampling_rate=44100
hp_sampling_rate = 44100
filter_length = 2048
n_mel_channels = 128
hop_length = 512
win_length = 2048
mel_fmin = 0.0
mel_fmax = None
segment_size = 16384

#create pytorch model
pth_model = TTS(language=lang,  use_hf= False ,use_int8= False)

def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

def cal_loss_mel(audio1, audio2):
    #print("===cal_loss_mel===")
    pth_audio_norm, sampling_rate = load_wav_to_torch(audio1, hp_sampling_rate)
    pth_audio_norm = pth_audio_norm.unsqueeze(0)

    pth_spec = mel_spectrogram_torch(
        pth_audio_norm,
        filter_length,
        n_mel_channels,
        hp_sampling_rate,
        hop_length,
        win_length,
        mel_fmin,
        mel_fmax,
        center=False,
        )

    ov_audio_norm, sampling_rate = load_wav_to_torch(audio2, hp_sampling_rate)
    ov_audio_norm = ov_audio_norm.unsqueeze(0)

    ov_spec = mel_spectrogram_torch(
        ov_audio_norm,
        filter_length,
        n_mel_channels,
        hp_sampling_rate,
        hop_length,
        win_length,
        mel_fmin,
        mel_fmax,
        center=False,
        )
    
    # 获取第三个维度的最小值
    min_dim = min(pth_spec.size(2), ov_spec.size(2))

    # 裁剪两个 tensor 的第三个维度
    pth_spec = pth_spec[:, :, :min_dim]
    ov_spec = ov_spec[:, :, :min_dim]

    loss_mel = F.l1_loss(pth_spec, ov_spec)
    
    del pth_audio_norm, ov_audio_norm, pth_spec, ov_spec
    
    return loss_mel


def read_file_to_list(filename, data_count = 20):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # remove the newline character at the end of line
    return [line.strip() for line in lines][:data_count]

def transform_fn(data_item):
    #print("transform_fn    ", data_item[0])
    data_item = data_item[0]
    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(data_item, lang, pth_model.hps, device, symbol_to_id=pth_model.symbol_to_id, bert_model=pth_model.bert_model, use_ov=False)
    
    x_tst = phones.to(device).unsqueeze(0)
    tones = tones.to(device).unsqueeze(0)
    lang_ids = lang_ids.to(device).unsqueeze(0)
    bert = bert.to(device).unsqueeze(0)
    ja_bert = ja_bert.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)

    speaker_ids = pth_model.hps.data.spk2id
    speaker_id = speaker_ids['ZH'] # ZH_MIX_EN
    speakers = torch.LongTensor([speaker_id]).to(device)

    sdp_ratio=0.2
    noise_scale=0.6
    noise_scale_w=0.8
    speed=1.0
    
    inputs_dict = {}
    inputs_dict['phones'] = x_tst
    inputs_dict['phones_length'] = x_tst_lengths
    inputs_dict['speakers'] = speakers
    inputs_dict['tones'] = tones
    inputs_dict['lang_ids'] = lang_ids
    inputs_dict['bert'] = bert
    inputs_dict['ja_bert'] = ja_bert
    inputs_dict['noise_scale'] = torch.tensor([noise_scale])
    inputs_dict['length_scale'] = torch.tensor([1. / speed])
    inputs_dict['noise_scale_w'] = torch.tensor([noise_scale_w])
    inputs_dict['sdp_ratio'] = torch.tensor([sdp_ratio])
    return inputs_dict

def validation_fn(ov_int8_model:ov.CompiledModel, dataset:torch.utils.data.DataLoader):
    correct_count = 0
    total_count = 0
    i = 0
    for sample in dataset:
        data_item = sample[0]

        clean_audio_path = Path(f"pth_audio.wav")
        noisy_audio_path =  Path(f"ov_audio.wav")
        i += 1

        #{'EN-US': 0, 'EN-BR': 1, 'EN_INDIA': 2, 'EN-AU': 3, 'EN-Default': 4}
        sdp_ratio=0.2
        noise_scale=0.6
        noise_scale_w=0.8
        speed=1.0

        pth_model.tts_to_file(data_item, 1, clean_audio_path, speed=speed, use_ov = False)

        #mode inference and save audio to file
        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(data_item, lang, pth_model.hps, device, symbol_to_id=pth_model.symbol_to_id, bert_model=pth_model.bert_model, use_ov=False)
    
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)

        speaker_ids = pth_model.hps.data.spk2id
        speaker_id = speaker_ids['ZH'] # ZH_MIX_EN
        speakers = torch.LongTensor([speaker_id]).to(device)

        ov_audio_list=[]
        ov_inputs = {}
        ov_inputs['phones'] = x_tst
        ov_inputs['phones_length'] = x_tst_lengths
        ov_inputs['speakers'] = speakers
        ov_inputs['tones'] = tones
        ov_inputs['lang_ids'] = lang_ids
        ov_inputs['bert'] = bert
        ov_inputs['ja_bert'] = ja_bert
        ov_inputs['noise_scale'] = torch.tensor([noise_scale])
        ov_inputs['length_scale'] = torch.tensor([1. / speed])
        ov_inputs['noise_scale_w'] = torch.tensor([noise_scale_w])
        ov_inputs['sdp_ratio'] = torch.tensor([sdp_ratio])
        
        ov_output=ov_int8_model(ov_inputs)
        ov_audio=ov_output[0][0]

        ov_audio_list.append(ov_audio)
        ov_audio = audio_numpy_concat(ov_audio_list, sampling_rate, speed=speed)
        soundfile.write(noisy_audio_path, ov_audio, sampling_rate)

        diff = cal_loss_mel(clean_audio_path, noisy_audio_path)
        
        #print(f"===mel_loss==={diff}")
        if diff < 1.2:
            correct_count+=1

        total_count += 1

    # 计算正确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MeloTTS quantization nncf accuracy control", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="MeloTTS OpenVINO IR .xml")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")
    parser.add_argument("-num", "--data_num", default=30, help="The number of samples used for model quantization")
    
    args = parser.parse_args()
    ov_model_path = args.model_id

    data_count = int(args.data_num)

    print("data_count ", data_count)

    encoder_calibration_data = read_file_to_list('newq.txt', data_count=data_count)

    ov_model = ov.Core().read_model(ov_model_path)

    val_data_loader = DataLoader(encoder_calibration_data, batch_size=1, shuffle=False)
    calibration_dataset = nncf.Dataset(val_data_loader, transform_fn)
    quantized_model = nncf.quantize_with_accuracy_control(
        model=ov_model,
        calibration_dataset = calibration_dataset,
        validation_dataset =  calibration_dataset,
        validation_fn = validation_fn ,
        model_type=nncf.ModelType.TRANSFORMER,
        max_drop=0.2,
        drop_type=nncf.DropType.ABSOLUTE,
        fast_bias_correction = True,
    )

    ov.save_model(quantized_model, Path(f"{args.output_dir}/tts_int8_{lang}.xml"))

