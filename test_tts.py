from melo.api import TTS
from pathlib import Path
import time

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

use_ov = True  ## Used to control whether to use torch or openvino
use_int8 = True
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH', device=device, use_int8=True)
speaker_ids = model.hps.data.spk2id


dur_time_list = []
loop_num = 10

if use_ov:
    ov_path = "/tts_ov"
    if not Path(ov_path).exists():
        model.tts_convert_to_ov(ov_path)
    model.ov_model_init(ov_path)

for i in range(loop_num):
    if not use_ov:
        output_path = 'zh.wav'
        start = time.perf_counter()
        model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed, use_ov = use_ov)
        end = time.perf_counter()
    else:
        output_path = 'zh_ov.wav'
        start = time.perf_counter()
        model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed, use_ov=use_ov, noise_scale=0.1, noise_scale_w=0.8)
        end = time.perf_counter()

    dur_time = (end - start) * 1000
    dur_time_list.append(dur_time)

if loop_num > 1:
    avg_lantecy = sum(dur_time_list[1:]) / (len(dur_time_list) - 1)
    print(f"MeloTTS model e2e avg latency: {avg_lantecy:.2f} ms")
