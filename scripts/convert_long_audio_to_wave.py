import os
import glob
import sys
import tqdm
from moviepy.editor import AudioFileClip as a_clip

def is_long_audio(input_audio_file):
    try:
        a_clip(input_audio_file)
    except:
        return False
    return True

if len(sys.argv) < 3:
    print("Usage: python convert_long_audio_to_wave.py <long_audio_input_directory> <wav_output_directory>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

input_audio_files = tqdm.tqdm(glob.glob(os.path.join(input_dir, '*.*')))
for input_audio_file in input_audio_files:
    if not is_long_audio(input_audio_file):
        continue
    output_audio_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_audio_file))[0] + '.wav')
    os.system(f'ffmpeg -i "{input_audio_file}" -acodec pcm_s16le -ac 1 -ar 16000 "{output_audio_file}"')

