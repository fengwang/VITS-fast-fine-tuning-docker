# usage: python convert_long_audio_to_wave.py
import os
import glob
import sys
import tqdm
from moviepy.editor import AudioFileClip as a_clip

import logging
logging.basicConfig( format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='/output/convert_long_audio_to_wave.log', encoding='utf-8', level=logging.DEBUG )

def is_long_audio(input_audio_file):
    try:
        a_clip(input_audio_file)
    except:
        return False
    return True

'''
if len(sys.argv) < 3:
    print("Usage: python convert_long_audio_to_wave.py <long_audio_input_directory> <wav_output_directory>")
    sys.exit(1)
'''

input_dir = '/input'
output_dir = '/output/denoised_audio'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info( 'Created output folder {output_dir}' )

if len(sys.argv) == 3:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

logging.info( f"input_dir: {input_dir}" )
logging.info( f"output_dir: {output_dir}" )


input_audio_files = glob.glob(os.path.join(input_dir, '*.*'))
#input_audio_files = tqdm.tqdm(glob.glob(os.path.join(input_dir, '*.*')))
# TODO: parallel here
for input_audio_file in tqdm.tqdm(input_audio_files):
    if not is_long_audio(input_audio_file):
        logging.info(f"{input_audio_file} is not long audio. skipping...")
        continue
    logging.info( f"processing {input_audio_file}" )
    output_audio_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_audio_file))[0] + '.wav')
    os.system(f'ffmpeg -i "{input_audio_file}" -acodec pcm_s16le -ac 1 -ar 16000 "{output_audio_file}"')
    logging.info( f"processed {input_audio_file}" )

