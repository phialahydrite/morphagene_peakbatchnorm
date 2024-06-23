#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    morphagene_batchnorm.py -w <inputwavfiles> -a <amplitude>
    
Use pydub and numpy to match peak amplitude to a target value, while maintaining preexisting splices.
Uses wavefile.py by X-Raym [https://github.com/X-Raym/wavfile.py/blob/master/wavfile.py]
"""

import pydub
import numpy as np
import os, sys, getopt, glob
from wavfile import read, write

def pydub_to_np(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate

def match_target_amplitude(audio, target_maxdB):
    '''
    Match sound's max amplitude to a target value
    '''
    change_in_dBFS = target_maxdB - audio.max_dBFS 
    return audio.apply_gain(change_in_dBFS)


def main(argv):
    inputwavefiles = ''
    try:
        opts, args = getopt.getopt(argv,"hw:a:",["wavfiles=","amplitude="])
    except getopt.GetoptError:
        print('Error in usage, correct format:\n'+\
            'morphagene_batchnorm.py -w <inputwavfiles> -a <amplitude>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Morphagene reel batch peak amplitude normalization using pydub and numpy, maintaining splices:\n'+\
                  'morphagene_peakbatchnorm.py -w <inputwavfiles> -a <amplitude>\n'+\
                  '"-a" is the desired peak amplitude in dBs.\n')
            sys.exit()
        elif opt in ("-w", "--wavfiles"):
            inputwavefiles = arg
        elif opt in ("-a", "--amplitude"):
            amplitude = float(arg)

    print(f'Input wave files: {inputwavefiles}')
    print(f'Output normalized peak amplitude: {amplitude}dBs')

    files = glob.glob(inputwavefiles + '*.wav')
    for fi in files:
        # read markers list from input Morphagene reels using wavfile.py, ignoring everything else
        _, _, _, markers_list, _, _ = read(fi,readmarkerslist=True)

        # use pydub for normalization and peak amplitude adjustment
        sound = pydub.AudioSegment.from_file(fi)
        print(f'Amplified {os.path.basename(fi)} by {amplitude - sound.max_dBFS}dBs')
        normalized_sound = match_target_amplitude(sound, amplitude)
        normalized_sound_array,sample_rate = pydub_to_np(normalized_sound)

        # write peak amplitude normalized wav file with labels using wavfile.py
        print(f'Normalizing file {fi} in {inputwavefiles}')
        write(inputwavefiles + f'norm_{os.path.basename(fi)}',sample_rate,
                            normalized_sound_array.astype('float32'),
                            normalized=True,
                            markers=markers_list)

if __name__ == "__main__":
    main(sys.argv[1:])
