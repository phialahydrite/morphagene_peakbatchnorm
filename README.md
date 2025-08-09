# morphagene_peakbatchnorm
Morphagene reel batch peak amplitude normalization using pydub and numpy, maintaining splices.

Requires pydub and wavfile.py (modified from X-Raym, https://github.com/X-Raym/wavfile.py/blob/master/wavfile.py).

# Example
Modify volume of reels in folder such that all resulting reels have a peak amplitude of -12dB
```
python morphagene_peakbatchnorm.py -w "path/to/morphagene/reels/' -a -12
```
