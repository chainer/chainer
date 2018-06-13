# WaveNet
A Chainer implementation of mel-spectrogram vocoder using WaveNet. 

## Usage
1. Install requirements.
    - `pip3 install -r requirements.txt`
2. Download dataset.
    - `wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz`
    - `tar -xf VCTK-Corpus.tar.gz`
3. Start training.
    - `python3 train.py -g <gpu id> --dataset <directory of dataset e.g. ./VCTK-Corpus/>`
    - You can change other parameters. Please see args.
4. Generate audio with trained model.
    - `python3 generate.py -i <input file> -m <trained model e.g. snapshot_iter_500000>`

## Details
- Default parameters of WaveNet are same as [nv-wavenet](https://github.com/NVIDIA/nv-wavenet/).
- Mel-spectrograms are calculated with [librosa](https://github.com/librosa/librosa).
- If you want to get more audible results, use MoL Wavenet with exponential moving average.
