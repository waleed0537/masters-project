"""
Quick standalone tester for RawTFNet / RawTFNet_small.
Run this from the root of RawTFNet-Pytorch (same folder as main.py).

Usage:
    python test_audio.py --model_name rawtfnet_small --model_path ./ckpts/Best_RawTFNet_16.pth file1.wav file2.flac ...

It does NOT use the ASVspoof dataset/protocol pipeline at all -- just loads
whatever audio file(s) you point it at directly.
"""
import argparse
import numpy as np
import librosa
import torch


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def load_audio(path, sample_rate=16000, max_len=64600):
    X, _ = librosa.load(path, sr=sample_rate)
    X = pad(X, max_len)
    x_inp = torch.from_numpy(X).float().unsqueeze(0)  # shape (1, max_len)
    return x_inp


def main():
    parser = argparse.ArgumentParser(description='Quick single/multi-file RawTFNet tester')
    parser.add_argument('--model_name', type=str, required=True, choices=['rawtfnet', 'rawtfnet_small'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--max_len', type=int, default=64600,
                         help='number of raw audio samples fed to the model (default ~4s at 16kHz)')
    parser.add_argument('audio_files', nargs='+', help='one or more .wav/.flac files to test')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    if args.model_name == 'rawtfnet':
        from model_scripts.rawtfnet import RawTFNet as Model
    else:
        from model_scripts.rawtfnet import RawTFNet_small as Model

    model = Model(args).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print('Model loaded:', args.model_path)

    with torch.no_grad():
        for path in args.audio_files:
            try:
                x = load_audio(path, sample_rate=args.sample_rate, max_len=args.max_len).to(device)
                out = model(x)
                # With batch_size=1 some models squeeze away the batch dim,
                # so normalize the output to shape (batch, num_classes) first.
                out = out.view(x.shape[0], -1)
                # index 1 = "bonafide" score convention used in produce_evaluation_file in main.py
                score = out[:, 1].item()
                verdict = 'BONAFIDE (likely real)' if score > 0 else 'SPOOF (likely fake)'
                print(f'{path}\tscore={score:.4f}\t{verdict}')
            except Exception as e:
                print(f'{path}\tERROR: {e}')


if __name__ == '__main__':
    main()