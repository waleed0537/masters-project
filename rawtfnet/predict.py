"""
predict.py
Minimal, clean inference path for RawTFNet_small.
Mirrors the exact loading/model-construction logic from test_audio.py,
but exposes it as a single reusable function: run_inference(audio_path) -> score

Run from the root of RawTFNet-Pytorch (same folder as main.py):
    python predict.py --model_path ./ckpts/Best_RawTFNet_16.pth --audio LA_E_1000048.flac
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


def load_model(model_path, sample_rate=16000, max_len=64600, device='cpu'):
    """
    Builds the same args object shape that test_audio.py successfully uses
    to construct RawTFNet_small, then loads the trained weights.
    """
    args = argparse.Namespace(
        model_name='rawtfnet_small',
        model_path=model_path,
        sample_rate=sample_rate,
        max_len=max_len,
    )

    from model_scripts.rawtfnet import RawTFNet_small as Model

    model = Model(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_inference(audio_path, model, sample_rate=16000, max_len=64600, device='cpu'):
    """
    Single entry point: audio file path in -> bonafide score out.
    This is the function you keep stable — it's what gets ported/exported later.
    """
    x = load_audio(audio_path, sample_rate=sample_rate, max_len=max_len).to(device)
    with torch.no_grad():
        out = model(x)
        out = out.view(x.shape[0], -1)   # normalize shape, same as test_audio.py
        score = out[:, 1].item()
    return score


def main():
    parser = argparse.ArgumentParser(description='Minimal RawTFNet_small inference')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--max_len', type=int, default=64600)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    model = load_model(args.model_path, sample_rate=args.sample_rate,
                        max_len=args.max_len, device=device)
    print('Model loaded:', args.model_path)

    score = run_inference(args.audio, model, sample_rate=args.sample_rate,
                           max_len=args.max_len, device=device)
    verdict = 'BONAFIDE (likely real)' if score > 0 else 'SPOOF (likely fake)'
    print(f'{args.audio}\tscore={score:.4f}\t{verdict}')


if __name__ == '__main__':
    main()