"""
Compute Equal Error Rate (EER) from a CM score file and an ASVspoof
key/protocol file containing bonafide/spoof labels.

This re-implements the standard EER computation used in the official
ASVspoof eval-package (asvspoof-challenge/2021/eval-package/eval_metrics.py).

Usage:
    python compute_eer.py --score_file ./score_LA.txt --key_file ./keys/LA/CM/trial_metadata.txt

score_file format (one line per trial, whitespace separated):
    <utt_id> <score>
    (this is exactly what main.py's produce_evaluation_file writes)

key_file format: any ASVspoof-style protocol/key file where each line
contains the utterance id as one whitespace-separated token, and the
label as another token that is literally "bonafide" or "spoof"
(case-insensitive). This covers both the ASVspoof2019 protocol format
and the ASVspoof2021 trial_metadata.txt format.
"""
import argparse
import numpy as np


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def load_scores(score_file):
    scores = {}
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id, score = parts[0], parts[1]
            scores[utt_id] = float(score)
    return scores


def load_labels(key_file, phase=None):
    labels = {}
    with open(key_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if phase is not None and phase not in tokens:
                continue
            label = None
            for tok in tokens:
                if tok.lower() in ('bonafide', 'spoof'):
                    label = tok.lower()
                    break
            if label is None:
                continue
            # utt id: assume it's the token right before the label search
            # fails for weird formats, so instead find whichever token
            # matches a typical utt-id-looking pattern is unreliable;
            # standard ASVspoof formats put utt_id as the 2nd token.
            utt_id = tokens[1] if len(tokens) > 1 else tokens[0]
            labels[utt_id] = label
    return labels


def main():
    parser = argparse.ArgumentParser(description='Compute EER from score file + key/protocol file')
    parser.add_argument('--score_file', type=str, required=True, help='path to score file (utt_id score per line)')
    parser.add_argument('--key_file', type=str, required=True, help='path to key/protocol file with bonafide/spoof labels')
    parser.add_argument('--phase', type=str, default=None, choices=[None, 'progress', 'eval', 'hidden_track'],
                         help='if the key file mixes phases (ASVspoof2021 trial_metadata.txt), filter to just this phase')
    args = parser.parse_args()

    scores = load_scores(args.score_file)
    labels = load_labels(args.key_file, phase=args.phase)

    bona_scores = []
    spoof_scores = []
    missing = 0

    for utt_id, label in labels.items():
        if utt_id not in scores:
            missing += 1
            continue
        if label == 'bonafide':
            bona_scores.append(scores[utt_id])
        else:
            spoof_scores.append(scores[utt_id])

    bona_scores = np.array(bona_scores)
    spoof_scores = np.array(spoof_scores)

    print(f'Bonafide trials found: {bona_scores.size}')
    print(f'Spoof trials found:    {spoof_scores.size}')
    if missing:
        print(f'Warning: {missing} utterances in key file had no matching score in score file')

    if bona_scores.size == 0 or spoof_scores.size == 0:
        print('ERROR: need at least one bonafide and one spoof trial with scores to compute EER.')
        return

    eer, threshold = compute_eer(bona_scores, spoof_scores)
    print(f'\nEER = {eer * 100:.2f}%  (threshold = {threshold:.4f})')


if __name__ == '__main__':
    main()