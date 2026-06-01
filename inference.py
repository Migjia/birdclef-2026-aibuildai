#!/usr/bin/env python3
"""
BirdCLEF+ 2026 V23: Standalone Inference Script
Self-contained inference for Kaggle submission (no external imports from training scripts)
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d

# Suppress warnings
warnings.filterwarnings('ignore')

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CPU thread limits
OMP_NUM_THREADS = 4
os.environ['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(OMP_NUM_THREADS)

# Audio parameters
SR = 32000
WINDOW_SEC = 5
N_WINDOWS = 12
WINDOW_SAMPLES = SR * WINDOW_SEC

# Mel-spectrogram parameters
N_MELS_SED = 256
N_FFT_SED = 2048
HOP_SED = 512
FMIN_SED = 20
FMAX_SED = 16000
TOP_DB_SED = 80


def file_to_chunks(path):
    """Load audio file and chunk into 5-second windows."""
    y, sr0 = sf.read(str(path), dtype='float32', always_2d=False)

    if y.ndim == 2:
        y = y.mean(axis=1)

    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)

    n = 60 * SR
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]

    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC

    return chunks, ends


def audio_to_mel(chunks):
    """Convert audio chunks to mel-spectrograms."""
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)

    return np.stack(mels)[:, None].astype(np.float32)


def make_onnx_session(path):
    """Create ONNX inference session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(path), sess_options=opts, providers=['CPUExecutionProvider']
    )
    return session


def stable_sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def run_sed_inference(sessions, mel_batch, target_cols):
    """Run SED inference."""
    fold_preds = []

    for session in sessions:
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        outputs = session.run(output_names, {input_name: mel_batch})

        clip_logits = outputs[0]
        frame_logits = outputs[1]

        clip_probs = stable_sigmoid(clip_logits)
        frame_max = frame_logits.max(axis=1)
        frame_probs = stable_sigmoid(frame_max)

        combined = 0.5 * clip_probs + 0.5 * frame_probs
        fold_preds.append(combined)

    preds = np.mean(fold_preds, axis=0)
    preds = np.clip(preds, 0.0, 1.0)

    return preds


def load_taxonomy_map(taxonomy_csv_path, target_cols):
    """Load Perch taxonomy mapping."""
    perch_df = pd.read_csv(taxonomy_csv_path)
    perch_classes = perch_df['ebird2021'].tolist()

    taxonomy_map = {}
    unmapped_count = 0

    for bc_class in target_cols:
        if bc_class in perch_classes:
            idx = perch_classes.index(bc_class)
            taxonomy_map[bc_class] = [idx]
        else:
            genus = bc_class.split('_')[0]
            genus_indices = [
                i for i, c in enumerate(perch_classes)
                if c.startswith(genus + '_')
            ]

            if not genus_indices:
                unmapped_count += 1
                genus_indices = [0]

            taxonomy_map[bc_class] = genus_indices

    return taxonomy_map


def map_perch_to_birdclef(perch_logits, taxonomy_map, target_cols):
    """Map Perch logits to BirdCLEF classes."""
    n_windows = perch_logits.shape[0]
    n_classes = len(target_cols)

    birdclef_logits = np.zeros((n_windows, n_classes), dtype=np.float32)

    for i, bc_class in enumerate(target_cols):
        perch_indices = taxonomy_map[bc_class]
        birdclef_logits[:, i] = perch_logits[:, perch_indices].max(axis=1)

    return birdclef_logits


def run_perch_inference(session, chunks, taxonomy_map, target_cols):
    """Run Perch inference."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    outputs = session.run(output_names, {input_name: chunks})
    perch_logits = outputs[3]

    birdclef_logits = map_perch_to_birdclef(perch_logits, taxonomy_map, target_cols)
    preds = stable_sigmoid(birdclef_logits)

    return preds


def compute_global_prior(data_dir, target_cols):
    """Compute global class prior from training data."""
    train_path = Path(data_dir) / 'train.csv'

    if not train_path.exists():
        return np.ones(len(target_cols), dtype=np.float32) / len(target_cols)

    df = pd.read_csv(train_path)
    class_counts = df['primary_label'].value_counts()

    global_prior = np.zeros(len(target_cols), dtype=np.float32)
    for i, cls in enumerate(target_cols):
        global_prior[i] = class_counts.get(cls, 0)

    total = global_prior.sum()
    if total > 0:
        global_prior = global_prior / total
    else:
        global_prior = np.ones(len(target_cols), dtype=np.float32) / len(target_cols)

    return global_prior


def apply_global_prior(preds, global_prior, lambda_prior=0.4):
    """Apply Bayesian prior in logit space."""
    preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
    logits = np.log(preds_clipped / (1 - preds_clipped))

    prior_clipped = np.clip(global_prior, 1e-7, 1 - 1e-7)
    prior_logits = np.log(prior_clipped)

    logits_with_prior = logits + lambda_prior * prior_logits[None, :]
    preds_with_prior = stable_sigmoid(logits_with_prior)

    return preds_with_prior


def apply_temporal_smoothing(preds, sigma=0.65):
    """Apply Gaussian temporal smoothing."""
    smoothed = gaussian_filter1d(preds, sigma=sigma, axis=0, mode='nearest')
    return smoothed


def linear_blend(perch_preds, sed_preds, w_perch=0.55, w_sed=0.45):
    """Linear probability blend."""
    blended = (w_perch * perch_preds) + (w_sed * sed_preds)
    blended = np.clip(blended, 0.0, 1.0)
    return blended


def get_target_columns(data_dir):
    """Extract target columns from sample_submission.csv."""
    sample_sub_path = Path(data_dir) / 'sample_submission.csv'
    df = pd.read_csv(sample_sub_path)

    target_cols = [c for c in df.columns if c != 'row_id']
    assert len(target_cols) == 234, f"Expected 234 target columns, got {len(target_cols)}"

    return target_cols


def main():
    import argparse
    parser = argparse.ArgumentParser(description='BirdCLEF+ 2026 V23 Inference')
    parser.add_argument('--input', type=str, required=True, help='Data directory path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    args = parser.parse_args()

    print("="*80)
    print("BirdCLEF+ 2026 V23: Inference")
    print("="*80)

    # Configuration (load from same directory as this script)
    config = {
        'sed_model_dir': '/home/mingjiacai/aibuildai-linux-x86_64-v0.1.1/high_lb_notebooks/BC2026-Distilled-SED-Public',
        'perch_model_path': '/home/mingjiacai/aibuildai-linux-x86_64-v0.1.1/high_lb_notebooks/perch_v2_no_dft.onnx/perch_v2_no_dft.onnx',
        'taxonomy_csv_path': '/home/mingjiacai/aibuildai-linux-x86_64-v0.1.1/high_lb_notebooks/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1/assets/perch_v2_ebird_classes.csv',
        'fast_mode': True,
        'w_perch': 0.55,
        'w_sed': 0.45,
        'temporal_sigma': 0.65,
        'lambda_prior': 0.4
    }

    # Get target columns
    target_cols = get_target_columns(args.input)
    print(f"Target classes: {len(target_cols)}")

    # Load models
    print("\nLoading models...")

    sed_model_dir = Path(config['sed_model_dir'])
    if config['fast_mode']:
        sed_model_paths = sorted(sed_model_dir.glob('sed_fold*.onnx'))[:2]
        print(f"FAST_MODE: Using {len(sed_model_paths)} SED folds")
    else:
        sed_model_paths = sorted(sed_model_dir.glob('sed_fold*.onnx'))
        print(f"Using all {len(sed_model_paths)} SED folds")

    sed_sessions = [make_onnx_session(p) for p in sed_model_paths]

    perch_session = make_onnx_session(config['perch_model_path'])
    taxonomy_map = load_taxonomy_map(config['taxonomy_csv_path'], target_cols)
    global_prior = compute_global_prior(args.input, target_cols)

    print("Models loaded successfully")

    # Process test soundscapes
    test_dir = Path(args.input) / 'test_soundscapes'
    audio_files = sorted(test_dir.glob('*.ogg'))

    if not audio_files:
        print(f"WARNING: No test files found in {test_dir}")
        print("Creating fallback submission with uniform priors")

        sample_sub_path = Path(args.input) / 'sample_submission.csv'
        sample_df = pd.read_csv(sample_sub_path)
        uniform_prior = 1.0 / len(target_cols)

        rows = []
        for row_id in sample_df['row_id']:
            row = {'row_id': row_id}
            row.update({col: uniform_prior for col in target_cols})
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Fallback submission saved to: {args.output}")
        return

    print(f"\nProcessing {len(audio_files)} test files...")

    all_row_ids = []
    all_perch_preds = []
    all_sed_preds = []

    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing {i+1}/{len(audio_files)}: {audio_path.name}")

        chunks, ends = file_to_chunks(audio_path)

        # SED inference
        mel_batch = audio_to_mel(chunks)
        sed_preds = run_sed_inference(sed_sessions, mel_batch, target_cols)

        # Perch inference
        perch_preds = run_perch_inference(perch_session, chunks, taxonomy_map, target_cols)

        # Post-processing
        perch_preds = apply_global_prior(perch_preds, global_prior, lambda_prior=config['lambda_prior'])
        perch_preds = apply_temporal_smoothing(perch_preds, sigma=config['temporal_sigma'])
        sed_preds = apply_temporal_smoothing(sed_preds, sigma=config['temporal_sigma'])

        # Store predictions
        filename_stem = audio_path.stem
        for j, end_sec in enumerate(ends):
            row_id = f"{filename_stem}_{int(end_sec)}"
            all_row_ids.append(row_id)
            all_perch_preds.append(perch_preds[j])
            all_sed_preds.append(sed_preds[j])

    # Blend
    perch_preds_all = np.array(all_perch_preds)
    sed_preds_all = np.array(all_sed_preds)
    blended_preds = linear_blend(perch_preds_all, sed_preds_all,
                                  w_perch=config['w_perch'], w_sed=config['w_sed'])

    # Create DataFrame
    df = pd.DataFrame(blended_preds, columns=target_cols)
    df.insert(0, 'row_id', all_row_ids)

    # Validate
    assert 'row_id' in df.columns
    assert len(df.columns) == 235
    assert list(df.columns[1:]) == target_cols

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("="*80)


if __name__ == '__main__':
    main()
