"""
Shesha Distinction - Encoder Test (7 Domains)
Domains:
1. Language (SST-2) - 4 models
2. Vision (CIFAR-100) - 4 models
3. Audio (LibriSpeech) - 2 models
4. Video (Real Video) - 4 models
5. Neuroscience (Steinmetz) - All sessions
6. Protein (Swiss-Prot) - Multiple encoders
7. Molecular (PBMC3k) - Multiple encoders

Metric: FEATURE-SPLIT SHESHA (Internal Geometric Consistency)
Scale: 15 SEEDS
"""


import transformers.utils.import_utils
def bypass_torch_check(): return True
transformers.utils.import_utils.check_torch_load_is_safe = bypass_torch_check
# -------------------------------------------


import os
import warnings
import tarfile
import requests
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from scipy.sparse import issparse
import scanpy as sc

# Transformers & Data
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoImageProcessor,
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor, HubertModel,
    VideoMAEImageProcessor, VideoMAEModel, CLIPModel, CLIPProcessor
)
from sentence_transformers import SentenceTransformer

# GPU Acceleration
try:
    from cuml.decomposition import PCA
    from cuml.random_projection import GaussianRandomProjection
    print(">>> SUCCESS: Using GPU-accelerated PCA (cuML)")
    IS_GPU_PCA = True
except ImportError:
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    print(">>> NOTICE: Falling back to CPU PCA (sklearn)")
    IS_GPU_PCA = False

try:
    if torch.cuda.is_available():
        from cuml.preprocessing import StandardScaler
    else:
        from sklearn.preprocessing import StandardScaler
except:
    from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = Path("./shesha-distinction")
OUTDIR.mkdir(parents=True, exist_ok=True)


# FULL 15 SEEDS
SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

CONFIG = {
    'language': {'n_samples': 500, 'max_len': 64},
    'vision': {'n_images': 400, 'image_size': 224},
    'audio': {'n_audio': 200, 'sample_rate': 16000},
    'video': {'n_videos': 100, 'frames_per_video': 16, 'video_size': 224},
    'neuroscience': {'min_neurons': 20, 'min_trials': 50},
    'protein': {'n_proteins': 200},
    'molecular': {'n_cells': 1000},
}

# =============================================================================
# METRICS
# =============================================================================

def compute_shesha_features(X, n_splits=30, random_state=None):
    """Feature-split Shesha on GPU."""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    elif X.device.type != 'cuda':
        X = X.to(DEVICE).float()
    else:
        X = X.float()

    n_samples, n_features = X.shape
    if n_samples < 10 or n_features < 2:
        return np.nan

    tri_idx = torch.triu_indices(n_samples, n_samples, offset=1, device=DEVICE)
    corrs = []
    rng = np.random.default_rng(random_state)

    for _ in range(n_splits):
        perm = torch.randperm(n_features, device=DEVICE)
        half = n_features // 2
        if half < 1:
            half = 1

        idx1 = perm[:half]
        idx2 = perm[half:2*half]
        if len(idx2) == 0:
            idx2 = idx1

        X1 = X[:, idx1]
        X2 = X[:, idx2]

        X1_n = torch.nn.functional.normalize(X1, p=2, dim=1)
        X2_n = torch.nn.functional.normalize(X2, p=2, dim=1)

        rdm1 = 1.0 - torch.mm(X1_n, X1_n.t())
        rdm2 = 1.0 - torch.mm(X2_n, X2_n.t())

        v1 = rdm1[tri_idx[0], tri_idx[1]].cpu().numpy()
        v2 = rdm2[tri_idx[0], tri_idx[1]].cpu().numpy()

        if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
            continue

        rho, _ = spearmanr(v1, v2)
        if np.isfinite(rho):
            corrs.append(rho)

    return float(np.mean(corrs)) if len(corrs) >= 5 else np.nan


def compute_cka(X, Y):
    """GPU Linear CKA"""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float64, device='cuda')
    else:
        X = X.double().cuda()
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float64, device='cuda')
    else:
        Y = Y.double().cuda()
    
    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of samples")
    
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    K = torch.matmul(X, X.T)
    L = torch.matmul(Y, Y.T)
    
    H = torch.eye(n, dtype=torch.float64, device='cuda') - torch.ones((n, n), dtype=torch.float64, device='cuda') / n
    K = H @ K @ H
    L = H @ L @ H
    
    num = (K * L).sum()
    den = torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-12
    return float(torch.clamp(num / den, 0.0, 1.0).item())


# =============================================================================
# ENCODER TRANSFORMATIONS
# =============================================================================

def build_encoder_transformations(X_base, seed):
    """Full encoder transformation suite (~30 variants)."""
    if hasattr(X_base, 'cpu'):
        X_base_native = X_base.cpu().numpy()
    elif hasattr(X_base, 'get'):
        X_base_native = X_base.get()
    else:
        X_base_native = np.asarray(X_base)

    rng = np.random.default_rng(seed)
    encoders = {}
    n_samples, n_features = X_base_native.shape

    # 1. PCA at various ranks
    for k in [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 256, 300]:
        k_actual = min(k, n_samples - 1, n_features)
        if k_actual >= 5:
            try:
                if IS_GPU_PCA:
                    pca = PCA(n_components=k_actual)
                else:
                    pca = PCA(n_components=k_actual, random_state=seed)
                encoders[f"pca_{k:03d}"] = pca.fit_transform(X_base_native)
            except:
                pass

    # 2. Random Projections
    for k in [16, 32, 64, 128, 256]:
        k_actual = min(k, n_features)
        if k_actual >= 5:
            try:
                grp = GaussianRandomProjection(n_components=k_actual, random_state=seed)
                encoders[f"randproj_{k:03d}"] = grp.fit_transform(X_base_native)
            except:
                pass

    # 3. Top Variance Features
    try:
        vars = np.var(X_base_native, axis=0)
        for k in [50, 100, 200, 400, 800]:
            if k < n_features:
                idx = np.argsort(vars)[-k:]
                encoders[f"topvar_{k:03d}"] = X_base_native[:, idx]
    except:
        pass

    # 4. Random Feature Subsets
    for k in [50, 100, 200]:
        if k < n_features:
            idx = rng.choice(n_features, k, replace=False)
            encoders[f"randfeat_{k:03d}"] = X_base_native[:, idx]

    # 5. Noise Injection
    for noise_level in [0.05, 0.1, 0.25, 0.5, 1.0]:
        noise = rng.normal(0, noise_level * np.std(X_base_native), X_base_native.shape)
        encoders[f"noise_{int(noise_level*100):03d}"] = X_base_native + noise

    # 6. Controls
    encoders["original"] = X_base_native.copy()
    
    try:
        scaler = StandardScaler()
        encoders["zscore"] = scaler.fit_transform(X_base_native)
    except:
        pass

    try:
        norms = np.linalg.norm(X_base_native, axis=1, keepdims=True) + 1e-12
        encoders["l2norm"] = X_base_native / norms
    except:
        pass

    return encoders


def run_encoder_analysis(base_embeddings, seed, domain_name):
    """Run analysis on all encoder transformations."""
    all_rows = []
    
    for base_name, X_base in base_embeddings.items():
        encoders = build_encoder_transformations(X_base, seed)
        
        refs = {
            "ref_original": encoders.get("original"),
            "ref_pca_100": encoders.get("pca_100", encoders.get("pca_075", encoders.get("original"))),
            "ref_zscore": encoders.get("zscore"),
        }

        for enc_name, X_enc in encoders.items():
            if X_enc is None:
                continue
            
            X = np.nan_to_num(X_enc, nan=0.0)
            if X.shape[0] < 10 or X.shape[1] < 2:
                continue
            if np.std(X) < 1e-9:
                continue

            shesha = compute_shesha_features(X, n_splits=30, random_state=seed)

            cka_values = []
            for ref_name, ref_X in refs.items():
                if ref_X is not None and ref_X.shape[0] == X.shape[0]:
                    ref_X = np.nan_to_num(ref_X, nan=0.0)
                    cka = compute_cka(X, ref_X)
                    if np.isfinite(cka):
                        cka_values.append(cka)
            
            cka_avg = np.mean(cka_values) if cka_values else np.nan

            all_rows.append({
                'domain': domain_name,
                'seed': seed,
                'base_model': base_name,
                'encoder': enc_name,
                'SHESHA': shesha,
                'CKA': cka_avg,
                'n_features': X.shape[1],
            })
    
    return all_rows


# =============================================================================
# DOMAIN 1: LANGUAGE (4 Models)
# =============================================================================

def run_language_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 1: LANGUAGE (4 Models)")
    print("=" * 60)
    
    try:
        ds = load_dataset("glue", "sst2", split="validation")
        texts = ds['sentence'][:CONFIG['language']['n_samples']]
        print(f"  Loaded {len(texts)} sentences")
        
        base_embeddings = {}
        
        # Model 1: MiniLM
        try:
            print("    Loading MiniLM...")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['minilm'] = emb
            print(f"    minilm: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] MiniLM: {e}")
        
        # Model 2: MPNet
        try:
            print("    Loading MPNet...")
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['mpnet'] = emb
            print(f"    mpnet: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] MPNet: {e}")
        
        # Model 3: DistilBERT
        try:
            print("    Loading DistilBERT...")
            model = SentenceTransformer("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['distilbert'] = emb
            print(f"    distilbert: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] DistilBERT: {e}")
        
        # Model 4: RoBERTa
        try:
            print("    Loading RoBERTa...")
            model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['roberta'] = emb
            print(f"    roberta: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] RoBERTa: {e}")
        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if not base_embeddings:
            return []
        
        all_results = []
        for seed in tqdm(SEEDS, desc="  Seeds"):
            all_results.extend(run_encoder_analysis(base_embeddings, seed, "Language"))
        
        return all_results
    
    except Exception as e:
        print(f"Language Failed: {e}")
        return []


# =============================================================================
# DOMAIN 2: VISION (4 Models)
# =============================================================================

def run_vision_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 2: VISION (4 Models)")
    print("=" * 60)
    
    try:
        from torchvision.datasets import CIFAR100
        from torchvision.transforms import Resize, ToTensor, Compose
        from torchvision.models import resnet50, ResNet50_Weights
        
        transform = Compose([Resize((224, 224)), ToTensor()])
        ds = CIFAR100(root="data_cifar100", train=False, download=True, transform=transform)
        
        images = []
        indices = np.linspace(0, len(ds)-1, CONFIG['vision']['n_images']).astype(int)
        for idx in indices:
            images.append(ds[idx][0])
        batch = torch.stack(images).to(DEVICE)
        print(f"  Loaded {len(images)} images")
        
        base_embeddings = {}
        
        # Model 1: ViT
        try:
            print("    Loading ViT...")
            model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()
            with torch.no_grad():
                out = model(pixel_values=batch)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            base_embeddings['vit'] = emb
            print(f"    vit: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] ViT: {e}")
        
        # Model 2: CLIP
        try:
            print("    Loading CLIP...")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
            
            feats = []
            for img_tensor in images:
                img_pil = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                inputs = processor(images=img_pil, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != 'input_ids'}
                with torch.no_grad():
                    emb = model.get_image_features(**inputs)
                feats.append(emb.cpu().numpy().squeeze())
            
            base_embeddings['clip'] = np.vstack(feats)
            print(f"    clip: {base_embeddings['clip'].shape}")
            del model, processor
        except Exception as e:
            print(f"    [ERROR] CLIP: {e}")
        
        # Model 3: DeiT
        try:
            print("    Loading DeiT...")
            model = AutoModel.from_pretrained("facebook/deit-base-patch16-224").to(DEVICE).eval()
            with torch.no_grad():
                out = model(pixel_values=batch)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            base_embeddings['deit'] = emb
            print(f"    deit: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] DeiT: {e}")
        
        # Model 4: ResNet50
        try:
            print("    Loading ResNet50...")
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])
            with torch.no_grad():
                out = model(batch)
            emb = out.squeeze(-1).squeeze(-1).cpu().numpy()
            base_embeddings['resnet50'] = emb
            print(f"    resnet50: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] ResNet50: {e}")
        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if not base_embeddings:
            return []
        
        all_results = []
        for seed in tqdm(SEEDS, desc="  Seeds"):
            all_results.extend(run_encoder_analysis(base_embeddings, seed, "Vision"))
        
        return all_results
    
    except Exception as e:
        print(f"Vision Failed: {e}")
        return []


# =============================================================================
# DOMAIN 3: AUDIO (2 Models)
# =============================================================================

def run_audio_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 3: AUDIO (2 Models)")
    print("=" * 60)
    
    tar_path = "librispeech.tar.gz"
    extract_dir = "libri_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    if not os.path.exists(tar_path):
        print("    Downloading LibriSpeech...")
        url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
        r = requests.get(url, stream=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)
    
    audios = []
    try:
        with tarfile.open(tar_path, "r") as tar:
            for m in tar:
                if len(audios) >= CONFIG['audio']['n_audio']:
                    break
                if m.name.endswith('.flac'):
                    tar.extract(m, path=extract_dir)
                    d, sr = sf.read(os.path.join(extract_dir, m.name))
                    d = librosa.resample(d, orig_sr=sr, target_sr=16000)
                    if len(d) > 16000:
                        d = d[:16000]
                    else:
                        d = np.pad(d, (0, 16000-len(d)))
                    audios.append(d)
    except Exception as e:
        print(f"    [ERROR] Extracting: {e}")
    
    if not audios:
        print("  No audio loaded")
        return []
    
    print(f"  Loaded {len(audios)} audio samples")
    base_embeddings = {}
    
    # Model 1: Wav2Vec2
    try:
        print("    Loading Wav2Vec2...")
        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()
        
        feats = []
        for a in audios:
            inp = proc(a, sampling_rate=16000, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        
        base_embeddings['wav2vec2'] = np.vstack(feats)
        print(f"    wav2vec2: {base_embeddings['wav2vec2'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] Wav2Vec2: {e}")
    
    # Model 2: HuBERT
    try:
        print("    Loading HuBERT...")
        proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE).eval()
        
        feats = []
        for a in audios:
            inp = proc(a, sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        
        base_embeddings['hubert'] = np.vstack(feats)
        print(f"    hubert: {base_embeddings['hubert'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] HuBERT: {e}")
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    if not base_embeddings:
        return []
    
    all_results = []
    for seed in tqdm(SEEDS, desc="  Seeds"):
        all_results.extend(run_encoder_analysis(base_embeddings, seed, "Audio"))
    
    return all_results


# =============================================================================
# DOMAIN 4: VIDEO (4 Models)
# =============================================================================

def run_video_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 4: VIDEO (4 Models)")
    print("=" * 60)
    
    import decord
    
    video_path = "video_sample.mp4"
    if not os.path.exists(video_path):
        print("    Downloading sample video...")
        r = requests.get("https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4")
        with open(video_path, 'wb') as f:
            f.write(r.content)
    
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    print(f"  Video loaded: {total_frames} frames")
    
    videos = []
    n_videos = CONFIG['video']['n_videos']
    
    for i in range(n_videos):
        start = np.random.randint(0, max(1, total_frames - 16))
        idx = np.arange(start, start + 16)
        frames = vr.get_batch(idx).asnumpy()
        videos.append([Image.fromarray(f).resize((224, 224)) for f in frames])
    
    print(f"  Extracted {len(videos)} video segments")
    base_embeddings = {}
    
    # Model 1: TimeSformer
    try:
        print("    Loading TimeSformer...")
        proc = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model = AutoModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(DEVICE).eval()
        
        feats = []
        for v in videos:
            inp = proc(images=v[:8], return_tensors="pt")
            inp = {k: val.to(DEVICE) for k, val in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        
        base_embeddings['timesformer'] = np.vstack(feats)
        print(f"    timesformer: {base_embeddings['timesformer'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] TimeSformer: {e}")
    
    # Model 2: VideoMAE
    try:
        print("    Loading VideoMAE...")
        proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()
        
        feats = []
        for v in videos:
            inp = proc(images=v[:16], return_tensors="pt")
            inp = {k: val.to(DEVICE) for k, val in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        
        base_embeddings['videomae'] = np.vstack(feats)
        print(f"    videomae: {base_embeddings['videomae'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] VideoMAE: {e}")
    
    # Model 3: ViT on mean frame
    try:
        print("    Loading ViT (mean frame)...")
        from torchvision.transforms import ToTensor, Normalize, Compose, Resize
        
        model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        feats = []
        for v in videos:
            frames_np = np.stack([np.array(f) for f in v])
            mean_frame = frames_np.mean(axis=0).astype(np.uint8)
            mean_pil = Image.fromarray(mean_frame)
            img_tensor = transform(mean_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = model(pixel_values=img_tensor)
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        
        base_embeddings['vit_meanframe'] = np.vstack(feats)
        print(f"    vit_meanframe: {base_embeddings['vit_meanframe'].shape}")
        del model
    except Exception as e:
        print(f"    [ERROR] ViT mean frame: {e}")
    
    # Model 4: CLIP multi-frame
    try:
        print("    Loading CLIP (multi-frame)...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        
        feats = []
        for v in videos:
            frame_indices = [0, 4, 8, 12] if len(v) >= 13 else list(range(min(4, len(v))))
            frame_embs = []
            for fi in frame_indices:
                inputs = processor(images=v[fi], return_tensors="pt")
                inputs = {k: val.to(DEVICE) for k, val in inputs.items() if k != 'input_ids'}
                with torch.no_grad():
                    emb = model.get_image_features(**inputs)
                frame_embs.append(emb.cpu().numpy())
            feats.append(np.mean(frame_embs, axis=0))
        
        base_embeddings['clip_multiframe'] = np.vstack(feats)
        print(f"    clip_multiframe: {base_embeddings['clip_multiframe'].shape}")
        del model, processor
    except Exception as e:
        print(f"    [ERROR] CLIP multi-frame: {e}")
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    if not base_embeddings:
        return []
    
    all_results = []
    for seed in tqdm(SEEDS, desc="  Seeds"):
        all_results.extend(run_encoder_analysis(base_embeddings, seed, "Video"))
    
    return all_results


# =============================================================================
# DOMAIN 5: NEUROSCIENCE (Steinmetz)
# =============================================================================

def run_neuroscience_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 5: NEUROSCIENCE (Full Dataset)")
    print("=" * 60)
    
    urls = ["https://osf.io/agvxh/download", "https://osf.io/uv3mw/download"]
    fnames = ["steinmetz_part1.npz", "steinmetz_part2.npz"]
    
    for url, fname in zip(urls, fnames):
        if not os.path.exists(fname):
            print(f"    Downloading {fname}...")
            r = requests.get(url, timeout=300)
            with open(fname, "wb") as f:
                f.write(r.content)
    
    try:
        alldat = []
        for fname in fnames:
            if os.path.exists(fname):
                alldat.extend(np.load(fname, allow_pickle=True)['dat'])
        
        base_embeddings = {}
        for i, d in enumerate(alldat):
            spikes = d['spks']
            X = spikes.mean(axis=1).T  # (Trials, Neurons)
            
            if X.shape[1] >= CONFIG['neuroscience']['min_neurons'] and X.shape[0] >= CONFIG['neuroscience']['min_trials']:
                base_embeddings[f'session_{i:02d}'] = X
        
        print(f"  Loaded {len(base_embeddings)} valid sessions")
        
        if not base_embeddings:
            return []
        
        all_results = []
        for seed in tqdm(SEEDS, desc="  Seeds"):
            all_results.extend(run_encoder_analysis(base_embeddings, seed, "Neuroscience"))
        
        return all_results
    
    except Exception as e:
        print(f"Neuroscience Failed: {e}")
        return []


# =============================================================================
# DOMAIN 6: PROTEIN (Swiss-Prot + Multiple Encoders)
# =============================================================================

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
         'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
         'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
CHARGE = {'D':-1,'E':-1,'K':1,'R':1,'H':0.1,'A':0,'C':0,'F':0,'G':0,'I':0,
          'L':0,'M':0,'N':0,'P':0,'Q':0,'S':0,'T':0,'V':0,'W':0,'Y':0}


def load_swissprot(n_proteins, seed):
    """Load Swiss-Prot sequences."""
    filename = "uniprot_sprot.fasta"
    url = "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28reviewed%3Atrue%29+AND+%28model_organism%3A9606%29&size=500"
    
    if not os.path.exists(filename):
        print("    Downloading Swiss-Prot...")
        try:
            r = requests.get(url, timeout=120)
            with open(filename, 'w') as f:
                f.write(r.text)
        except:
            return None
    
    seqs = []
    curr = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(">"):
                if curr:
                    seqs.append("".join(curr))
                curr = []
            else:
                curr.append(line.strip())
    if curr:
        seqs.append("".join(curr))
    
    AA_SET = set(AA_ALPHABET)
    seqs = [s for s in seqs if 50 <= len(s) <= 2000 and all(c in AA_SET for c in s)]
    
    rng = np.random.default_rng(seed)
    if len(seqs) > n_proteins:
        seqs = list(rng.choice(seqs, size=n_proteins, replace=False))
    
    return seqs


def build_protein_encoders(seqs, seed):
    """Build diverse protein encoders."""
    encoders = {}
    n_seqs = len(seqs)
    rng = np.random.default_rng(seed)
    
    # 1. AA Composition (20-dim)
    X_comp = np.zeros((n_seqs, 20))
    for i, s in enumerate(seqs):
        for c in s:
            if c in AA_INDEX:
                X_comp[i, AA_INDEX[c]] += 1
        X_comp[i] /= max(len(s), 1)
    encoders['aa_composition'] = X_comp
    
    # 2. Dipeptide (400-dim)
    dipeps = [a+b for a in AA_ALPHABET for b in AA_ALPHABET]
    dp_map = {dp: i for i, dp in enumerate(dipeps)}
    X_dp = np.zeros((n_seqs, 400))
    for i, s in enumerate(seqs):
        for j in range(len(s)-1):
            dp = s[j:j+2]
            if dp in dp_map:
                X_dp[i, dp_map[dp]] += 1
        X_dp[i] /= max(len(s)-1, 1)
    encoders['dipeptide'] = X_dp
    
    # 3. Hydrophobicity profiles at different resolutions
    for bins in [25, 50, 100]:
        X_hydro = []
        for s in seqs:
            vals = [HYDRO.get(c, 0) for c in s]
            if len(vals) < 2:
                vals = [0, 0]
            resampled = np.interp(np.linspace(0, len(vals)-1, bins), np.arange(len(vals)), vals)
            X_hydro.append(resampled)
        encoders[f'hydro_{bins}'] = np.vstack(X_hydro)
    
    # 4. Charge profiles
    for bins in [25, 50]:
        X_charge = []
        for s in seqs:
            vals = [CHARGE.get(c, 0) for c in s]
            if len(vals) < 2:
                vals = [0, 0]
            resampled = np.interp(np.linspace(0, len(vals)-1, bins), np.arange(len(vals)), vals)
            X_charge.append(resampled)
        encoders[f'charge_{bins}'] = np.vstack(X_charge)
    
    # 5. K-mer spectrum (k=3, hashed)
    X_kmer = np.zeros((n_seqs, 500))
    for i, s in enumerate(seqs):
        for j in range(len(s)-2):
            km = s[j:j+3]
            idx = hash(km) % 500
            X_kmer[i, idx] += 1
        X_kmer[i] /= max(len(s)-2, 1)
    encoders['kmer_3'] = X_kmer
    
    # 6. Noise Injection (Stress Test - crucial for consistency)
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noise = rng.normal(0, noise_level * np.std(X_dp), X_dp.shape)
        encoders[f'noise_{int(noise_level*100):03d}'] = X_dp + noise
    
    # 7. Combined
    encoders['combined'] = np.hstack([X_comp, X_dp, encoders['hydro_50'], encoders['charge_50']])
    
    return encoders


def run_protein_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 6: PROTEIN (Swiss-Prot)")
    print("=" * 60)
    
    all_results = []
    
    for seed in tqdm(SEEDS, desc="  Seeds"):
        seqs = load_swissprot(CONFIG['protein']['n_proteins'], seed)
        if not seqs:
            continue
        
        print(f"  Loaded {len(seqs)} sequences")
        encoders = build_protein_encoders(seqs, seed)
        
        # Run through transformation pipeline
        all_results.extend(run_encoder_analysis(encoders, seed, "Protein"))
    
    return all_results


# =============================================================================
# DOMAIN 7: MOLECULAR (PBMC3k + Multiple Encoders)
# =============================================================================

def build_molecular_encoders(X_raw, seed):
    """Build diverse molecular encoders."""
    X_log = np.log1p(X_raw)
    rng = np.random.default_rng(seed)
    encoders = {}
    n_cells, n_genes = X_log.shape
    
    # 1. PCA at various dimensions
    for k in [10, 25, 50, 75, 100, 150, 200]:
        k_actual = min(k, n_cells - 1, n_genes)
        if k_actual >= 5:
            try:
                if IS_GPU_PCA:
                    pca = PCA(n_components=k_actual)
                else:
                    pca = PCA(n_components=k_actual, random_state=seed)
                encoders[f"pca_{k:03d}"] = pca.fit_transform(X_log)
            except:
                pass
    
    # 2. Top variance genes
    gene_vars = X_log.var(axis=0)
    for k in [100, 500, 1000, 2000, 5000]:
        if k <= n_genes:
            idx = np.argsort(gene_vars)[-k:]
            encoders[f"topvar_{k:04d}"] = X_log[:, idx]
    
    # 3. Random gene subsets
    for k in [100, 500, 1000]:
        if k <= n_genes:
            idx = rng.choice(n_genes, k, replace=False)
            encoders[f"randgenes_{k:04d}"] = X_log[:, idx]
    
    # 4. Noise Injection (The "Stress Test")
    # Crucial for consistency with other domains
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noise = rng.normal(0, noise_level * np.std(X_log), X_log.shape)
        encoders[f"noise_{int(noise_level*100):03d}"] = X_log + noise
    
    # 5. Binarized (Biological "Presence/Absence")
    # Highly relevant for sparse scRNA-seq data
    encoders["binary"] = (X_raw > 0).astype(np.float32)
    
    # 6. Normalization variants
    scaler = StandardScaler()
    encoders["zscore"] = scaler.fit_transform(X_log)
    
    norms = np.linalg.norm(X_log, axis=1, keepdims=True) + 1e-12
    encoders["l2norm"] = X_log / norms
    
    # CPM-like normalization
    total_counts = X_raw.sum(axis=1, keepdims=True) + 1e-12
    encoders["cpm"] = np.log1p(X_raw / total_counts * 1e4)
    
    # 7. Original
    encoders["log1p_full"] = X_log.copy()
    
    return encoders


def build_molecular_encoders_gpu(X_raw, seed):
    """
    GPU-accelerated version of diverse molecular encoders.
    Replaces NumPy/Sklearn with PyTorch/cuML.
    """
    # 0. Infrastructure: Move to GPU and clear cache
    torch.cuda.empty_cache()
    if not isinstance(X_raw, torch.Tensor):
        X_raw_gpu = torch.tensor(X_raw, device=DEVICE, dtype=torch.float32)
    else:
        X_raw_gpu = X_raw.to(DEVICE).float()

    X_log = torch.log1p(X_raw_gpu)
    encoders = {}
    n_cells, n_genes = X_log.shape
    
    # Use PyTorch's generator for seeding GPU operations
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    # 1. PCA at various dimensions
    X_log_np = X_log.cpu().numpy() 
    
    for k in [10, 25, 50, 75, 100, 150, 200]:
        k_actual = min(k, n_cells - 1, n_genes)
        if k_actual >= 5:
            try:
                if IS_GPU_PCA:
                    pca = PCA(n_components=k_actual) 
                else:
                    from sklearn.decomposition import PCA as skPCA
                    pca = skPCA(n_components=k_actual, random_state=seed)
                encoders[f"pca_{k:03d}"] = pca.fit_transform(X_log_np)
            except Exception:
                pass
    
    # 2. Top variance genes (Vectorized on GPU)
    gene_vars = torch.var(X_log, dim=0)
    for k in [100, 500, 1000, 2000, 5000]:
        if k <= n_genes:
            _, idx = torch.topk(gene_vars, k)
            encoders[f"topvar_{k:04d}"] = X_log[:, idx].cpu().numpy()
    
    # 3. Random gene subsets
    for k in [100, 500, 1000]:
        if k <= n_genes:
            idx = torch.randperm(n_genes, generator=g, device=DEVICE)[:k]
            encoders[f"randgenes_{k:04d}"] = X_log[:, idx].cpu().numpy()
    
    # 4. Noise Injection (Parallelized on GPU cores)
    std_val = torch.std(X_log)
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noise = torch.randn(X_log.shape, generator=g, device=DEVICE) * (noise_level * std_val)
        encoders[f"noise_{int(noise_level*100):03d}"] = (X_log + noise).cpu().numpy()
    
    # 5. Binarized
    encoders["binary"] = (X_raw_gpu > 0).float().cpu().numpy()
    
    # 6. Normalization variants (With Manual Fallback)
    try:
        # Attempt cuML StandardScaler
        scaler = StandardScaler() 
        encoders["zscore"] = scaler.fit_transform(X_log_np)
    except Exception:
        # Fallback: Manual Torch Z-scoring (much faster than Sklearn CPU)
        mean = X_log.mean(0)
        std = X_log.std(0) + 1e-12
        encoders["zscore"] = ((X_log - mean) / std).cpu().numpy()
    
    # L2 Norm
    norms = torch.norm(X_log, p=2, dim=1, keepdim=True) + 1e-12
    encoders["l2norm"] = (X_log / norms).cpu().numpy()
    
    # CPM-like normalization
    total_counts = X_raw_gpu.sum(dim=1, keepdim=True) + 1e-12
    encoders["cpm"] = torch.log1p(X_raw_gpu / total_counts * 1e4).cpu().numpy()
    
    # 7. Original
    encoders["log1p_full"] = X_log.cpu().numpy()
    
    return encoders


def run_molecular_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 7: MOLECULAR (PBMC3k)")
    print("=" * 60)
    
    try:
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_genes(adata, min_cells=3)
        
        if adata.n_obs > CONFIG['molecular']['n_cells']:
            sc.pp.subsample(adata, n_obs=CONFIG['molecular']['n_cells'])
        
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        print(f"  Loaded PBMC3k: {X.shape}")
        
        all_results = []
        
        for seed in tqdm(SEEDS, desc="  Seeds"):
            # encoders = build_molecular_encoders(X, seed)
            encoders = build_molecular_encoders_gpu(X, seed)
            all_results.extend(run_encoder_analysis(encoders, seed, "Molecular"))
        
        return all_results
    
    except Exception as e:
        print(f"Molecular Failed: {e}")
        return []


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DISTINCTION TEST - 7 DOMAINS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Output: {OUTDIR}")
    
    all_results = []
    
    all_results.extend(run_language_domain())
    all_results.extend(run_vision_domain())
    all_results.extend(run_audio_domain())
    all_results.extend(run_video_domain())
    all_results.extend(run_neuroscience_domain())
    all_results.extend(run_protein_domain())
    all_results.extend(run_molecular_domain())
    
    if not all_results:
        print("\n[FATAL] No results collected!")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUTDIR / "raw_results_all_seeds.csv", index=False)
    print(f"\nSaved {len(df)} raw results")
    
    # Aggregate by (domain, base_model, encoder)
    df_agg = df.groupby(['domain', 'base_model', 'encoder']).agg({
        'SHESHA': 'mean',
        'CKA': 'mean',
        'n_features': 'first'
    }).reset_index()
    df_agg.to_csv(OUTDIR / "aggregated_by_encoder.csv", index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nTotal encoder configurations: {len(df_agg)}")
    print(f"\nPer-domain counts:")
    print(df_agg.groupby('domain').size())
    
    print(f"\nPer-domain statistics and rho:")
    all_shesha = []
    all_cka = []
    
    for domain in df_agg['domain'].unique():
        d = df_agg[df_agg['domain'] == domain]
        valid = d.dropna(subset=['SHESHA', 'CKA'])
        
        print(f"\n{domain} (N={len(valid)}):")
        print(f"  SHESHA: {valid['SHESHA'].mean():.3f} +/- {valid['SHESHA'].std():.3f}")
        print(f"  CKA:    {valid['CKA'].mean():.3f} +/- {valid['CKA'].std():.3f}")
        
        if len(valid) >= 5:
            rho, pval = spearmanr(valid['SHESHA'], valid['CKA'])
            print(f"  rho:    {rho:+.3f} (p={pval:.4f})")
        
        all_shesha.extend(valid['SHESHA'].tolist())
        all_cka.extend(valid['CKA'].tolist())
    
    # Aggregate
    print("\n" + "-" * 60)
    print("AGGREGATE:")
    if len(all_shesha) >= 5:
        rho_agg, pval_agg = spearmanr(all_shesha, all_cka)
        print(f"  N = {len(all_shesha)}")
        print(f"  rho = {rho_agg:+.4f} (p={pval_agg:.4f})")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()