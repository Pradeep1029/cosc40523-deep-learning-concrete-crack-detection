"""Retrieve a public mirror of Özgenel's 40k concrete crack image dataset.

Uses the Kaggle public dataset API, falling back to a CC-BY-4.0 Hugging Face
mirror. Counts and image directories are verified before any model is trained.
Do not use a mirror's prebuilt train/test partition: the main experiment creates
its own reproducible 70/15/15 split across all 40,000 patches.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
from zipfile import ZipFile, BadZipFile
import requests

KAGGLE = 'https://www.kaggle.com/api/v1/datasets/download/arnavr10880/concrete-crack-images-for-classification'
HF_BASE = 'https://huggingface.co/datasets/mohammadnajeeb/concrete_crack_images/resolve/main/data/'
HF_ZIPS = ['train.zip', 'validation.zip', 'test.zip']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def fetch(url: str, dest: Path) -> str:
    print('Downloading', url, flush=True)
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(45, 180), headers={'User-Agent': 'Mozilla/5.0 concrete-crack-reproducibility/1.0'}, allow_redirects=True) as r:
        r.raise_for_status()
        with dest.open('wb') as out:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    out.write(chunk)
                    sha.update(chunk)
    if dest.stat().st_size < 1000000:
        raise RuntimeError(f'Implausibly small archive: {dest.stat().st_size} bytes from {url}')
    print(f'Archive {dest.name}: {dest.stat().st_size:,} bytes, SHA256 {sha.hexdigest()}', flush=True)
    return sha.hexdigest()


def extract_zip(archive: Path, output: Path, duplicate_check: dict):
    counters = Counter()
    with ZipFile(archive) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        for member in files:
            pure = PurePosixPath(member.filename.replace('\\', '/'))
            if '..' in pure.parts:
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if pure.suffix.lower() not in EXTENSIONS:
                continue
            matches = [p for p in pure.parts[:-1] if p.lower() in ('positive', 'negative')]
            if len(matches) != 1:
                continue
            cls = matches[0].capitalize()
            # Copy filenames as-is where unique. Distinct archives occasionally
            # repeat names for different split partitions; keep both as patches.
            name = pure.name
            key = (cls, name.lower())
            if key in duplicate_check:
                existing_crc, existing_size = duplicate_check[key]
                if (existing_crc, existing_size) == (member.CRC, member.file_size):
                    raise ValueError(f'Duplicate source patch appears twice: {pure}')
                name = f'{archive.stem}_{pure.name}'
                key = (cls, name.lower())
                if key in duplicate_check:
                    raise ValueError(f'Name collision after prefixing: {pure}')
            duplicate_check[key] = (member.CRC, member.file_size)
            target = output / cls / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member, 'r') as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
            counters[cls] += 1
    if not sum(counters.values()):
        raise ValueError(f'Archive contains no recognizable class images: {archive.name}')
    print('Unpacked', archive.name, dict(counters), flush=True)


def get_dataset(out: Path, source='auto'):
    if any((out / c).exists() for c in ('Positive', 'Negative')):
        raise FileExistsError(f'{out} already contains class folders; remove them before rerunning downloader')
    candidates = ([('kaggle', [KAGGLE]), ('huggingface', [HF_BASE + x for x in HF_ZIPS])]
                  if source == 'auto' else [('kaggle', [KAGGLE])] if source == 'kaggle'
                  else [('huggingface', [HF_BASE + x for x in HF_ZIPS])])
    failures = []
    for name, urls in candidates:
        with tempfile.TemporaryDirectory(prefix='crack_download_') as td:
            target = Path(td)/'assembled'
            duplicate_check = {}
            hashes = {}
            try:
                for i, url in enumerate(urls):
                    temp = Path(td)/f'{i}.zip'
                    hashes[url] = fetch(url, temp)
                    extract_zip(temp, target, duplicate_check)
                    temp.unlink()
                actual = {label: len(list((target / label).glob('*'))) for label in ('Negative', 'Positive')}
                if actual != {'Negative': 20000, 'Positive': 20000}:
                    raise ValueError(f'Expected 20,000 images of each class; got {actual}')
                out.mkdir(parents=True, exist_ok=True)
                for label in ('Negative', 'Positive'):
                    shutil.move(str(target / label), str(out / label))
                meta = {'dataset_title': 'Concrete Crack Images for Classification',
                        'original_doi': '10.17632/5y9wdsg2zt.1',
                        'mirror': name, 'urls': urls, 'zip_sha256': hashes,
                        'counts': actual, 'total_images': 40000, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                (out / '_provenance.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
                print('Verified complete 40,000-image dataset', json.dumps(meta, indent=2), flush=True)
                return meta
            except (requests.RequestException, BadZipFile, ValueError, RuntimeError) as e:
                failures.append(f'{name}: {e}')
                print('Mirror failed:', failures[-1], flush=True)
    raise RuntimeError('All public dataset sources failed: ' + '; '.join(failures))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data'))
    p.add_argument('--source', choices=['auto','kaggle','huggingface'], default='auto')
    a = p.parse_args()
    get_dataset(a.output, a.source)
