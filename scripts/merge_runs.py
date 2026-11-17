"""Merge independently trained models only if their manifests/protocol agree."""
import argparse
import hashlib
import json
from pathlib import Path
from shutil import copy2


def merge(root: Path, dest: Path):
    names = ('logistic', 'cnn', 'mobilenet')
    items = []
    hashes = []
    for name in names:
        p = root / name
        data = json.loads((p/'metrics.json').read_text())
        if data['status'] != 'complete' or len(data['models']) != 1 or data['models'][0]['model'] != name:
            raise ValueError(f'Run for {name} incomplete or mismatched')
        if data['seed'] != 42 or data['dataset_total_discovered'] != 40000 or data['dataset_total_used'] != 40000:
            raise ValueError(f'Run for {name} does not match the planned 40,000-image protocol')
        if data['split'] != 'image_level_stratified':
            raise ValueError(f'Unexpected split method in {name}')
        hashes.append(hashlib.sha256((p/'splits.csv').read_bytes()).hexdigest())
        items.append(data)
    if len(set(hashes)) != 1:
        raise ValueError('Split assignments differ between models; cannot compare')
    ref = items[0]
    for item in items[1:]:
        if item['split_counts'] != ref['split_counts']:
            raise ValueError('Class counts differ between model runs')
    combined = {**ref, 'models': [item['models'][0] for item in items]}
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'metrics.json').write_text(json.dumps(combined, indent=2)+'\n')
    copy2(root/'logistic'/'splits.csv', dest/'splits.csv')
    for name in names:
        copy2(root/name/f'{name}_errors.csv', dest/f'{name}_errors.csv')
    print(json.dumps({'model_metrics': {item['models'][0]['model']: item['models'][0]['test_metrics'] for item in items},
                      'source_split_sha256': hashes[0]}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path('runs/per_model'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/final_seed42'))
    a = p.parse_args()
    merge(a.run_dir, a.output_dir)
