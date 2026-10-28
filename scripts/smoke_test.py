"""Offline CNN training, modality ablations and contact-candidate replay on real data."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import yaml


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                         nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(192, 48), nn.ReLU())


class FusionCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.visual = Encoder()
        self.tactile = Encoder()
        self.vision_head = nn.Linear(48, classes)
        self.touch_head = nn.Linear(48, classes)
        self.fusion_head = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, classes))

    def forward(self, vision, touch):
        v, t = self.visual(vision), self.tactile(touch)
        return self.vision_head(v), self.touch_head(t), self.fusion_head(torch.cat([v, t], dim=1))


def images(records, field, size):
    return torch.stack([torch.from_numpy(np.asarray(Image.open(ROOT / r[field]).convert("RGB").resize((size, size))).copy()).permute(2, 0, 1).float() / 255 for r in records])


def accuracy(logits, labels, classes):
    prediction = logits.argmax(1)
    return {"accuracy": float((prediction == labels).float().mean()),
            "balanced_accuracy": float(np.mean([float((prediction[labels == k] == k).float().mean()) for k in range(classes)])),
            "count": len(labels)}


def contact_replay(model, records, vision, touch, cfg, classes, output):
    """Select among recorded contacts using visible features only, then reveal touch.

    Coordinates/actions are absent from this dataset. This is offline candidate replay,
    not closed-loop robot control or a learned spatial contact policy.
    """
    with torch.no_grad():
        vf, tf = model.visual(vision), model.tactile(touch)
    episodes = defaultdict(dict)
    for i, record in enumerate(records):
        if record['split'] == 'test':
            episodes[record['sequence']].setdefault(record['contact'], []).append(i)
    traces, summaries, demo = [], {}, None
    for policy in ['random', 'active_visual_diversity']:
        rng = np.random.default_rng(cfg['seed'])
        trials = []
        for sequence, contacts in sorted(episodes.items()):
            candidates = [indices[len(indices) // 2] for _, indices in sorted(contacts.items())]
            assert len({records[i]['label'] for i in candidates}) == 1
            initial = candidates[0]
            selected, available = [], candidates.copy()
            label = records[initial]['label']
            steps = []
            with torch.no_grad():
                probability = model.vision_head(vf[initial:initial + 1]).softmax(1)[0]
                steps.append({'step': 0, 'sample_index': initial, 'prediction': int(probability.argmax()),
                              'confidence': float(probability.max()), 'probabilities': probability.cpu().tolist()})
                while len(selected) < cfg['max_contacts'] and available and (len(selected) < cfg['min_contacts'] or float(probability.max()) < cfg['confidence_threshold']):
                    if policy == 'random':
                        chosen = int(rng.choice(available))
                    else:
                        observed = vf[[initial] + selected]
                        distances = torch.cdist(vf[available], observed).min(dim=1).values
                        chosen = available[int(distances.argmax())]
                    selected.append(chosen)
                    available.remove(chosen)
                    pooled_touch = tf[selected].mean(dim=0, keepdim=True)
                    probability = model.fusion_head(torch.cat([vf[initial:initial + 1], pooled_touch], 1)).softmax(1)[0]
                    steps.append({'step': len(selected), 'sample_index': chosen, 'prediction': int(probability.argmax()),
                                  'confidence': float(probability.max()), 'probabilities': probability.cpu().tolist()})
            assert cfg['min_contacts'] <= len(selected) <= cfg['max_contacts']
            stop_reason = ('confidence' if float(probability.max()) >= cfg['confidence_threshold'] else
                           'contact_budget' if len(selected) >= cfg['max_contacts'] else 'candidates_exhausted')
            trial = {'policy': policy, 'sequence': sequence, 'target': classes[label], 'target_id': label,
                     'available_contacts': len(candidates), 'contacts_used': len(selected),
                     'correct': int(probability.argmax()) == label, 'stop_reason': stop_reason,
                     'steps': steps, 'chosen_contacts': [records[i]['contact'] for i in selected]}
            trials.append(trial)
            if policy == 'active_visual_diversity' and (demo is None or len(steps) > len(demo['steps'])):
                demo = trial
        summaries[policy] = {'episodes': len(trials), 'accuracy': float(np.mean([x['correct'] for x in trials])),
                             'mean_contacts': float(np.mean([x['contacts_used'] for x in trials])),
                             'max_contacts': max(x['contacts_used'] for x in trials)}
        traces.extend(trials)
    (output / 'contact_traces.json').write_text(json.dumps(traces, indent=2) + '\n')
    frames = []
    for step in demo['steps']:
        record = records[step['sample_index']]
        canvas = Image.new('RGB', (700, 320), '#f4f6f8')
        draw = ImageDraw.Draw(canvas)
        visual = Image.open(ROOT / record['vision']).resize((240, 240))
        canvas.paste(visual, (15, 45))
        if step['step']:
            tactile = Image.open(ROOT / record['touch']).resize((240, 240))
            canvas.paste(tactile, (275, 45))
        else:
            draw.text((310, 160), 'Touch not revealed', fill='black')
        draw.text((15, 12), f"Offline active contact replay | step {step['step']} | target: {demo['target']}", fill='black')
        draw.text((530, 65), 'Prediction:\n' + classes[step['prediction']], fill='black')
        draw.text((530, 120), f"Confidence:\n{step['confidence']:.3f}", fill='black')
        draw.text((15, 295), f"Recording: {demo['sequence']} | heuristic: visible feature diversity", fill='black')
        frames.append(canvas)
    frames[0].save(output / 'active_demo.gif', save_all=True, append_images=frames[1:], duration=1300, loop=0)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/smoke.yaml'))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    start = time.time()
    torch.set_num_threads(4)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['device'])
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable. Use device: cpu.')
    records = json.loads((ROOT / 'data/processed/samples.json').read_text())
    data_summary = json.loads((ROOT / 'data/processed/summary.json').read_text())
    classes = data_summary['classes']
    vision, touch = images(records, 'vision', cfg['image_size']).to(device), images(records, 'touch', cfg['image_size']).to(device)
    labels = torch.tensor([r['label'] for r in records], device=device)
    train = torch.tensor([i for i, r in enumerate(records) if r['split'] == 'train'], device=device)
    test = torch.tensor([i for i, r in enumerate(records) if r['split'] == 'test'], device=device)
    assert set(data_summary['train_sequences']).isdisjoint(data_summary['test_sequences'])
    model = FusionCNN(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    counts = torch.bincount(labels[train], minlength=len(classes)).float()
    sampling_weights = 1 / counts[labels[train]]
    before = next(model.parameters()).detach().clone()
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        order = train[torch.multinomial(sampling_weights, len(train), replacement=True)]
        total, seen = 0., 0
        for ids in order.split(cfg['batch_size']):
            optimizer.zero_grad()
            predictions = model(vision[ids], touch[ids])
            loss = sum(nn.functional.cross_entropy(p, labels[ids]) for p in predictions) / 3
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            total += float(loss.detach()) * len(ids); seen += len(ids)
        history.append({'epoch': epoch + 1, 'loss': total / seen})
        print(f"epoch {epoch + 1:02d}: mean loss {total / seen:.5f}", flush=True)
    assert not torch.equal(before, next(model.parameters()).detach()), 'Optimizer did not update model'
    model.eval()
    with torch.no_grad():
        predictions = model(vision[test], touch[test])
    metrics = {name: accuracy(prediction, labels[test], len(classes)) for name, prediction in zip(['vision_only', 'touch_only', 'vision_touch_fusion'], predictions)}
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    checkpoint = output / 'smoke_model.pt'
    torch.save({'state_dict': model.state_dict(), 'classes': classes, 'config': cfg}, checkpoint)
    restored = FusionCNN(len(classes)).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['state_dict'])
    restored.eval()
    with torch.no_grad():
        for left, right in zip(model(vision[test[:2]], touch[test[:2]]), restored(vision[test[:2]], touch[test[:2]])):
            torch.testing.assert_close(left, right)
    replay = contact_replay(model, records, vision, touch, cfg, classes, output)
    result = {'status': 'passed', 'config': cfg, 'torch_version': str(torch.__version__), 'device': str(device),
              'mps_available': torch.backends.mps.is_available(), 'classes': classes,
              'train_pairs': len(train), 'test_pairs': len(test), 'split_unit': 'recording',
              'checks': ['real_video_decoding', 'timestamp_alignment', 'recording_disjoint_split', 'finite_loss_and_gradients',
                         'optimizer_parameter_update', 'three_modality_heads', 'checkpoint_reload_equivalence', 'contact_selection_and_stopping'],
              'history': history, 'classification': metrics, 'contact_replay': replay,
              'elapsed_seconds': time.time() - start,
              'limitations': ['Only 3 classes from a small, severely imbalanced probe subset.',
                              'Custom recording-disjoint split, not the official benchmark split.',
                              'Recording separation does not guarantee separation of physical objects.',
                              'Contact replay selects prerecorded candidates using visible feature diversity; no spatial coordinates or real robot actions.',
                              'Smoke-test metrics do not establish fusion or active-policy improvement.']}
    (output / 'smoke_report.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'training_loss.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'loss']); writer.writeheader(); writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot([x['epoch'] for x in history], [x['loss'] for x in history], marker='o')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy', title='Smoke training on real paired frames')
    axes[1].bar(['Vision', 'Touch', 'Fusion'], [x['accuracy'] for x in metrics.values()])
    axes[1].set(ylim=(0, 1), ylabel='Accuracy', title='Tiny held-out recording split (not benchmark)')
    figure.tight_layout(); figure.savefig(output / 'smoke_results.png', dpi=150); plt.close(figure)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
