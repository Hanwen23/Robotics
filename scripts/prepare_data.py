"""Extract timestamp-aligned real image pairs and split by recording, never by frame."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def frame(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {index}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-contact", type=int, default=3)
    args = parser.parse_args()
    raw = ROOT / "data/raw/touch_and_go"
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    names = {}
    for line in (ROOT / "references/category_reference.txt").read_text().splitlines():
        name, value = line.split(":", 1)
        names[int(value.split()[0])] = name.strip()
    contacts = defaultdict(list)
    for line in (ROOT / "references/train_nointer_touch_instances.txt").read_text().splitlines():
        seq, start, end, label = line.split(",")
        if (raw / seq / "video.mp4").exists() and int(label) >= 0:
            contacts[seq].append((int(start), int(end), int(label)))
    sequences_by_class = defaultdict(set)
    for seq, intervals in contacts.items():
        for _, _, label in intervals:
            sequences_by_class[label].add(seq)
    # Three classes have >=2 separate recordings, allowing recording-disjoint validation.
    eligible = sorted(k for k, seqs in sequences_by_class.items() if len(seqs) >= 2)
    class_ids = {label: index for index, label in enumerate(eligible)}
    test_sequences = set()
    for label in eligible:
        sequences = sorted(sequences_by_class[label])
        test_sequences.update(sequences[::max(2, len(sequences) // max(1, len(sequences) // 3))])
    train_sequences = set(contacts) - test_sequences
    assert not train_sequences & test_sequences
    records, timing_errors, previews = [], [], []
    for seq in sorted(contacts):
        usable = [(i, a, b, c) for i, (a, b, c) in enumerate(contacts[seq]) if c in class_ids]
        if not usable:
            continue
        path = raw / seq
        visual_times = np.load(path / "time1.npy", allow_pickle=False)
        tactile_times = np.load(path / "time2.npy", allow_pickle=False)
        assert np.all(np.diff(visual_times) >= 0) and np.all(np.diff(tactile_times) >= 0)
        visual_cap = cv2.VideoCapture(str(path / "video.mp4"))
        tactile_cap = cv2.VideoCapture(str(path / "gelsight.mp4"))
        assert visual_cap.isOpened() and tactile_cap.isOpened()
        assert len(visual_times) <= int(visual_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert len(tactile_times) <= int(tactile_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for contact, start, end, label in usable:
            if not (0 <= start <= end < len(visual_times)):
                raise ValueError(f"Contact interval outside video: {seq}, {start}, {end}")
            indices = np.unique(np.linspace(start, end, args.frames_per_contact, dtype=int))
            for visual_index in indices:
                tactile_index = int(np.argmin(np.abs(tactile_times - visual_times[visual_index])))
                delta = float(abs(tactile_times[tactile_index] - visual_times[visual_index]))
                if delta > 0.1:
                    raise ValueError(f"Alignment error over 100 ms: {seq}, {delta}")
                timing_errors.append(delta)
                visual = Image.fromarray(frame(visual_cap, visual_index)).resize((96, 96))
                tactile = Image.fromarray(frame(tactile_cap, tactile_index)).resize((96, 96))
                sample_id = f"{seq}_{contact:03d}_{visual_index:06d}"
                sample_dir = out / seq
                sample_dir.mkdir(exist_ok=True)
                vp, tp = sample_dir / f"{sample_id}_vision.jpg", sample_dir / f"{sample_id}_touch.jpg"
                visual.save(vp, quality=95)
                tactile.save(tp, quality=95)
                split = "test" if seq in test_sequences else "train"
                records.append({"id": sample_id, "sequence": seq, "contact": f"{seq}_{contact:03d}",
                                "vision": str(vp.relative_to(ROOT)), "touch": str(tp.relative_to(ROOT)),
                                "label": class_ids[label], "material": names[label], "source_label": label,
                                "split": split, "vision_frame": int(visual_index), "touch_frame": tactile_index,
                                "time_error_seconds": delta})
                if len([x for x in previews if x[0] == label]) < 2:
                    previews.append((label, visual, tactile, split))
        visual_cap.release()
        tactile_cap.release()
    (out / "samples.json").write_text(json.dumps(records, indent=2) + "\n")
    summary = {"classes": [names[x] for x in eligible], "source": "real Touch and Go paired video frames",
               "split_unit": "recording/sequence", "split_is_official": False,
               "excluded_classes": {names[k]: "only one recording" for k in sequences_by_class if k not in class_ids},
               "matched_labeled_sequences": len(contacts),
               "used_sequences": len({x['sequence'] for x in records}),
               "contacts": len({x['contact'] for x in records}), "pairs": len(records),
               "split_counts": dict(Counter(x['split'] for x in records)),
               "class_counts": {split: dict(Counter(x['material'] for x in records if x['split'] == split)) for split in ['train', 'test']},
               "train_sequences": sorted({x['sequence'] for x in records if x['split'] == 'train'}),
               "test_sequences": sorted({x['sequence'] for x in records if x['split'] == 'test'}),
               "max_alignment_error_ms": max(timing_errors) * 1000,
               "mean_alignment_error_ms": float(np.mean(timing_errors) * 1000),
               "limitations": "Probe subset, class imbalance, few independent recordings; only validates the pipeline, not research performance."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    canvas = Image.new("RGB", (400, 120 * len(previews)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, visual, tactile, split) in enumerate(previews):
        canvas.paste(visual, (0, i * 120 + 20))
        canvas.paste(tactile, (100, i * 120 + 20))
        draw.text((210, i * 120 + 50), f"{names[label]} ({split})\nvision / touch", fill="black")
    (ROOT / "outputs").mkdir(exist_ok=True)
    canvas.save(ROOT / "outputs/data_preview.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
