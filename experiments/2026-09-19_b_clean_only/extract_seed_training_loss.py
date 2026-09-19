"""Extract 10-epoch training-loss checkpoints from the six B6 model fits."""

import csv
import re
from pathlib import Path


OUT = Path(__file__).resolve().parent
PATTERN = re.compile(r"\[training\] (Conformer|AlexNet): .*? (\d+)/200 .*? loss=([0-9.eE+-]+)")


def main():
    rows = []
    for seed in (42, 43, 44):
        # PowerShell's *> redirection writes UTF-16 LE on this host.
        text = (OUT / f"matched_seed{seed}.log").read_text(encoding="utf-16")
        blocks = {"Conformer": [], "AlexNet": []}
        for line in text.splitlines():
            match = PATTERN.search(line)
            if not match:
                continue
            model, epoch_text, loss_text = match.groups()
            epoch = int(epoch_text)
            if epoch == 10:
                blocks[model].append([])
            if not blocks[model]:
                raise ValueError(f"Missing first epoch for {seed} {model}")
            blocks[model][-1].append((epoch, float(loss_text)))
        for model, model_blocks in blocks.items():
            if len(model_blocks) != 4 or any([epoch for epoch, _ in block] != list(range(10, 201, 10))
                                             for block in model_blocks):
                raise ValueError(f"Incomplete loss history for {seed} {model}: "
                                 f"{[len(block) for block in model_blocks]}")
            for index, block in enumerate(model_blocks):
                snr = "-8" if index < 2 else "-16"
                fit_role = "inner_weight_fit" if index % 2 == 0 else "full_training"
                rows.extend({"seed": seed, "snr": snr, "model": model,
                             "fit_role": fit_role, "epoch": epoch, "training_loss": loss}
                            for epoch, loss in block)
    with (OUT / "matched_seed_training_loss.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} training-loss checkpoints; validation loss was not logged.")


if __name__ == "__main__":
    main()
