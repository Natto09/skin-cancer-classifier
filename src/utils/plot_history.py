"""
Parses training-log text (paste terminal output, or pass --log_file) and
plots accuracy/loss curves. Replaces plot_history.py.

BUG FIX vs the original: the original script's regex looked for
r'Accuracy: ([\\d.]+)', but every trainer in this project actually prints
"Acc: NN.NN%" (see src/train/trainer.py's epoch summary line) -- so the
original regex never matched and accuracies list stayed empty. Fixed to
match "Acc:" (the real log format), and now parses BOTH train and val
accuracy/loss instead of just one column.

Usage:
    python -m src.utils.plot_history --log_file train_log.txt --out training_curves.png
"""

import argparse
import re

import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"Epoch (\d+).*?Train Loss:\s*([\d.]+)\s*Acc:\s*([\d.]+)%.*?"
    r"Val Loss:\s*([\d.]+)\s*Acc:\s*([\d.]+)%"
)


def parse_log(text):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    for line in text.strip().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        e, tl, ta, vl, va = m.groups()
        epochs.append(int(e))
        train_loss.append(float(tl)); train_acc.append(float(ta))
        val_loss.append(float(vl)); val_acc.append(float(va))
    return epochs, train_loss, train_acc, val_loss, val_acc


def plot_history(text, out_path=None):
    epochs, train_loss, train_acc, val_loss, val_acc = parse_log(text)
    if not epochs:
        print("[WARN] No matching 'Epoch N | Train Loss: ... Acc: ...%  Val Loss: ... Acc: ...%' "
              "lines found in the input.")
        return

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, color="#1f77b4", marker="o", label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, color="#2ca02c", marker="o", label="Val Accuracy", linewidth=2)
    plt.title("Accuracy over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, color="#ff7f0e", marker="o", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, color="#d62728", marker="o", label="Val Loss", linewidth=2)
    plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7); plt.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy/loss curves from a training log.")
    parser.add_argument("--log_file", default=None, help="Path to a saved training log (stdout capture)")
    parser.add_argument("--out", default=None, help="Save the plot here instead of showing it")
    args = parser.parse_args()

    if args.log_file:
        with open(args.log_file, encoding="utf-8") as f:
            text = f.read()
    else:
        print("Paste training log text, then Ctrl-D (Linux/Mac) or Ctrl-Z+Enter (Windows):")
        import sys
        text = sys.stdin.read()

    plot_history(text, out_path=args.out)


if __name__ == "__main__":
    main()
