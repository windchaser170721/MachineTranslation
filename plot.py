import re
import matplotlib.pyplot as plt
import config
import numpy as np

from utils import get_std_opt, NoamOpt
from model import make_model

def check_opt():
    model = make_model(
        config.src_vocab_size, config.tgt_vocab_size,
        config.n_layers, config.d_model,
        config.d_ff, config.n_heads, config.dropout
    )
    opt1 = get_std_opt(model)
    opt2 = NoamOpt(512, 1, 20000, None)
    opt3 = NoamOpt(256, 1, 10000, None)
    opts   = [opt1, opt2, opt3]
    labels = ["512:10000", "512:20000", "256:10000"]
    colors = ["#FF4500", "#006400", "#4B0082"]  
    steps = np.arange(1, 50000)
    plt.figure(figsize=(10, 6))
    for opt_obj, label, color in zip(opts, labels, colors):
        rates = [opt_obj.rate(i) for i in steps]
        plt.plot(steps, rates, label=label, color=color, linewidth=3)

    plt.title("Learning Rate Schedules", fontsize=18, weight="bold")
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Learning Rate", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12, frameon=True, edgecolor="gray")
    plt.tight_layout()
    plt.savefig("respics/opt.png", dpi=300, bbox_inches="tight")

def res_plot():

    epochs = []
    train_losses = []
    dev_losses = []
    dev_bleus = []

    log_path = "exp17/train.log"  
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m_train = re.search(r'Epoch:\s*(\d+),\s*loss:\s*([\d.]+)', line)
            if m_train:
                epoch = int(m_train.group(1))
                loss = float(m_train.group(2))
                if epoch not in epochs:
                    epochs.append(epoch)
                train_losses.append(loss)

            m_dev = re.search(r'Epoch:\s*(\d+),\s*Dev loss:\s*([\d.]+),\s*Bleu Score:\s*([\d.]+)', line)
            if m_dev:
                dev_losses.append(float(m_dev.group(2)))
                dev_bleus.append(float(m_dev.group(3)))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, dev_losses, label="Dev Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs. Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("respics/loss_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, dev_bleus, label="Dev BLEU", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("Validation BLEU vs. Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("respics/bleu_curve.png", dpi=300)
    plt.show()
