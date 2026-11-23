import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from model import GPTMini

# =========================================================
# Configuration
# =========================================================
CONFIG = {
    # Data
    "data_path": "data/tiny-shakespeare.txt",
    "train_fraction": 0.80,
    "val_fraction": 0.10,

    # Model
    "block_size": 128,
    "n_embed": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.10,

    # Training
    "batch_size": 32,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-6,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "max_iters": 12000,
    "warmup_iters": 300,
    "seed": 42,

    # Evaluation and saving
    "eval_interval": 500,
    "eval_iters": 100,
    "patience": 6,
    "min_delta": 1e-4,
    "output_dir": "output",
    "resume_path": "output/tinygpt_resume.pt",
    "best_path": "output/tinygpt_best.pt",
    "final_path": "output/tinygpt_final.pt",

    # Generation
    "sample_tokens": 300,

    # W&B
    "wandb_project": "tinygpt-from-scratch",
    "wandb_run_name": "tinygpt-warmup-cosine",
}


# =========================================================
# Reproducibility and device
# =========================================================
torch.manual_seed(CONFIG["seed"])

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# =========================================================
# Data and character tokenizer
# =========================================================
with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}


def encode(text_string):
    return [stoi[char] for char in text_string]


def decode(token_ids):
    return "".join([itos[token_id] for token_id in token_ids])


data = torch.tensor(encode(text), dtype=torch.long)

n_total = len(data)
n_train = int(CONFIG["train_fraction"] * n_total)
n_val = int(CONFIG["val_fraction"] * n_total)

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]


def get_batch(split):
    if split == "train":
        dataset = train_data
    elif split == "val":
        dataset = val_data
    elif split == "test":
        dataset = test_data
    else:
        raise ValueError(f"Unknown split: {split}")

    if len(dataset) <= CONFIG["block_size"] + 1:
        raise ValueError(
            f"The {split} split is too small for block_size="
            f"{CONFIG['block_size']}."
        )

    indices = torch.randint(
        len(dataset) - CONFIG["block_size"] - 1,
        (CONFIG["batch_size"],),
    )

    x = torch.stack([
        dataset[index:index + CONFIG["block_size"]]
        for index in indices
    ])

    y = torch.stack([
        dataset[index + 1:index + CONFIG["block_size"] + 1]
        for index in indices
    ])

    return x.to(device), y.to(device)


# =========================================================
# Build model and optimizer
# =========================================================
model = GPTMini(
    vocab_size=vocab_size,
    block_size=CONFIG["block_size"],
    n_embed=CONFIG["n_embed"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["n_layer"],
    dropout=CONFIG["dropout"],
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)


# =========================================================
# Warmup + cosine learning-rate schedule
# =========================================================
def learning_rate_multiplier(current_iteration):
    if current_iteration < CONFIG["warmup_iters"]:
        return (current_iteration + 1) / CONFIG["warmup_iters"]

    progress = (
        current_iteration - CONFIG["warmup_iters"]
    ) / max(
        1,
        CONFIG["max_iters"] - CONFIG["warmup_iters"],
    )

    progress = min(1.0, progress)

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    min_ratio = CONFIG["min_learning_rate"] / CONFIG["learning_rate"]

    return min_ratio + (1.0 - min_ratio) * cosine_factor


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=learning_rate_multiplier,
)


# =========================================================
# Loss estimation
# =========================================================
@torch.no_grad()
def estimate_loss(split, num_batches):
    model.eval()

    losses = []

    for _ in range(num_batches):
        x, y = get_batch(split)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# =========================================================
# Checkpoint helpers
# =========================================================
def save_checkpoint(path, iteration, best_val_loss):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": CONFIG,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# =========================================================
# W&B setup
# =========================================================
wandb.init(
    project=CONFIG["wandb_project"],
    name=CONFIG["wandb_run_name"],
    config=CONFIG,
)

wandb.watch(
    model,
    log="all",
    log_freq=100,
)


# =========================================================
# Resume if available
# =========================================================
start_iteration = 0
best_val_loss = float("inf")
evaluations_without_improvement = 0

if os.path.exists(CONFIG["resume_path"]):
    checkpoint = load_checkpoint(CONFIG["resume_path"])

    start_iteration = checkpoint["iteration"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Resumed from iteration {start_iteration}; "
        f"best validation loss: {best_val_loss:.4f}"
    )


# =========================================================
# Training loop
# =========================================================
training_start_time = time.time()

for iteration in range(start_iteration, CONFIG["max_iters"]):
    model.train()

    x_batch, y_batch = get_batch("train")

    logits, loss = model(x_batch, y_batch)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        CONFIG["grad_clip"],
    )

    optimizer.step()
    scheduler.step()

    current_learning_rate = optimizer.param_groups[0]["lr"]

    wandb.log(
        {
            "train/loss": loss.item(),
            "train/learning_rate": current_learning_rate,
            "train/iteration": iteration,
        },
        step=iteration,
    )

    should_evaluate = (
        iteration % CONFIG["eval_interval"] == 0
        or iteration == CONFIG["max_iters"] - 1
    )

    if should_evaluate:
        train_loss = estimate_loss(
            split="train",
            num_batches=CONFIG["eval_iters"],
        )

        val_loss = estimate_loss(
            split="val",
            num_batches=CONFIG["eval_iters"],
        )

        val_perplexity = math.exp(val_loss)

        elapsed_time = time.time() - training_start_time

        print(
            f"iteration {iteration:6d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val perplexity {val_perplexity:.2f} | "
            f"lr {current_learning_rate:.2e} | "
            f"time {elapsed_time:.1f}s"
        )

        wandb.log(
            {
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
                "eval/val_perplexity": val_perplexity,
                "eval/learning_rate": current_learning_rate,
                "eval/iteration": iteration,
            },
            step=iteration,
        )

        # Always save a resumable checkpoint
        save_checkpoint(
            path=CONFIG["resume_path"],
            iteration=iteration,
            best_val_loss=best_val_loss,
        )

        improved = val_loss < best_val_loss - CONFIG["min_delta"]

        if improved:
            best_val_loss = val_loss
            evaluations_without_improvement = 0

            save_checkpoint(
                path=CONFIG["best_path"],
                iteration=iteration,
                best_val_loss=best_val_loss,
            )

            wandb.log(
                {
                    "best/val_loss": best_val_loss,
                    "best/iteration": iteration,
                },
                step=iteration,
            )

            print(
                f"New best validation loss: {best_val_loss:.4f}; "
                f"saved to {CONFIG['best_path']}"
            )
        else:
            evaluations_without_improvement += 1

        # Generate a sample for qualitative monitoring
        start_ids = torch.zeros(
            (1, 1),
            dtype=torch.long,
            device=device,
        )

        sample_ids = model.generate(
            start_ids,
            max_new_tokens=CONFIG["sample_tokens"],
        )[0].tolist()

        generated_sample = decode(sample_ids)

        print("---- generated sample ----")
        print(generated_sample)
        print("--------------------------")

        wandb.log(
            {
                "samples/generated_text": wandb.Table(
                    columns=["iteration", "generated_text"],
                    data=[[iteration, generated_sample]],
                )
            },
            step=iteration,
        )

        if evaluations_without_improvement >= CONFIG["patience"]:
            print(
                f"Early stopping at iteration {iteration}; "
                f"no validation improvement for "
                f"{CONFIG['patience']} evaluations."
            )
            break


# =========================================================
# Final evaluation on held-out test set
# =========================================================
best_checkpoint = torch.load(
    CONFIG["best_path"],
    map_location=device,
    weights_only=False,
)

model.load_state_dict(best_checkpoint["model_state_dict"])

test_loss = estimate_loss(
    split="test",
    num_batches=200,
)

test_perplexity = math.exp(test_loss)

print(
    f"Final test loss: {test_loss:.4f} | "
    f"Final test perplexity: {test_perplexity:.2f}"
)

wandb.log(
    {
        "test/loss": test_loss,
        "test/perplexity": test_perplexity,
    }
)

save_checkpoint(
    path=CONFIG["final_path"],
    iteration=best_checkpoint["iteration"],
    best_val_loss=best_val_loss,
)

wandb.finish()