import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tokenizers import ByteLevelBPETokenizer
from src.data.dataset import RoboScholarDataset
from src.model.transformer import Transformer
import wandb
import hydra
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = ByteLevelBPETokenizer(
        cfg.paths.tokenizer_vocab,
        cfg.paths.tokenizer_merges
    )

    print("Loading dataset...")
    dataset = RoboScholarDataset(cfg.paths.qa_pairs, tokenizer, cfg.model.max_len)
    val_size = int(len(dataset) * cfg.training.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size} | Val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    print("Building model...")
    model = Transformer(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        max_len=cfg.model.max_len
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    wandb.init(project=cfg.wandb.project, config=dict(cfg))

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.num_epochs):

        # training
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

            logits, _ = model(input_ids, mask)

            loss = criterion(logits.view(-1, cfg.model.vocab_size), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch+1} | step {step}/{len(train_loader)} | loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids  = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                mask       = batch["mask"].to(device)

                logits, _ = model(input_ids, mask)
                loss = criterion(logits.view(-1, cfg.model.vocab_size), target_ids.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()