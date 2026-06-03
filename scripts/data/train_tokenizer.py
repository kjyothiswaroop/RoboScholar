import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).parent.parent.parent
PAPERS_PATH = ROOT / "data" / "raw" / "papers.jsonl"
TOKENIZER_DIR = ROOT / "data" / "tokenizer"
VOCAB_SIZE = 32000


def iter_texts():
    with open(PAPERS_PATH) as f:
        for line in f:
            p = json.loads(line)
            yield p["title"] + "\n" + p["full_text"]


def main():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iter_texts(),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
    )
    tokenizer.save_model(str(TOKENIZER_DIR))
    print(f"Tokenizer saved to {TOKENIZER_DIR}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Quick sanity check
    sample = "The robot uses SLAM to localize and build a map of its environment."
    encoded = tokenizer.encode(sample)
    print(f"\nSample: {sample}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs:    {encoded.ids}")


if __name__ == "__main__":
    main()
