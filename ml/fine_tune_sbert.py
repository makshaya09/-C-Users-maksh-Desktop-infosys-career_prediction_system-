"""
Sentence-BERT Fine-Tuning Pipeline for Career Domain Embeddings.
Prepares positive/negative pairs from job descriptions and career profiles.
Provides automated fallback to pretrained all-MiniLM-L6-v2 when training pairs are below threshold.
Milestone 2: Advanced ML & Recommendation Engine.
"""

import os
import sys
import json
from typing import List, Tuple, Dict, Any

# Set path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.config import SBERT_MODEL_DIR, SBERT_MODEL_NAME, DATA_DIR
from ml.skill_embeddings import CAREER_DESCRIPTIONS


def prepare_training_pairs(dataset_path: str = None) -> List[Tuple[str, str, float]]:
    """
    Constructs semantic similarity training pairs (Text1, Text2, Score)
    from career dataset profiles and canonical job descriptions.
    """
    if dataset_path is None:
        dataset_path = os.path.join(DATA_DIR, "career_dataset.csv")

    pairs = []
    if not os.path.exists(dataset_path):
        return pairs

    import pandas as pd
    df = pd.read_csv(dataset_path).dropna(subset=["skills", "career"])

    # Positive pairs: user skills mapped to their career description (score 1.0)
    for _, row in df.iterrows():
        skills = str(row["skills"])
        career = str(row["career"])
        if career in CAREER_DESCRIPTIONS:
            pairs.append((skills, CAREER_DESCRIPTIONS[career], 1.0))

    # Hard negative pairs: skills paired with a distinct career description (score 0.0)
    careers = list(CAREER_DESCRIPTIONS.keys())
    for i, (_, row) in enumerate(df.iterrows()):
        skills = str(row["skills"])
        career = str(row["career"])
        # Pick another career
        other_career = careers[(careers.index(career) + 1 + (i % (len(careers) - 1))) % len(careers)]
        pairs.append((skills, CAREER_DESCRIPTIONS[other_career], 0.0))

    return pairs


def fine_tune_sbert(
    output_dir: str = None,
    min_pairs_required: int = 100,
    epochs: int = 3,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Fine-tunes Sentence-BERT if sufficient training pairs exist.
    If pairs are insufficient or environment does not support torch training,
    transparently falls back to the pretrained all-MiniLM-L6-v2 model and logs status.
    """
    if output_dir is None:
        output_dir = SBERT_MODEL_DIR

    os.makedirs(output_dir, exist_ok=True)
    status_file = os.path.join(output_dir, "sbert_status.json")

    print("[SBERT] Checking training corpus and dependencies for fine-tuning...")
    pairs = prepare_training_pairs()

    if len(pairs) < min_pairs_required:
        msg = (
            f"Corpus size ({len(pairs)} pairs) is below recommended threshold ({min_pairs_required}). "
            f"Using pretrained {SBERT_MODEL_NAME}."
        )
        print(f"[SBERT] {msg}")
        result = {
            "status": "using_pretrained",
            "model_name": SBERT_MODEL_NAME,
            "training_pairs_count": len(pairs),
            "reason": "Dataset size insufficient for domain fine-tuning without overfitting; utilizing robust pretrained model."
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        return result

    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        print(f"[SBERT] Preparing {len(pairs)} training examples...")
        train_examples = [
            InputExample(texts=[t1, t2], label=float(label))
            for t1, t2, label in pairs
        ]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        model = SentenceTransformer(SBERT_MODEL_NAME)
        train_loss = losses.CosineSimilarityLoss(model=model)

        print(f"[SBERT] Fine-tuning {SBERT_MODEL_NAME} for {epochs} epochs...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            show_progress_bar=True
        )

        model.save(output_dir)
        print(f"[SBERT] Fine-tuned model saved to: {output_dir}")

        result = {
            "status": "fine_tuned",
            "model_name": SBERT_MODEL_NAME,
            "output_dir": output_dir,
            "training_pairs_count": len(pairs),
            "epochs": epochs,
            "loss": "CosineSimilarityLoss"
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        return result

    except Exception as e:
        print(f"[SBERT] Fine-tuning note: {e}. Falling back to pretrained {SBERT_MODEL_NAME}.")
        result = {
            "status": "using_pretrained_fallback",
            "model_name": SBERT_MODEL_NAME,
            "training_pairs_count": len(pairs),
            "error_note": str(e),
            "reason": "Fine-tuning environment fallback; using pretrained model."
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        return result


if __name__ == "__main__":
    fine_tune_sbert()
