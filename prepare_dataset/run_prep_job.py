import sys
import json

from prepare_qwen import QwenDatasetPreparer
from prepare_gemma import GemmaDatasetPreparer


MODEL_FAMILIES = {
    "Qwen": QwenDatasetPreparer,
    "Gemma": GemmaDatasetPreparer,
}


def main() -> None:
    cfg = json.load(sys.stdin)
    model_family = cfg.pop("model_family")
    cls = MODEL_FAMILIES[model_family]
    prep = cls()
    print(f"[JOB] Starting dataset prep for {model_family}...")
    out_dir = prep.run_with_config(cfg)
    print(f"[JOB] Completed. Saved at: {out_dir}")
    print("[JOB] If debug was enabled, see .log/dataloader_examples.html")


if __name__ == "__main__":
    main()
