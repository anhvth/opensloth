# export_fp8.py
import argparse
from unsloth import FastLanguageModel
from peft import PeftModel
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def main(base, lora, outdir):
    model, tokenizer = FastLanguageModel.from_pretrained(base,
        load_in_4bit=False, load_in_8bit=False, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora)
    model = model.merge_and_unload()
    
    oneshot(model=model,
        recipe=QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"]
        )
    )
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"Saved FP8 model to {outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("lora")
    p.add_argument("outdir")
    args = p.parse_args()
    main(args.base, args.lora, args.outdir)