# export_fp8.py
import argparse
from unsloth import FastLanguageModel
from peft import PeftModel
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import llmcompressor
from packaging import version
import warnings

required_version = "0.6.0.1"
current_version = llmcompressor.__version__
if version.parse(current_version) < version.parse(required_version):
    raise RuntimeError(f"llmcompressor>={required_version} required, found {current_version}")
elif version.parse(current_version) > version.parse(required_version):
    warnings.warn(f"llmcompressor version {current_version} is newer than tested {required_version}. Proceed with caution.")

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