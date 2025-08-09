#!/usr/bin/env python3
"""
Test the final GRPO model with a sample math problem

This script demonstrates how to use the final trained model for inference.
Run this after completing all 4 training steps.

Note: This script is for inference only and uses unsloth directly.
For training, always use the OpenSloth CLI tools (os-sft, os-grpo) which
handle proper import isolation and multi-GPU coordination.
"""

import sys
import os
import torch

def _import_unsloth_local():
    # Lazy import to avoid GPU registry until actually needed
    from unsloth import FastLanguageModel  # type: ignore
    return FastLanguageModel

def test_model():
    """Test the final GRPO model with a sample math problem"""
    
    # Check if model exists
    model_path = "outputs/grpo_final_model"
    if not os.path.exists(model_path):
        print("❌ Error: Final model not found at outputs/grpo_final_model")
        print("Please complete all training steps first:")
        print("1. bash grpo_tutorial/scripts/00_prepare_sft_dataset.sh")
        print("2. bash grpo_tutorial/scripts/01_train_sft.sh")
        print("3. bash grpo_tutorial/scripts/02_prepare_grpo_dataset.sh")
        print("4. bash grpo_tutorial/scripts/03_train_grpo.sh")
        return False
    
    print("🔄 Loading the final GRPO model...")
    
    try:
        # Load the model
        FastLanguageModel = _import_unsloth_local()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Set up the chat template
        system_prompt = "You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>"
        reasoning_start = "<start_working_out>"
        
        chat_template_str = (
            "{% if messages[0]['role'] == 'system' %}"
                "{{ messages[0]['content'] + eos_token }}"
            "{% else %}"
                "{{ '" + system_prompt + "' + eos_token }}"
            "{% endif %}"
            "{% for message in messages[1:] %}"
                "{% if message['role'] == 'user' %}"
                    "{{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
        )
        
        tokenizer.chat_template = chat_template_str
        
        # Test problems
        test_problems = [
            "What is the square root of 144?",
            "Solve for x: 2x + 5 = 13",
            "If a triangle has sides of length 3, 4, and 5, what is its area?",
        ]
        
        print("✅ Model loaded successfully!")
        print("\n🧮 Testing the model with sample math problems...")
        print("=" * 60)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n📝 Test Problem {i}: {problem}")
            print("-" * 40)
            
            # Prepare the prompt
            messages = [
                {"role": "user", "content": problem},
            ]
            
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part
            generated_text = response[len(input_text):].strip()
            
            print("🤖 Model Response:")
            print(generated_text)
            print()
        
        print("🎉 Testing completed successfully!")
        print("\n💡 Tips:")
        print("- The model should now use proper reasoning format with <start_working_out> tags")
        print("- Answers should be enclosed in <SOLUTION> tags") 
        print("- The reasoning quality should be improved compared to the base model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model()
