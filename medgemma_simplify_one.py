import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"


def build_messages(report_text: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful medical translator. "
                        "Translate the following chest X-ray sentence into simple English "
                        "that a person with secondary education and basic medical knowledge "
                        "can understand, in 1–3 short sentences. "
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Original radiology report:\n{report_text}",
                }
            ],
        },
    ]


def load_model():
    print("Loading MedGemma 4B (CPU)...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,      # safer on CPU
        device_map={"": "cpu"},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def simplify_report(model, processor, report_text: str, max_new_tokens: int = 256) -> str:
    messages = build_messages(report_text)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # ensure CPU tensors
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded.strip()


if __name__ == "__main__":
    example_report = (
        "FINDINGS: The lungs are clear. No focal consolidation, pleural effusion, "
        "or pneumothorax is seen. The cardiomediastinal silhouette is within normal limits."
    )

    model, processor = load_model()
    simplified = simplify_report(model, processor, example_report)

    print("\n=== Original report ===\n")
    print(example_report)
    print("\n=== MedGemma simplified ===\n")
    print(simplified)

