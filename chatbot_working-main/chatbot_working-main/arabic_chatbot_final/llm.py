# llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

_MODEL_CACHE = {}


def load_model(model_name):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    _MODEL_CACHE[model_name] = gen_pipeline
    return gen_pipeline


def generate_answer(model_name, question, context):
    if not context or not context.strip():
        return "عذرًا، لم أجد معلومات متعلقة بسؤالك."
    gen_pipeline = load_model(model_name)
    eos_id = getattr(gen_pipeline.tokenizer, "eos_token_id", None)

    prompt = (
        "المعلومات التالية مأخوذة من مصادر موثوقة:\n"
        f"{context}\n\n"
        "أجب عن السؤال التالي بالعربية الفصحى وباختصار شديد (جملة أو جملتان كحد أقصى)، "
        "واعتمد فقط على المعلومات أعلاه. إن لم تجد الإجابة فأجب: لا أعرف.\n"
        f"السؤال: {question}\n"
        "الإجابة المختصرة:"
    )

    results = gen_pipeline(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        return_full_text=False,
        repetition_penalty=1.1,
    )
    answer = results[0]["generated_text"].strip()
    if len(answer) > 200:
        answer = answer[:200].rstrip() + "…"
    return answer
