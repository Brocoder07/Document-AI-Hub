from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.core.config import settings

_tokenizer = None
_model = None
_generator = None

def get_generator():
    global _tokenizer, _model, _generator

    if _generator is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_GEN_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(settings.TEXT_GEN_MODEL)
        _generator = pipeline(
            "text-generation",
            model=_model,
            tokenizer=_tokenizer,
            device=-1
        )
    return _generator

def generate_answer(prompt: str):
    gen = get_generator()
    output = gen(
        prompt,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.5
    )
    return output[0]["generated_text"]
