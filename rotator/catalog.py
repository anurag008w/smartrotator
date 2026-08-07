"""
catalog.py — Popular/latest models per provider.

Dashboard ke Model Manager me "add model" picker ke liye — admin yahan se
check karke select karta hai (config.yaml edit kiye bina). Server kisi bhi
model id ko as-is forward karta hai, isliye list accurate-na-bhi ho toh bhi
kuch tootega nahi — sirf UI convenience hai.

Naye models jab nikle, yahan add kar do. Also custom model id type karke
bhi add ho sakta hai (dashboard me free-text option hai).
"""

MODEL_CATALOG: dict[str, list[str]] = {
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro-exp-03-25",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen-3-30b-a3b",
    ],
    "openrouter": [
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "google/gemini-2.5-flash:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "openai/gpt-4o-mini:free",
        "qwen/qwen-3-235b-a22b:free",
    ],
    "nvidia": [
        # flagship chat LLMs (build.nvidia.com hosted, OpenAI-compatible)
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.2-3b-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "minimaxai/minimax-m3",
        "mistralai/mistral-large",
        "mistralai/mistral-large-2-instruct",
        "mistralai/mixtral-8x22b-v0.1",
        "ai21labs/jamba-1.5-large-instruct",
        "ibm/granite-3.0-8b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "mistralai/codestral-22b-instruct-v0.1",
        # NVIDIA ke apne (Nemotron family)
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-nano-3-30b-a3b",
        # VISION support (chat + image input)
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        # fast / small
        "google/gemma-3-12b-it",
        "google/gemma-4-31b-it",
        "stepfun-ai/step-3.7-flash",
    ],
    "zen": [
        "opencode/big-pickle",
        "opencode/deepseek-v4-flash-free",
    ],
}
