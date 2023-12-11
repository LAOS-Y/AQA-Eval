from copy import deepcopy

from aqa.configs import Config

FASTCHAT_MODEL_CONFIG = Config(
    NAME="FastChatModel",
    DEVICE="cuda",
    NUM_GPUS=1,
    MAX_GPU_MEMORY=None,
    DTYPE=None,
    LOAD_8BIT=False,
    CPU_OFFLOADING=False,
    CONV_TEMPLATE=None,
    CONV_SYSTEM_MSG=None,
    TEMPERATURE=0.0,
    REPETITION_PENALTY=1.0,
    MAX_NEW_TOKENS=512,
    EXLLAMA_CONFIG=None,
    XFT_CONFIG=None,
    REVISION="main",
    JUDGE_SENT_END=False,
)

VICUNA_V15_7B_16K_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
VICUNA_V15_7B_16K_CONFIG.update(MODEL_PATH="lmsys/vicuna-7b-v1.5-16k")

VICUNA_V15_13B_16K_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
VICUNA_V15_13B_16K_CONFIG.update(MODEL_PATH="lmsys/vicuna-13b-v1.5-16k")

VICUNA_V13_33B_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
VICUNA_V13_33B_CONFIG.update(MODEL_PATH="lmsys/vicuna-33b-v1.3")

LLAMA2_7B_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
LLAMA2_7B_CONFIG.update(MODEL_PATH="meta-llama/Llama-2-7b-hf")

LLAMA2_7B_CHAT_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
LLAMA2_7B_CHAT_CONFIG.update(MODEL_PATH="meta-llama/Llama-2-7b-chat")

LLAMA2_13B_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
LLAMA2_13B_CONFIG.update(MODEL_PATH="meta-llama/Llama-2-13b-hf")

LLAMA2_13B_CHAT_CONFIG = deepcopy(FASTCHAT_MODEL_CONFIG)
LLAMA2_13B_CHAT_CONFIG.update(MODEL_PATH="meta-llama/Llama-2-13b-chat")

LLAMA2_70B = deepcopy(FASTCHAT_MODEL_CONFIG)
LLAMA2_70B.update(MODEL_PATH="meta-llama/Llama-2-70b-hf")

LLAMA2_70B_CHAT = deepcopy(FASTCHAT_MODEL_CONFIG)
LLAMA2_70B_CHAT.update(MODEL_PATH="meta-llama/Llama-2-70b-chat")

__all__ = [k for k in globals().keys() if "_CONFIG" in k]