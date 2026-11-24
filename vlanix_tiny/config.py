"""Independent Tiny preset; upstream defaults remain unchanged."""
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

EXPECTED_PARAMS = 6066176

def make_config():
    return MiniMindConfig(hidden_size=256, num_hidden_layers=6,
        use_moe=False, vocab_size=4096, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=832,
        max_position_embeddings=512, tie_word_embeddings=True,
        dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0)

def make_model():
    return MiniMindForCausalLM(make_config())
