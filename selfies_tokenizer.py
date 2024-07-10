import os
import json
import typing
from transformers import PreTrainedTokenizer, AutoTokenizer
import selfies as sf

class SELFIESTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file=None, **kwargs):
        self.vocab = {}
        self.ids_to_tokens = {}
        if vocab_file is not None:
            self.load_vocabulary(vocab_file)
        else:
            self._init_vocabulary()
        super().__init__(**kwargs)

    def _init_vocabulary(self):
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[nop]']
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ids_to_tokens[i] = token
        
        alphabet = sf.get_semantic_robust_alphabet()
        for token in alphabet:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.ids_to_tokens[idx] = token

    def load_vocabulary(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text: str) -> typing.List[str]:
        return sf.split_selfies(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, '[UNK]')

    def convert_tokens_to_string(self, tokens: typing.List[str]) -> str:
        return ''.join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self.vocab.copy()

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        
        # Save tokenizer configuration
        config = {
            "model_type": "selfies",
            "tokenizer_class": "SELFIESTokenizer",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]"
            }
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
        
        return (vocab_file, config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = cls(vocab_file=os.path.join(pretrained_model_name_or_path, "vocab.json"), **kwargs)
        return tokenizer

if __name__ == "__main__":
    # Example SELFIES string
    example_selfies = "[C][=C][O][C][nop][nop][MASK]"

    # Initialize the tokenizer
    tokenizer = SELFIESTokenizer()

    # Tokenize the SELFIES
    tokens = tokenizer.tokenize(example_selfies)
    print(f"Tokens: {tokens}")

    # Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")

    # Convert IDs back to tokens
    recovered_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"Recovered tokens: {recovered_tokens}")

    # Convert tokens back to string
    recovered_string = tokenizer.convert_tokens_to_string(recovered_tokens)
    print(f"Recovered SELFIES: {recovered_string}")
    print(f"Recovered SELFIES matches original: {recovered_string == example_selfies}")

    # Save the tokenizer
    save_dir = "./selfies_tokenizer"
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")

    # Load the tokenizer using AutoTokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    print(f"Loaded tokenizer vocab size: {loaded_tokenizer.vocab_size}")

    # Test the loaded tokenizer
    loaded_tokens = loaded_tokenizer.tokenize(example_selfies)
    print(f"Tokens from loaded tokenizer: {loaded_tokens}")