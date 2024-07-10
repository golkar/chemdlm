import os
import json
import typing
from transformers import PreTrainedTokenizer, AutoTokenizer
import selfies as sf


class SELFIESTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file=None,
        bos_token='[BOS]',
        eos_token='[EOS]',
        sep_token='[SEP]',
        cls_token='[CLS]',
        pad_token='[PAD]',
        mask_token='[MASK]',
        unk_token='[UNK]',
        **kwargs
    ):
        self.vocab = {}
        self.ids_to_tokens = {}
        
        special_tokens = [pad_token, unk_token, cls_token, sep_token, mask_token, bos_token, eos_token]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ids_to_tokens[i] = token
        
        if vocab_file is not None:
            self.load_vocabulary(vocab_file)
        else:
            self._init_vocabulary()
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs
        )

    def _init_vocabulary(self):
        alphabet = list(sf.get_semantic_robust_alphabet()) + ['[nop]']
        for token in alphabet:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.ids_to_tokens[idx] = token

    def load_vocabulary(self, vocab_file):
        with open(vocab_file, 'r') as f:
            loaded_vocab = json.load(f)
        self.vocab.update(loaded_vocab)
        self.ids_to_tokens.update({v: k for k, v in loaded_vocab.items()})

    def _tokenize(self, text: str) -> typing.List[str]:
        return sf.split_selfies(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

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
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token
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
