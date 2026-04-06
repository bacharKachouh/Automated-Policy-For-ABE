"""
Access policy extraction using a fine-tuned GPT-2 model.

The model is trained to complete prompts of the form::

    Data Type: Medical Records
    Sensitivity: Highly Confidential
    Department: Oncology
    ...
    ### Access Policy:

and returns a human-readable policy sentence which is then parsed into an
ABE policy string by :mod:`src.abe.policy_parser`.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.config import POLICY_MODEL_PATH


class PolicyExtractor:
    """
    Load a fine-tuned GPT-2 model and generate access policy text.

    Parameters
    ----------
    model_path : str or Path, optional
        Directory containing the saved model and tokenizer.
        Defaults to ``config.POLICY_MODEL_PATH``.
    """

    _GENERATION_KWARGS = dict(
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        no_repeat_ngram_size=2,
    )

    def __init__(self, model_path=None):
        path = str(model_path or POLICY_MODEL_PATH)
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self._GENERATION_KWARGS = dict(
            self._GENERATION_KWARGS,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def extract(self, prompt_text):
        """
        Generate an access policy sentence from *prompt_text*.

        Only the first sentence after ``### Access Policy:`` is returned.

        Parameters
        ----------
        prompt_text : str
            Data attribute prompt (output of :func:`src.utils.xml_utils.format_attribute_prompt`).

        Returns
        -------
        str
            Single-sentence access policy, e.g.
            ``"This data can be accessed by users with role Doctor in Oncology with clearance level 2."``
        """
        inputs = self.tokenizer.encode(prompt_text, return_tensors="pt")
        outputs = self.model.generate(inputs, **self._GENERATION_KWARGS)
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        policy_part = generated.split("### Access Policy:")[-1].strip()
        return policy_part.split(".")[0].strip() + "."
