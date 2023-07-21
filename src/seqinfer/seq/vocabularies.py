from enum import Enum

from Bio.Data import IUPACData


class SpecialToken(str, Enum):  # strEnum is not available until py3.11
    """Enum class for special tokens"""

    UNK = "<unk>"  # Unknown token
    CLS = "<cls>"  # Classification token
    PAD = "<pad>"  # Padding token
    MASK = "<mask>"  # Masking token
    EOS = "<eos>"  # End-of-sequence token


def build_vocabulary_dict(vocabulary: list[str], offset: int = 0) -> dict[str, int]:
    """helper function to build vocabulary dict from input vocabulary

    Args:
        vocabulary (list[str]): vocabulary of tokens.
        offset (int, optional):
            int added to the index. Default to 0.
    Returns:
        dict[str, int]: vocabulary dict with tokens as keys and indices as values.
    """
    return {token: i + offset for i, token in enumerate(vocabulary)}


esm_protein_letters = (
    # https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/constants.py#L8
    [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ]
)

# Nucleotides
ambiguous_dna_vocabulary_dict = build_vocabulary_dict(IUPACData.ambiguous_dna_letters)
unambiguous_dna_vocabulary_dict = build_vocabulary_dict(IUPACData.unambiguous_dna_letters)
extended_dna_vocabulary_dict = build_vocabulary_dict(IUPACData.extended_dna_letters)
ambiguous_rna_vocabulary_dict = build_vocabulary_dict(IUPACData.ambiguous_rna_letters)
unambiguous_rna_vocabulary_dict = build_vocabulary_dict(IUPACData.unambiguous_rna_letters)

# Amino acids
protein_vocabulary_dict = build_vocabulary_dict(IUPACData.protein_letters)
extended_protein_vocabulary_dict = build_vocabulary_dict(IUPACData.extended_protein_letters)
esm_protein_vocabulary_dict = build_vocabulary_dict(esm_protein_letters)
