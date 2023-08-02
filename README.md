# SeqInfer

<!-- [![codecov](https://codecov.io/gh/username/repository/branchname/graph/badge.svg)](https://codecov.io/gh/username/repository)-->

SeqInfer is a Python package for sequence inference, enabling outcome prediction, sequence
generation, and meaningful representation discovery, etc for sequence-like data. 

Initially focused on biological sequences such as DNA, RNA, and protein sequences, it aims to provide
essential tools and algorithms for handling sequence data. However, the package is designed to be
easily expandable to accommodate other types of sequences, such as SMILE strings or time series.
Relevant helper modules may be added in the future development. 

**This library was renamed to SeqInfer from SeqLearn to avoid potential conflicts and confusion given that SeqLearn has been used by other people's repo.

## Table of Contents

-   [Installation](README.md#installation)
-   [Usage](README.md#usage)
-   [Project Structure](README.md#project-structure)
-   [Examples](README.md#examples)
-   [Contributing](README.md#contributing)
-   [License](README.md#license)

## Installation
You can install `SeqInfer` using pip:
`pip install seqinfer` 
Or 
`pip install git+https://github.com/jiajiexiao/seqinfer.git`

## Usage

To use SeqInfer, simply import the desired modules from the `seqs` and `learners` sub-packages.

For example, you can prepare the data as below: 
```python
from seqinfer.seq.datasets import SeqFromFileDataset
from seqinfer.seq.transforms import Compose, KmerTokenizer, OneHotEncoder, ToTensor
from seqinfer.seq.vocabularies import unambiguous_dna_vocabulary_dict

seq_dataset = SeqFromFileDataset(
    seq_file="examples/toys/CCA-TXXAGG-AG-TGG-TC-A-T/pos.fasta",
    seq_file_fmt="fasta",
    transform_sequences=Compose(
        [
            KmerTokenizer(
                k=1,
                stride=1,
                vocab_dict=unambiguous_dna_vocabulary_dict,
                num_output_tokens=3,
                special_tokens=None,
            ),
            OneHotEncoder(vocab_size=len(unambiguous_dna_vocabulary_dict)),
            ToTensor(),
        ]
    ),
)
```



## Project Structure

The SeqInfer package is organized into two major parts:

1.  `seq`: Contains modules to define and manage the data/dataset of sequences and provides various
    related transformation operations.
2.  `infer`: Contains modules for different learners (learning algorithms) to conduct learning
    tasks such as classification, regression, self-supervised representation learning, sequence
    generation, etc.

## Examples

The `examples` folder contains illustrative examples demonstrating the usage of SeqInfer for various
tasks, including classification, regression, multitask learning, etc. Each example includes a README
to guide you through the usage and expected results.

## Contributing

We welcome contributions to improve and extend SeqInfer. If you would like to contribute, please
follow our [contribution guidelines](CONTRIBUTING.md) (To be added).

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

----------

We hope you find SeqInfer useful for your sequence learning tasks! If you encounter any issues or
have suggestions for improvement, please feel free to open an issue or submit a pull request. Happy
coding!