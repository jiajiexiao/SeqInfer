# SeqLearn

SeqLearn is a Python package designed for learning from sequences. 

Initially focused on biological sequences such as DNA, RNA, and protein sequences, it provides
essential tools and algorithms for handling sequence data. However, the package is designed to be
easily expandable to accommodate other types of sequences, such as SMILE strings or time series.
Relevant helper modules may be added in the future development. 

## Table of Contents

-   [Installation](README.md#installation)
-   [Usage](README.md#usage)
-   [Project Structure](README.md#project-structure)
-   [Examples](README.md#examples)
-   [Contributing](README.md#contributing)
-   [License](README.md#license)

## Installation


1. Clone seqlearn repository
```
git clone https://github.com/jiajiexiao/seqlearn.git
cd seqlearn/
```

2. Install in development mode
### Install in “develop” or “editable” mode:
```
python setup.py develop
```
or
```
pip install -e ./
```


<!-- You can install SeqLearn using pip:

`pip install seqlearn`  -->

## Usage

To use SeqLearn, simply import the desired modules from the `seqs` and `learners` sub-packages.

For example, you can prepare the data as below: 
```python
from seqlearn.seqs.datasets import SeqFromFileDataset
from seqlearn.seqs.transforms import KmerTokenizer, OneHotEncoder, ToTensor
from seqlearn.seqs.vocabularies import unambiguous_dna_vocabulary_dict

seq_dataset = SeqFromFileDataset(
    seq_file="examples/aptamers/pos.fasta",
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

The SeqLearn package is organized into two major parts:

1.  `seqs`: Contains modules to define and manage the data/dataset of sequences and provides various
    related transformation operations.
2.  `learners`: Contains modules for different learners (learning algorithms) to conduct learning
    tasks such as classification, regression, self-supervised representation learning, sequence
    generation, etc.

## Examples

The `examples` folder contains illustrative examples demonstrating the usage of SeqLearn for various
tasks, including classification, regression, multitask learning, etc. Each example includes a README
to guide you through the usage and expected results.

## Contributing

We welcome contributions to improve and extend SeqLearn. If you would like to contribute, please
follow our [contribution guidelines](CONTRIBUTING.md) (To be added).

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

----------

We hope you find SeqLearn useful for your sequence learning tasks! If you encounter any issues or
have suggestions for improvement, please feel free to open an issue or submit a pull request. Happy
coding!