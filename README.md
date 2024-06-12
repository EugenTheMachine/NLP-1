# nlp-course

Hi, welcome to the course on Natural Language Processing.  
This repo is part of the corresponding Mentoring Program.

## Prerequisites

* Python
* [Pytorch](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
* [git](https://www.atlassian.com/git/glossary#commands)

## Objectives of the Course

**This course IS** a practice-focused course with the following objectives:
* Giving you hands-on experience with the most common tasks: NER, Classification, Search, introduction to LLMs.
* Getting acquainted with the most commonly used tools like `transformers`.
* Doing that in a more realistic setup -- with realistic data and a requirement to write code outside of notebooks.

**This course IS NOT** a fundamental, university-like course on NLP.  
There are many resources more suitable for a theoretical deep dive:
* [Stanford CS224N](https://web.stanford.edu/class/cs224n/), [Stanford CS224N Playlist](https://youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&si=tbv4KybgAxGLH8si)
* [Stanford CS25](https://web.stanford.edu/class/cs25/), [Stanford CS25 Playlist](https://youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&si=PP1ijpXoGDYNKvq1)
* [Speech and Language Processing by Dan Jurafsky & James Martin](https://web.stanford.edu/~jurafsky/slp3/)

Feel free to use them as a reference throughout the course and generally at your work.

## Structure

The course consists of 5 modules:  
1. [OCR and Labeling](Module_1_Intro/README.md) - Set up an annotation pipeline for unstructured PDFs.
2. [Classification](Module_2_Classification/README.md) - Create and evaluate a classification model, from a classical approach to transformers.
3. [Named Entity Recognition](Module_3_NER/readme.md) - Apply transformers to sequence labeling tasks, try multimodal architecture.
4. [Similarity Learning or Search](Module_4_Search/README.md) - Set up a vector database, try different encoders, and fine-tune your own.
5. [Intro to LLMs](Module_5_LLM/README.md) - Get acquainted with modern Large Language Models, both OpenAI and Open Source.

Each module is about a week or two to complete.

## Data

In case of any technical issues, please reach out to your mentor or course coordinators.

The dataset is a subset of [DocBank](https://doc-analysis.github.io/docbank-page/index.html), and you will work with it throughout the entire course.  

Use these links to download the dataset:
* [dataset_doc.zip](https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds_nlp/dataset_doc.zip)
* [dataset_doc_pdf.zip](https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds_nlp/dataset_doc_pdf.zip)

If the provided subset is not sufficient for your experiments, discuss with your mentor the usage of the rest of the dataset.

## Feedback & Contribution

If you encounter a problem with this repo, please create an issue here, on GitLab.  
Do not hesitate to reach out to your mentor as well.

Feel free to reach out to the following people on any issues and suggestions:
* vladimir_ageev@epam.com
* anton_guldinskii@epam.com

Core Contributors:

* vadim_radchenko@epam.com
* ali_oztas@epam.com
* aleksandr_fida@epam.com
* ekaterina_kasilina@epam.com
* jonathan_espinosa@epam.com
* elizaveta_kapitonova@epam.com
* mikhail_bulgakov@epam.com
* vladislav_belov1@epam.com

This course also includes materials from

* dmitrii_nikitko@epam.com

## Dependencies management

This course also introduces [poetry](https://python-poetry.org) - a dependency management tool.

For us, it solves two problems:
- Pinning dependencies, so the environment is easy to set up on every machine.
- Grouping dependencies for different tasks/sub-packages/experiments and so on.  
  This is useful at the experimental stage where you do not have a particular service yet but may run very different models with their own sets of dependencies.


It is recommended to use poetry within a virtual environment. You can create a virtual environment using `conda` or other virtual environment managers. After installing `poetry` with `pip install poetry`.

Please find the [pyproject.toml](../pyproject.toml) file with dependencies already set up. Executing `poetry install --no-root` in your new virtual environment should work.

- If you perform only `poetry install` without `--no-root`, you may need to fix the imports since `poetry install` creates a package for your project, affecting imports.
- You might also need to add the `Module_N_Module_Name` folder to your `PYTHONPATH` before running tests or any scripts. This can be done by running `export PYTHONPATH='/path/to/nlp-course/Module_N_Module_Name:$PYTHONPATH'`.
- You can use `poetry shell` or install your requirements within a virtual environment. It is personally recommended to create a virtual environment with `conda`:
    - `conda create -n ENV_NAME python=3.11`
    - `conda activate ENV_NAME`
    - Then, within the environment, run `poetry install --no-root` and use it this way.
    - VSCode and PyCharm support selecting conda environments as your interpreter.
    - You may need to execute `poetry install --with module_name` to install dependencies for a specific module.

Once set up, we recommend running `poetry shell` and working within the poetry-managed virtual environment.

See the [documentation on grouping](https://python-poetry.org/docs/managing-dependencies/#installing-group-dependencies) for further details on how to install particular dependencies.

## Compute environment

Most modules of this course will require fine-tuning of BERT-like models of various sizes.  
It should be feasible to pass the whole course on rather modern laptops at level of Apple M1 + 16Gb of RAM.  
However, if you face any issues with compute, please consider falling back to Colab as your environment.

[Please find instructions on using Colab here](./readme_resources/how_to_work_in_colab.md)