# Module 2: Text Classification

## About the Module

### Objective: 

In this module, you will implement and fine-tune several models for a text classification task.  
You will learn how to clean text, train classic ML models, fine-tune BERT, and write some elements of the Transformer architecture from scratch.

### Task:

You are going to implement classical and NN-based models for the text classification task.  
There are 5 steps to complete:
1. **Data Preparation**

2. **Train Classical Models**

3. **Fine-Tune BERT**

4. **Implement Attention from Scratch**

5. **Compare Results**

### Prerequisites
- PyTorch

---

## Tasks Description

### Tasks:

All classifiers are inherited from [TextClassifier](src/base.py).   
**You can change arguments to the `train` method in your model classes as needed**. The same applies to initialization and predict.

Use the [training](scripts/train.py) and [inference](scripts/inference.py) scripts to run your models. See examples of [training configuration](scripts/config_train.json) and [inference configuration](scripts/config_inference.json).

**!Important:** Treat this task as a binary classification task, requiring just two labels from the dataset: `abstract` or `paragraph`. In practice, you might perform multi-classification, for instance, by adding an additional class `other`.

**Data**: use `train` and `test` from the provided dataset_doc.zip. Put `train` and `test` folders under [../dataset_doc](../dataset_doc/).

#### 1. Parse Dataset
- Modify the [src/data_extractor.py](src/data_extractor.py) script to generate `train` and `test` CSV files for your models.
- Once implemented, you can use [scripts/extract_data.py](scripts/extract_data.py) to run the parsing and create CSV files.

#### 2. Implement Classical Classifiers
- Implement basic text cleaning and preprocessing in [src/classical/preprocessor](src/classical/preprocessor.py).
- Implement additional feature generators in [src/classical/feature_generator.py](src/classical/feature_generator.py).
  - For example, add box-based features.
- Implement a [logistic regression classifier](src/classical/log_reg_classifier.py):
  - Add a train-validation split.
  - Perform hyperparameter tuning via grid-search and cross-validation.
- Complete missing methods in the [XGB classifier](src/classical/xgb_classifier.py):
  - Add a train-validation split.
  - Conduct hyperparameter tuning.
  - Implement early stopping based on validation score.

#### 3. Fine-Tune BERT
- Implement a BERT Text Classifier in [src/nn/bert_classifier.py](src/nn/bert_classifier.py).
  - Fill in the missing code.
  - Experiment with optimization parameters (e.g., learning rate).
  - Implement early stopping.
  - In case compute is not provided and not available for you, please use Colab

#### 4. Implement Transformer Model
- Implement the `attention` method in `MultiHeadAttentionBlock` in [src/nn/transformer_blocks.py](src/nn/transformer_blocks.py) and train your own transformer model.

#### 5. Compare Results
In a professional setting, you would use [MLflow](https://mlflow.org/docs/latest/tracking.html) or a similar tool to track your experiments and share results with the team. Here, you can dump your metrics to disk in .csv or .json and compare them manually. Feel free to adapt the inference script to log the model name, parameters, etc.  
**Please provide your results along with model artifacts to your mentor!**

The metrics to compare your models are precision, recall, and F1-score for the `abstract` label.

---

## Definition of Done
This outlines the criteria for a completed task:
- All models are trained on the train set and validated on the test set of the dataset.
- Your best precision, recall, and F1 scores for the test set are reported along with model artifacts.
- All tests are passing.

---

## Recommended Materials

After reading these, you should be able to answer and elaborate on the following questions:
- Describe the structure of an attention block and an attention head: what matrices does it have, what operations are performed (what is being multiplied by what, etc.)? What is its computational complexity?
- What are the main preprocessing steps for text data? Are they always appropriate and required?
- How do the Bag of Words and TF-IDF approaches differ?
- How are the Bag of Words and TF-IDF approaches different from the transformer-based approach in terms of the level of text understanding?
- How would you consider the choice of approach/model if you had a text classification problem on your project? What factors would you take into consideration when moving from simple to more complicated approaches? Give examples.
- What would happen if you change tokenizer for a pre-trained model?

### Preprocessing

- [Theory of Text Preprocessing](https://web.stanford.edu/~jurafsky/slp3/2.pdf) -- Learn about regular expressions, basic normalization techniques, BytePairEncoding, and other tokenization techniques. After reading this, you should be able to answer what the difference between lemmatization and stemming is, and whether we need them if using something like BPE.
- [Summary of Tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) -- After reading this, you should be able to explain how BPE is different from WordPiece.
- [Preprocessing with NLTK](https://www.nltk.org/book/ch03.html) -- Regular expressions, tokenization, normalization with a focus on NLTK. This is a practical reference.
- [Python regex HOWTO](https://docs.python.org/3/howto/regex.html) -- Official Python guide on regular expressions, keep it in your bookmarks, this is a practical reference.

### Classical ML

- [Vector Semantics Theory](https://web.stanford.edu/~jurafsky/slp3/6.pdf) -- After reading this, you should be able to answer questions like what embeddings are (TF-IDF, Word2Vec), how to measure the distance between vectors, and the difference between static and contextual embeddings.
- [TF-IDF sklearn example](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) -- See a practical example of building a classifier based on TF-IDF features.
- [Text classification with XGBoost](https://www.kaggle.com/code/diveki/classification-with-nlp-xgboost-and-pipelines) -- A practical example with an XGBoost text classifier along with sklearn Pipelines and hyper-parameter tuning.
- [Pretrained embeddings with NLTK & Gensim](https://www.nltk.org/howto/gensim.html#using-the-pre-trained-model) -- Practical guide on how to train or use static word vectors.

### Transformers
- [Theory of Transformers](https://web.stanford.edu/~jurafsky/slp3/10.pdf) -- Mathematical description of transformers. Please skip LLM-related materials for now.
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) -- The paper where transformers were introduced.
- [Transformers Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) -- Math behind transformers, attention variations, positional embeddings variations. After reading this, you should be able to answer questions about the time and memory complexity of transformers.
- [How Do Transformers Work?](https://huggingface.co/learn/nlp-course/chapter1/4) -- High-level material on how transformers work.
- [BERT 101](https://huggingface.co/blog/bert-101) -- Theoretical material on BERT architecture with examples using transformers.
- [Multi-Headed Attention Implementation](https://nn.labml.ai/transformers/mha.html) -- Check the implementation of multi-head attention.

