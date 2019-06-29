---
Title: Transfer Learning in NLP
Date: 2019-06-24
Author: Robert Hatem
Lang: en
Tags:
Image: images/life_expectancy.png
mathjax: true
---

### Language models have replaced word embeddings
Improving embeddings has driven advances in benchmark tasks in NLP for a while:
* Pretrained word embeddings on large amounts of unlabled data (word2vec, GLoVe).
* These pre-trained embeddings are fed in to the first layer of a neural networks, then trained further on a particular task.
* Pre-trained embeddings capture low-level information but miss more detailed, higher-level representations.
* Also, they still need to be trained for specific tasks.

The paradigm is shifting.
* Going from initializing not just the first layer of our models to pretraining the entire model with hierarchical representations.
* Pretraining entire models has been practice by the computer vision community for years, often done my learning to classify images on the ImageNet dataset.
* ULMFiT, ELMo, and OpenAI transformer allow the models to learn higher-level nuances of language, and be closer to having "ImageNet for language."

Which benchmark tasks should  universal language modeling should be good?
* Reading comprehension: read a paragraph, answer simple question about it.
* Natural language inference: Two sentences, classify them as contradiction, entailment, or neutral.
* Machine translation: translating from one lanuage to another.
* Constituency parsing: extract the syntactic structure of a sentence in the form of a (linearized) constituency parse tree.
* Langauge modeling: predic the next work given its previous word (as a conditional probability over words).

Which task is most representative of the space of NLP problems? Which task is most representative of overall understanding of natural language?
* __Language modeling__, because it needs to correctly do syntax, semantics, and common sense.
* Therefore, we should pretrain language models. The idea was first proposed in 2015, but recently it has shown great empirical results (ELMo, ULMFiT, OpenAI Transformer), achieving state-of-the-art results on a wide range of tasks.
  * RH: should we try other pretraining tasks and see how well they do? Language modeling seems to have done well but others could too.

How are these pretrained language models incorporated into downstream models? How does the knowledge transfer?
1. use the pretrained language model as a fixed feature extractor and feed the features into a model (EMLo).
2. Fine-tune the entire langauge model (ULMFiT).


### References
* [The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a), Thomas Wolf, May 14, 2018
* [NLP's ImageNet moment has arrived](http://ruder.io/nlp-imagenet/), Sebastian Ruder, July 12, 2018.

### Appendix
Notes on embeddings.

##### Word Embeddings
* word2vec
* GLoVe
* FastText
* ELMo

##### Sentence Embeddings
* Unsupervised
  * bag-of-words
  * skip-thought vectors
  * quick-thought vectors
* Supervised
  * InferSent
* Multi-task learning
  * General Purpose Sentence Representation (MILA/MSR)
  * Universal Sentence Encoder (Google)

Some of the most popular benchmarks tasks are shown below.

##### Benchmark Tasks
* Word embeddings: many
* Sentence embeddings
  * SentEval - 17 common tasks and 10 probing tasks
  * GLUE/SuperGLUE: several tasks combined for universal embeddings
* Document embeddings
  * Dataset of Wikiedia articles, split by category (e.g. sports or histroy) and visualized using T-SNE.
  * Cosine similarity to nearby documents (Wiki articles). E.g. Beyonce article is near Rihanna article.

