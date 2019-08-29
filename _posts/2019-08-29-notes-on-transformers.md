---
Title: Notes on Transformers
Date: 2019-08-29
Author: Robert Hatem
Lang: en
Tags:
Image: images/life_expectancy.png
mathjax: true
---

# Notes
I took these notes while reading the excellent tutorial [_Transformers from Scratch_].(http://peterbloem.nl/blog/transformers) (August 18, 2019)
## Self-attention
A sequence-to-sequence operation that produces the output vector $\bf{y}_i$ as a weighted average of all the input vectors:
 
$$ \bf{y}_i = \sum_i \bf{w}_{ij}\bf{x}_i  $$
  
The weight is a function over $\bf{x}_i$ and $\bf{x}_j$:

$$ \bf{w'}_{ij} = \bf{x}_i^T \bf{x}_j $$

Apply softmax to squeeze the weights to the range [0,1]:
$$ \bf{w}_{ij} = \frac{\exp{w'}_{ij}}{\sum_{j}\exp{w'}_{ij}} $$

Self-attention propagates information _between_ vectors. Every other operation is applied to input vectors in isolation, without interaction between vectors.

Say we have a sequence of words. To apply self-attention, we assign each word $t$ in our vocab an embedding vector $v_t$. It turns the word sequence

$$ the, cat, walks, on, the , street$$

into the vector sequence 

$$ v_{the}, v_{cat}, v_{walks}, v_{on}, v_{the}, v_{street} $$

If we feed this into a self-attention layer, it outputs another sequence of vectors:

$$ y_{the}, y_{cat}, y_{walks}, y_{on}, y_{the}, y_{street} $$

where $y_{cat}$ is a weighted sum over all embedding vectors in the first sequence, weighted by their (normalized) dot-product with $v_{cat}$.

The dot product expresses how related two vectors in the input sequence are, with "related defined by the learning task, and the otput vectors are weighted sums over the whole input sequence, with the weights determined by these dot products.

Note that:
  * there are not parameters yet. Self-attention takes similarity between existing vectors.
  * Self attention sees its input as a set (all at once), not a sequence. Self-attention ignores sequential nature of the text input.
  
## In Pytorch: basic self-attention
### Additional tricks
1. Queries, keys, and values
2. Scaling the dot product
3. Multi-head attention

# Buidling transformers
A transformer is:
> Any architecture designed to process a connected set of units - such as the tokens in a sequence or the pixels in an image - wehere the only interaction between units is through self-attention.

## The transformer block
Most transformer blocks are structured roughly like this:
  1. Input
  2. self-attention layer
  3. layer normalization
  4. feed forward layer (a single MLP applied independently to each vector)
  5. another layer normalization
Residual connections are added around both, before the normalization.

# Classification transformer
* Task:
  * _sequence classification.
* Dataset:
  * IMDb sentiment classiment dateset (pos/neg)
* We must build a classifier out of sequence-to-sequence layerss:
    * Apply a global average pooling to the final output sequence, then project down to a vector with one element per class, then apply softmax to produce probabilities.

## Input: using the positions
* So far the network with permutation invariant - the order of the words doesn't affect the output. We want our model to use the word ordering in its predictions.
  * Solution: create a second vector of equal length, that represents the position of the word in the current sentence, and add this to the word embedding.
    1. __position embeddings__: simply embed the positions like we did the words.
    2. __position encodings__: instead, we don't learn the position vectors, we just choose some function to map the real positions to real valued vectors, and let the network figure out how to interpret these encodings.
  * We choose position embeddings in this implementation for simplicity.
    
# Text generation Transformer
This section now generate text instead of classifying text.

# Historical baggage
## Why is it called self-attention?
* Self-attention means instead of feeding the output sequence on the previous layer directly to the input of the next, an intermediate mechanism was introduced, that decided which elements of the input were relevant for a particular word of the output.

## The original transformer: encoders and decoders
Previously, people used encoder-decoder architectures. The _encoder_ takes the input sequence and map it to a single _letent_ vector representating the whole sequence. This vector is then passed to a _decoder_ which unpakcs it to the desired target sequence (for instance, the same sentence in another language).

In later transformers, like BERT and GPT-2, the encoder/decoder configuration was entirely dispensed with. This is sometimes called a decoder/encoder-only transformer.

# Modern transformers
* BERT - Simple stack of transformer blocks, pre-trained on a large general domain corpus from English books.
  * Masking - 
  * Next sequence classification
* GPT-2
  * This was the model by OpenAI not released to public since it could generate convincing fake news.
* Transformer-XL
  * Addresses the size of the dot-product matrix, which grows quadratically in the sequence length, and quickly becomes a bottleneck.
* Sparse transformers

# Going big

