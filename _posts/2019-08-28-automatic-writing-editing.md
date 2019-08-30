---
Title: Automatic Writing Editor
Date: 2019-08-28
Author: Robert Hatem
Lang: en
Tags:
Image: images/life_expectancy.png
mathjax: true
---

These are my thoughts on an automatic writing editor, a product I would love to have. Any tips or pointers are welcome!

There are many writing and editing tips and methods than concretely improves writing. For example, it's makes it easier to read to put the old informationat the end of a sentence and the new information at the beginning.  However, a person must do the editing, by reading through the draft and checking for all the items or rules violations on the list. This takes time and effort, and it should be automated so a writer can get instant feedback as they write. This would improve writing and communication.

Some software tools help improve writing, but they are limited. Grammarly checks for grammar and spelling mistakes, checks for plagarism, checking for the passive voice, readability scorees, but not much more. There are many more writing tips and methods that could be automated:

1. Putting old information at the end of a sentence and putting new information at the beginning of a sentence.
2. Cohension (do the sentences flow from one to the other) and coherence (how logical the ideas are).
2. Overall readability - is it easy to read the essay?
3. Avoiding nominalizations - "we analyzed the data" instead of "we performed an analysis of the data."

Software could check all of these rules while the writing is writing, for realtime feedback and instant correction. Grammarly is the leader in software for improving writing but their offerings seem limited. I want more.

## How to automate these tasks

1. Start with a paragraph. Turn into chunks, or phrases. Create a classifier to classify each new chunk as new information or old information. As the person writes, the model will tell them if the new part they wrote outs new information before old information (violating the rules).
  * The main challenge is lack of a dataset. The dataset must be labeled with half paragraphs, the next chunk, and a binary indicator for whether the chunk is old or new information. If such a dataset doesn't exist, I can't train a model on it.
2. Cohension and coherence are normal NLP tasks and there should be methods to to measure them. One challendge for making them useful is if they require the whole paragraph, or document, to take measurements. It wouldn't suggest a specific edit, making it hard to improve the document without a specific recommendation. 
