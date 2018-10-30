# Utilities related to D2 Clustering for Document Data

This repository includes python 2.7 scripts that process a document dataset file into .d2s format that is ready for applying software package **d2_kmeans**. The clustering result provided by **d2_kmeans** is then evaluated by different metrics. If you are interested in the software **d2_kmeans** and reproduce the results in the paper, please contact the author @JianboYe directly. 


The utilities involved were used for generating part of the results reported in the following paper:

[Jianbo Ye](http://personal.psu.edu/jxy198), Yanran Li, Zhaohui Wu, James Z. Wang, Wenjie Li, Jia Li, Determining Gains Acquired from Word Embedding Quantitatively Using Discrete Distribution Clustering, Proceedings of The Annual Meeting of the Association for Computational Linguistics (ACL), Vancouver, Canada, July 2017. Long paper.

## Quickstart

Download sample datasets from the author's webpage. 

```
$ wget http://infolab.stanford.edu/~wangz/project/linguistics/ACL17/acl2017dataset.zip
$ unzip acl2017dataset.zip 
```

Download pre-trained wordvecs, two of which are public downloadable.

- glove_6B_300d.bin
- GoogleNews-vectors-negative300.bin
- [word2vec_400_10_10.bin](https://psu.box.com/s/bah111znok5xs6cdztddfwdc9msq33g1)

Install python (version 2.7) and its dependencies. The tested versions are

- numpy (1.9.2)
- scipy (1.9.2)
- sklearn (0.16.1)
- cvxopt (1.1.7)
- gensim (0.12.1)
- nltk (3.0.5)
- mosek (optional, 7.x)

You may need adapt the code to newer versions if needed. 

After you configure the python environment properly, you can start from a sample dataset, say ``story_cluster.txt``, and a wordvec model, say `glove_6B_300d.bin`. The following command create d2s formated data from `story_cluster.txt`. Edit the source for adapting to other datasets.

```
$ python export_d2s.py
raw categories: 54
document count: 1983
average words: 22
(1983, 4849)
```

It creates two files: `story_cluster.d2s` and `story_cluster.d2s.vocab0`. At this point, you need to request a patent protected C/MPI software called **d2_kmeans** from the author to process the formated data (free academic license available). The software will take these two files are input and output clustering labels as a file named `story_cluster.d2s_[xxxxxx].label_o` in the same directory. Type the same command again to evaluate the result that was reported in the paper. 

```
$ python export_d2s.py
```


----
The MIT License (MIT)

Copyright (c) 2017 Jianbo Ye

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


