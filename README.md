# 1 Task description
You are given two tables (left table and right table) of electronic products. Each table is from a different shopping website. Each row in a table represents a product instance. For every pair of tuples(Li, Rj), where Li is a tuple in the left table and Rj is a tuple in the right table, it is either a match or a non-match. A pair of tuples is a match is they refer to the same real-world entity. Three files are provided in data.zip: ltable.csv (the left table), rtable.csv (the right table), and train.csv (the training set). The training set contains a subset of tuple pairs, where some of them are matches and some of them are non-matches. The training set has three columns "ltable_id", "rtable_id", and "label". "label" being 1/0 denotes match/non-match. The task is to find all remaining matching pairs like (Li,Rj) in the two tables, excluding those matches already found in the training set.

# 2 Solution outline

The solution includes 6 steps: (1) Data reading and EDA, (2) Blocking, (3) Feature engineering, (4) Model training and (5) Generating initial output (6) filtering extraneous results.

## 2.1 Data reading and EDA

In this step, we read the left table and right table, as well as the training set. We explore the dataset to get some ideas of designing the solution. For example, we found that the left table has 2554 rows and the right table has 22074 rows, so there are 2554*22074=56376996 pairs. Examining every pair is very inefficient, so we will need a blocking step to reduce the number of pairs that we will work on.

## 2.2 Blocking

We perform blocking on the attribute "brand", generating a candidate set of id pairs where the two ids in each pair share the same brand. This is based on the intuition that two products with different brand are unlikely to be the same entity. Our blocking method reduces the number of pairs from 56376996 to 256606.

## 2.3 Feature engineering

For each pair in the candidate set, we generate a feature vector of 15 dimensions by obtaining the Cosine similarity, Jaccard similarity, and Levenshtein distance of pair on the five attributes. The cosine similarity is calculated by using a Universal Sentence Encoder. (https:/Avww.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/) In this way, we obtain a feature matrix Xc for the candidate set. We do the same to the pairs in the training set to obtain a feature matrix Xt. The labels for the training set is denoted as yt.
 
## 2.4 Model training

We use a random forest classifier. We train the model on (Xt, yt). Since the number of non-matches is much more than the number of matches in the training set, we set class_weight="balanced” in random forest to handle this training data imbalance problem. We perform prediction on Xc to get predicted labels yc for the candidate set.

## 2.5 Generating initial output 
The pairs with yc = 1 are our predicted matching pairs M. We remove the matching pairs already in the training set from M to obtain M-. Finally, we save M- to output.csv, which results in 2480 possible matches. The implementation for the initial output (output.csv) is all carried out in the solution.py file.

## 2.6 Filtering output further

In truncate.py, SentenceBERT is used to calculate the Cosine similarity between the titles from each pair from output.csv. Only pairs where similarity >= 0.70 are picked and are output into the finalOutput.csv file which contains 255 matches.
