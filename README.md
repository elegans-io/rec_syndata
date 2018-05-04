# Monte Carlo for Recommender Test

## Creation of users and items

Starting from a uniform or power-law distribution, a population of users and one of items is created.

Users have discreet features (corresponding to age, gender, town etc), items continuous (corresponding to components in different categories, price).

Continuous features are computed according to a double power-law. Feature "0" is very popular, feature "n-1" is not. Once features are randomly ordered according to the power-law (feature "0" has highest probability to be the first, but will not necessarily be so), random values are assigned to each features according to a power-law. This means that the randomly assigned "predominant feature" will mostly characterise the item, the second feature half so and so on.


## Influencing purchases

Initially, the probability that a user is going to buy is either the same (`users_distribution="uniform"`) or follows a power-law (`users_distribution="zipf"`). Once a user has bought something, similar users have a higher probability to buy something.

Similarly, once a user has bought an item, the probability that they will buy a similar item next time they make a purchase is higher.

### Probability of purchase

To compute the probability P(A|B) that a user buys item A after item B, first, the cosine similarity(A, B) [-1, +1] between A and B is computed. Then:

```python
P(A|B) = p(A) * (1 - bias) + P(A) * bias * similarity(A, B)
if P(A|B) < 0:
   P(A|B) = 0
```

For bias=0 there is no influence. For bias=1 the probability decreases with similarity down to 0, then it remains null. It's therefore not possible that a user buys an item completely dissimilar from the one they have just bought.

Similarly, if `tout` is `True`, users influence each other, so after a user has bought an item, similar users have higher probability of buying an item.

## Analysis of results

Once the simulation has ran, we can test the goodness of some simple algorithms comparing how well we can predict the similarity amongst items or users (`get_similar_items`)

### Co-occurrence

With the function `compute_cooccurences` for each item the most frequently co-occurring items are computed. Are they similar as well?

### Log-likelihood Ratio on co-occurring items

Very simple implementation of [Mahout's Loglikelihood](https://github.com/apache/mahout/blob/master/math/src/main/java/org/apache/mahout/math/stats/LogLikelihood.java)

P(A|B) means probability that A is in a basket given that B is in such a basket.

### Log-likelihood Ratio on sequential items

As above, but instead of measuring when two items have been bought by the same user, here we check when an item has ben bought after another item. The co-occurrence item is therefore asymmetric.

P(A|B) means probability that A is bought after B.

This gives clearly the best results.


## Further developments

### Periodic purchases

At the moment, an item can be bought just once --we had books and movies in mind. For other businesses, like supermarket, items are bought periodically and may have a lifetime (diapers are bought for X years at a certain frequency, then babies learn how to go to the bathroom).

### Actual sequentiality

At the moment, P(A|B)=P(B|A). But in many cases, it shouldn't. For TV series and sagas, `P(episode_n in basket | episode_{n+k} in basket) >> P(episode_{n+k} in basket | episode_n in basket)`. So after episode_2 we should be able to recommend episode_3 and not episode_1.

### Checking different algorithms

A simple implementation of the network described [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/archive/45530.pdf) can take into account the profile (if `tout=True`) and thousands of features (corresponding to frequency of words in the description of the item).

## Example

Simple Python3 script to produce datasets to be tested with recommenders.

```python3
from synthetic_data import Simulator
# creates populations of 100 users without features
# and items with 10 continuous features.
# Users do not influence each other (tout is false), items do.
# Both start from a uniform distribution of being chosen.
s = Simulator(n_users=100, user_features=0, n_items=50, item_features=10, bias=10.0, users_distribution="uniform", items_distribution="uniform", tout=False)
s.run(1000)
# print the items similar to item 1
s.get_similar_items(1)
# compute the co-occurrence matrices (needed for LLR)
s.compute_cooccurrences()
# compute the "sequentiality" matrix (needed for sequential LLR)
s.compute_sequentials()
# get the most similar items to item 1 according to...
# ...LLR on sequential items
s.get_sequentials_llr_items(1)
# ...co-occurrence
s.get_cooccurred_items(1)
# ...LLR on co-occurrence
s.get_llr_items(1)
# export the data
s.export(separator=',', print_timestamp=False)
```
