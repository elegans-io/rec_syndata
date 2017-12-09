# Monte Carlo for Recommender Test

## Shortly

In brief, this is what the MC does:

* makes a list of users and items, each with a set of features, each feature with a certain number of possible values
* similarity between all users and all items is computed (cosine distance in the features)
* at start, users and items probabilities of being chosen follows a powerlaw
* after a user buys an item:
- another user is chosen, with probability being `powerlaw*(similarity with previous user)*bias`
- the powerlaw which choses next item is reset so that the most probable item would be next one, and on top of this powerlaw we apply the same factor as for users, ie `(similarity with previous user)*bias`

## What are the results?

The resulting buyings are quite realistic, in the sense that if a user buys an item, and after few observations (where possibly other users buy different items) buys another item, the two items are going to be similar.

This happens because items are chosen independently of users, and the "free mean path" of the item is still short.

On the contrary, two buyings from the same users made after many observations have a random similarity.

This reflect well the nature of real-life data, where the influence of interactions user-item back in the long past have little influence with today's interactions. On the contrary, what similar people buy today has much influence (if everyone buys Harry Potter, more people will buy it).


Simple Python3 functions to produce datasets to be tested with recommenders.

`make_observations`: Produce a list of observations --users who "buy" items.

Both users and items follow a simple power-law distribution:
* Items are "sold" with frequency `1/item_id^decay`
* Users "buy" with frequency `1/user_id^decay`

## Similarities

`bias` tells how similar users and items are.

RSD extract randomly `item1, item2, user1` and `user2`. It tries to
have `user2` similar to `user1`.

Once a user buys an item, its "personal" probabilities of buying
certain items change: now they'll have higher probability for items in
the same category, particularly the next one (`id1 = id1 + n_categories`).

It associates them in three observations: `(user1, item1), (user1, item2), (user2, item1)`.

#### Items

Items are in the same category (therefore similar) if their ID mod(n_categories) is the same. 

#### Users

The more users have features in common, the more they are similar
among themselves. Therefore, similarity ranges from 0 to len(features).

## Example

```python3
from synthetic_data import make_observations
n_users = 100
user_features = [5, 4]  # two users' features with 5 and 4 values resp.
n_items = 3000  # Must be much bigger than n_users, otherwise some users buy everything....
n_categories = 10
bias =  0.5 * n_items**decay  # so the last ones becomes comparable with the first ones when the user buys them
n_observations = 10000
obs, users = make_observations(n_users=n_users, user_features=user_features, n_items=n_items, n_categories=n_categories, n_observations=n_observations, bias=bias)
# write to file
from synthetic_data import write_dataset
write_dataset(users, n_items, n_categories, obs)
```
