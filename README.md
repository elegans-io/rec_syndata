# Recommender Synthetic Datasets

Simple Python3 functions to produce datasets to be tested with recommenders.

`make_observations`: Produce a list of observations --users who "buy" items.

Both users and items follow a simple power-law distribution:
* Items are "sold" with frequency `1/item_id`
* Users "buy" with frequency `1/user_id`

## Similarities

`bias` tells how many trials must be attempted if a non-similar item/item_category/user are found.

RSD extract randomly `item1, item2, user1` and `user2`. It associates them in three observations: `(user1, item1), (user1, item2), (user2, item1)`. It then tries to have `item2` similar to `item1` and `user2` similar to `user1`.

#### Items

Items are in the same category if their ID mod(n_categories) is the same. 

If item1 and item2 are not similar, it attempts `bias` times to extract a different item2 hoping to get a similar one. Given an item1, users have higher probability to buy item1+n_categories. If this does not happen, they have a higher probability of buying an item in the same category.

#### Users

If no features are in common between `user1` and `user2`, the latter is extracted again `bias` times unti (if) a similar one is found.

## Example

```python3
from synthetic_data import make_observations
n_users = 100
user_features = [5, 4]  # two users' features with 5 and 4 values resp.
n_items = 100
n_categories = 10
bias = 5 
n_observations = 10000
observations, users = make_observations(n_users=n_users, user_features=user_features, n_items=n_items, n_categories=n_categories, n_observations=n_observations, bias=bias)
# write to file
from synthetic_data import write_dataset
write_dataset(users, n_items, n_categories, observations)
```
