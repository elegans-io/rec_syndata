# Monte Carlo for Recommender Test

## Shortly

In brief, if a user buys an item, this item becomes more popular for other users, and similar users have a higher probability than expected to buy in the next timestamp. 


## What are the results?

The resulting buyings are quite realistic, in the sense that if a user buys an item, and after few observations (where possibly other users buy different items) buys another item, the two items are going to be similar.

This happens because items are chosen independently of users, and the "free mean path" of the item is still short.

On the contrary, two buyings from the same users made after many observations are not influenced.

This reflect well the nature of real-life data, where the influence of interactions user-item back in the long past have little influence with today's interactions. On the contrary, what similar people buy today has much influence (if everyone buys Harry Potter, more people will buy it).


Simple Python3 script to produce datasets to be tested with recommenders.

```python3
from synthetic_data import Simulator
# creates populations of users without features
# and items with 10 continuous features
s = Simulator(n_users=101, user_features=0, n_items=1500, item_features=10, bias=1.0)
# users have two features, the first with 3 possible values, the secondo 10
s = Simulator(n_users=101, user_features=[3, 10], n_items=1500, item_features=10, bias=1.0)
# do not impose a zipf distribution, but start with a uniform one:
s = Simulator(n_users=101, user_features=[3, 10], n_items=1500, item_features=10, bias=1.0, users_distribution="uniform", items_distribution="uniform")
# produces observations
s.run(1000)
# export the data
s.export(separator=',', print_timestamp=False)
```

Both users and items follow a simple power-law distribution:

* Items are "sold" with frequency `1/item_id^decay`
* Users "buy" with frequency `1/user_id^decay`

## Similarities

`bias` in [0, 1] tells how similar users and items influence each other.

