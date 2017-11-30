from typing import List, Tuple
from synthetic_data.utils import *

# Created by Mario Alemi 29 November 2017

def make_observations(n_users: int, user_features: List[int],
                      n_items: int, n_categories: int,
                      n_observations: int, sim: int, bias: int) -> List[Tuple]:
    """Produce a list of observations --users who "buy" items.

    Both users and items follow a simple power-law distribution.
    Items are "sold" with frequency 1/item_id
    Users "buy" with frequency 1/user_id

    Example:

```
from synthetic_data import make_observations
n_users = 100
user_features = [5, 4]  # two users' features with 5 and 4 values resp.
n_items = 100
n_categories = 10
bias = 5 
n_observations = 10000
sim = 11
observations = make_observations(n_users=n_users, user_features=user_features, n_items=n_items, n_categories=n_categories, n_observations=n_observations, sim=sim, bias=bias)
```
    
    :param int n_users: Number of users
    :param List[int] user_features: [feature1_n_values, feature2... ]
    :param int n_items: Number of items
    :param int n_observations: Number of observations to be produced (if odd, the lower even will be produced)
    :param int sim: $$ P(similar(A) in Obs_{i+1} | A in Obs_{i} ) = binomial(success >= 1, trial=bias, p=1/n_items)$$. In practice, if item A is bought, in the next observation we extract up to `bias` time unless a similar item is bought.
    :param int bias: how similar and dissimilar the items are
    :return List[Tuple3]: list of observations (user_id, item_id, timestamp)

    """
    
    users = make_users(n_users, user_features)
    observations = []
    get_user_from_int = {}
    u_count = 0
    for u in range(0, n_users):
        for _ in range(0, u):
            get_user_from_int[u_count] = n_users - 1 - u
            u_count += 1
            
    get_item_from_int = {}
    i_count = 0
    for i in range(0, n_items):
        for _ in range(0, i):
            get_item_from_int[i_count] = n_items - 1 - i
            i_count += 1

    for o in range(0, int(n_observations/2)):
        user1 = get_user_from_int[random.randint(0, u_count-1)]
        user2 = get_user_from_int[random.randint(0, u_count-1)]
        item1 = get_item_from_int[random.randint(0, i_count-1)]
        item2 = get_item_from_int[random.randint(0, i_count-1)]

        # let's see if we can have the "very similar one" (id = id + n_categories).
        # This means that, given 7 categories, after book 1 people tend to buy 8, after 8 go for 15 and so on till no more book

        if item1 + n_categories <= n_items - 1:
            trials = 0
            while (item2 != item1 + n_categories) and trials < bias:
                # ...we give one more chance...
                item2 = get_item_from_int[random.randint(0, i_count-1)]
                trials += 1

        # ...if not the very similar one, at least same category?
        trials = 0
        while (item1 % n_categories != item2 % n_categories) and trials < bias:
            # ...we give one more chance...
            item2 = get_item_from_int[random.randint(0, i_count-1)]
            trials += 1
            
        # Do users 1 & 2 have something in common?
        users_sim = False
        for i in range(len(user_features)):
            if (users[user1][i] == users[user2][i]):
                users_sim = True
                break

        # if not, try to get another, more similar, user 2
        if not users_sim:
            trials = 0
            for _ in range(bias):
                user2 = get_user_from_int[random.randint(0, u_count-1)]
                for i in range(len(user_features)):
                    if (users[user1][i] == users[user2][i]):
                        break
            
        observations.append((user1, item1, 3*o))
        observations.append((user1, item2, 3*o+1))
        observations.append((user2, item1, 3*o+2))

    still = n_observations - len(observations)
    for _ in range(still):
        user = get_user_from_int[random.randint(0, u_count-1)]
        item = get_item_from_int[random.randint(0, i_count-1)]
        observations.append((user, item2, n_observations))

    return observations, users

