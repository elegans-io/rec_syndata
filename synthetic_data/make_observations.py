import math
from typing import List, Tuple
from synthetic_data.utils import *

# Created by Mario Alemi 29 November 2017

observations = {}

def make_observations(n_users: int, user_features: List[int],
                      n_items: int, n_categories: int,
                      n_observations: int,  bias: int) -> (List[Tuple], List[Tuple]):
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
observations, users = make_observations(n_users=n_users, user_features=user_features, n_items=n_items, n_categories=n_categories, n_observations=n_observations, bias=bias)
```
    
    :param int n_users: Number of users
    :param List[int] user_features: [feature1_n_values, feature2... ]
    :param int n_items: Number of items
    :param int n_observations: Number of observations to be produced (if odd, the lower even will be produced)
    :param int sim: $$ P(similar(A) in Obs_{i+1} | A in Obs_{i} ) = binomial(success >= 1, trial=bias, p=1/n_items)$$. In practice, if item A is bought, in the next observation we extract up to `bias` time unless a similar item is bought.
    :param int bias: how similar and dissimilar the items are
    :return List[Tuple3]: list of observations (user_id, item_id, timestamp)

    """
    global observations
    
    users = make_users(n_users, user_features)

    def random_user(u=None):
        return random.choices(range(n_users), weights=user_probability_weights[u])[0]


    # Get a sold item, but not if already sold that user
    def sell(user_id):
        '''
        :return int item_id: -1 if the user has already bought all items...
        '''
        sold = [i_t[0] for i_t in observations.get(user_id, [(None, None)])]
        # No reselling
        weights = [w if i not in sold else 0 for i, w in enumerate(item_probability_weights[user_id])]
        if sum(weights) == 0:
            return -1
        else:
            return random.choices(population=range(n_items), weights=weights)[0]

    
    # Once a user reads from a certain categories, we
    # want them to favour this category of the others
    def increase_similar_items_p(user_id, item_id):
        for i in range(n_items):
            if i % n_categories == item_id % n_categories:
                item_probability_weights[user_id][i] += bias

        
    def users_similarity(user1, user2) -> int:
        # Do users 1 & 2 have some feature in common?
        users_sim = 0
        for i in range(len(user_features)):
            if (users[user1][i] == users[user2][i]):
                users_sim += 1
        return users_sim

    # the first item is the most bought etc, the same
    # for users.
    item_probability_weights = {}
    user_probability_weights = {}
    
    for u in range(0, n_users):
        # Only, this will change after a purchase:
        # the prob for similar items (same cat) will increase
        item_probability_weights[u] = [n_items/math.sqrt(i+1) for i in range(n_items)]
        # After u1 has made a purchase, the next user is more
        # likely to make the same purchase
        # in case of no previous purchase: 
        user_probability_weights[None] = [n_users/math.sqrt(u+1) for u in range(n_users)]
        for u2 in range(0, n_users):
            user_probability_weights[u] = [n_users/math.sqrt(u+1) for u in range(n_users)]
            user_probability_weights[u][u2] += users_similarity(u, u2)*bias
            
    obs_done = 0
    over_buyers = set()  # buyers who bought all items
    while(obs_done <= n_observations and len(over_buyers) < n_users):
        if obs_done % int(n_observations/10)==0 and obs_done != 0:
            pc = obs_done // int(n_observations/10)
            print("%s0 percent of the observations produced (%s)" % (pc, obs_done))
            
        if len(over_buyers) > 0:
            print("Warning: some users have bought all items")
        elif len(over_buyers) == int(n_users*0.5):
            print("Warning: 50% of users have bought all items")
        elif len(over_buyers) == int(n_users*0.9):
            print("Warning: 90% of users have bought all items")

        user_id1 = random_user()
        user_id2 = random_user(user_id1)
        item_id1 = sell(user_id1)
        if item_id1 == -1:  # buyer bought all items
            over_buyers.add(user_id1)
            continue
        observations.setdefault(user_id1, []).append((item_id1, obs_done))
        increase_similar_items_p(user_id1, item_id1)        
        obs_done += 1        
                     
        # Let's get item2 now...
        item_id2 = sell(user_id1)
        if item_id2 == -1:  # buyer bought all items
            over_buyers.add(user_id1)
            continue            
        observations.setdefault(user_id1, []).append((item_id2, obs_done))
        increase_similar_items_p(user_id1, item_id2)
        obs_done += 1

        # If user2 hasn't bought item1 yet, they'll do it now
        if item_id1 not in [i_t[0] for i_t in observations.get(user_id2, [(None, None)])]:
            observations.setdefault(user_id2, []).append((item_id1, obs_done))
            obs_done += 1
        increase_similar_items_p(user_id2, item_id1)

    if  len(over_buyers) == n_users:
        print("All buyers have bought all items...")

    return observations, users

