import random
from typing import List, Tuple, Dict

# Created by Mario Alemi 29 November 2017

def make_items(n_items, n_categories):
    """
    items divisible by sim are similar, 1 modular sim are dissimilar:
    """
    items = []
    for i in range(0, n_items):
        items.append((i, i % n_categories))
    return items
    
    
def make_users(n_users: int, user_features: List[int]) -> List[Tuple]:
    """
    Return [(feature1_value, feature2_value etc), .... ]

    Values of features are uniformly distributed.

    The idea is that users with some of the features having the same value
    have similar taste (see make_observations)
    
    :param int n_users: Number of users
    :param List[int] user_features: [feature1_n_values, feature2... ]
    """
    users = []
    for u in range(n_users):
        # values uniformly distributed in [-1, +1] for each feature
        f = [2.0*random.randint(0, f-1)/(f-1)-1 for f in user_features]
        users.append(tuple(f))

    return users

       

    

