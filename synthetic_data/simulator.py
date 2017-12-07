import random
from typing import List, Tuple, Dict
import math
import numpy as np
from functools import reduce
from operator import mul

# Created by Mario Alemi 29 November 2017

class Simulator:

    def __init__(self, n_users: int, user_features: List[int],
                    n_items: int, item_features: List[int],
                    bias: int,
                    powerlaw: float=0.5,
                    cross_influence: bool=True,
                    timestamp: bool=True) -> (List[Tuple], List[Tuple]):

        '''Produce a list of observations --users who "buy" items.

        :param int n_users: Number of users
        :param List[int] user_features: [feature1_n_values, feature2... ]
        :param int n_items: Number of items
        :param List[int] item_features: as for users
        :param int bias: how similar and dissimilar the items are
        :param bool cross_influence: after a purchase, a similar user is extracted and given the same item
        :param int timestamp: unix-like timestamp (in seconds)
        :return List[Tuple3]: list of observations (user_id, item_id, timestamp)

        '''
        self.observations = {}
        self.observations_list = []
        self.n_users = n_users
        self.n_items = n_items
        self.powerlaw = powerlaw
        self.bias = bias
        self.users = self.make_population(n_users, user_features)
        self.items = self.make_population(n_items, item_features)
        
        self._user_probability_weights = self.__make_probability_weights(self.users)
        self._item_probability_weights = self.__make_probability_weights(self.items)
        
        if timestamp:
            self._time_unites = 86400
        else:
            self._time_unites = 1


    def __make_probability_weights(self, population):
        '''Given an individual, get the probability of getting any other one.
        '''
        n = len(population)
        probability_weights = {}
        probability_weights[None] = [n/(i+1)**self.powerlaw for i in range(n)]
        for p in range(0, len(population)):            
            probability_weights[p] = list(probability_weights[None])
            # P(p_i | p_j) depends on the distance between p_i and p_j:
            for p2 in range(0, n):
                probability_weights[p][p2] += self._similarity(p, p2, population)*self.bias  # up if similar (>1), down if dissimilar (<1), nothing if 0
        return probability_weights

    
    def _similarity(self, i1: int, i2: int, population) -> np.float64:
        # Do 1 & 2 have some feature in common?
        norm1 = np.linalg.norm(population[i1])
        norm2 = np.linalg.norm(population[i2])
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        else:
            return np.dot(population[i1], population[i2]) / (np.linalg.norm(population[i1])* np.linalg.norm(population[i2]))


    def _random_user(self, u=None):
        '''Get next user
        '''
        return random.choices(range(self.n_users), weights=self._user_probability_weights[u])[0]


    # Get a sold item, but not if already sold that user
    def _sell(self, user_id, item_id=None):
        '''
        :param int user_id: user who is buying (needed for no-reselling)
        :param int item_id: item in the previous observation (if any)
        :return int item_id: -1 if the user has already bought all items...
        '''
        sold = [i_t[0] for i_t in self.observations.get(user_id, [(None, None)])]
        # item_probability_weights[item_id] gives the prob given the past item
        # after having put to 0 all sold items, shift so that the most probable is the next one....
        weights = [w if i not in sold else 0 for i, w in enumerate( self._item_probability_weights[item_id])]
        if item_id is not None:
            weights = np.roll(weights, -item_id-1)
                
        if sum(weights) == 0:
            return -1
        else:
            try:
                return random.choices(population=range(self.n_items), weights=weights)[0]
            except:
                print("ERROR: Weights not valid for random.choices: ", weights)
    

    def run(self, n_observations):
        obs_done = 0
        over_buyers = set()  # buyers who bought all items
        n_warning = 0
        user = None
        item = None
        # go till all users have bought all items or n_obs
        while(obs_done <= n_observations and len(over_buyers) < self.n_users):
            # ETA...
            if n_observations > 1000:
                if obs_done % int(n_observations/10)==0 and obs_done != 0:
                    pc = obs_done // int(n_observations/10)
                    print("%s0 percent of the observations produced (%s)" % (pc, obs_done))

            # some warning if there are overbuyers...
            if len(over_buyers) == 1 and n_warning == 0:
                print("Warning: user %s has bought all items" % list(over_buyers)[0])
                n_warning = 1
            elif len(over_buyers) == int(self.n_users*0.5) and n_warning == 1:
                print("Warning: 50% of users have bought all items")
                n_warning = 2
            
            user = self._random_user(user)  # new user given the previous user
            item = self._sell(user, item)  # given the item bought in the last iteration... (even if not by the same user)            
            if item == -1:  # buyer bought all items
                over_buyers.add(user)
                continue
            self.observations.setdefault(user, []).append((item, obs_done*self._time_unites))
            self.observations_list.append((user, item, obs_done*self._time_unites))
            obs_done += 1        

        if  len(over_buyers) == self.n_users:
            print("All buyers have bought all items...")


    def total_entropy():
        '''
        < Surprise(observation_list[:n] | observation_list[:n-1]) >
        '''
        print("not yet")

        
    @staticmethod            
    def make_population(n: int, features: List[int]) -> List[Tuple]:
        """
        Return [(value, value etc), .... ]

        Values of features are uniformly distributed between -1 and 1.

        NO TWO INDIVIDUALS WILL HAVE THE SAME SET OF FEATURES! The idea
        is that features are, hiddenly, the minimum set of dimension I can use
        to describe individuals, therefore each individual is unique.

        The idea is that users with some of the features having the same value
        have similar taste and similar items should be bought after a buying.

        :param int n: Number of individuals (items/users)
        :param List[int] features: [feature1_n_values, feature2... ]
        """
        individuals = set()
        if reduce(mul, features, 1) < n:
            raise ValueError("Not enought features to make different individuals")
        while len(individuals) < n:
            # values uniformly distributed in [-1, +1] for each feature
            f = [2.0*random.randint(0, f-1)/(f-1)-1 for f in features]
            individuals.add(tuple(f))

        return list(individuals)
