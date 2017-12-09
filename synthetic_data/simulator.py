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
        self.__user_features = user_features
        self.__item_features = item_features
        self.n_users = n_users
        self.n_items = n_items
        self.bias = bias
        print("INFO: creating populations")
        self.users = self.make_population(n_users, user_features)
        self.items = self.make_population(n_items, item_features)

        print("INFO: creating user probability weights")
        self._user_probability_weights = self.__make_probability_weights(self.users)
        print("INFO: creating item probability weights")
        self._item_probability_weights = self.__make_probability_weights(self.items)

        self.__hash = tuple(self.observations_list).__hash__()  # to be updated each time we change observations
        self.__max_information: float = None
        self.__recommender_information: float = None

        self.__last_max_information = self.__hash  # avoid computing twice the info for the same observations
        self.__last_recommender_information = self.__hash
                
        if timestamp:
            self._time_unites = 86400
        else:
            self._time_unites = 1


    def __make_probability_weights(self, population):
        '''Given an individual, get the probability of getting any other one according to
        a powerlaw distribution.

        :return List(List): [i][j] probability of j given i
        '''
        n = len(population)
        probability_weights = {}
        probability_weights[None] = [n/(i+1) for i in range(n)]  # first one is n, second n/2 etc
        for p in range(0, len(population)):
            # ETA...
            if len(population) > 100:
                steps = int(len(population)/10)
                if p % steps==0 and p != 0:
                    pc = p // steps 
                    print("INFO: %s0 percent of the probability weights produced (%s)" % (pc, p))

            probability_weights[p] = list(probability_weights[None])
            # P(p_i | p_j) depends on the distance between p_i and p_j:
            for p2 in range(n):
                probability_weights[p][p2] += population[p]["similarities"][p2]*self.bias  # up if similar (>1), down if dissimilar (<1), nothing if 0
        return probability_weights


    def update_item_weights(self, item_id):
        '''When an item is bought, its probability (given any other item in the previous observation,
        hence the factor "similarity*bias") increases.
        '''
        for i in range(len(self.items)):
            self._item_probability_weights[i][item_id] += self.items[i]["similarities"][item_id]*self.bias
            

    
    @staticmethod
    def get_similarity(f1: List, f2: List) -> np.float64:
        # Do 1 & 2 have some feature in common?
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        else:
            return np.dot(f1, f2) / (norm1*norm2)


    def _random_user(self, u=None):
        '''Get next user.
        :par int u: user who was in the previous observation
        :return Tuple(user_id, p): new user_id and the probability we had to get it
        '''
        weights=self._user_probability_weights[u]
        user = random.choices(range(self.n_users), weights=weights)[0]
        p = weights[user] / np.linalg.norm(weights)
        return (user, p)


    # Get a sold item, but not if already sold that user
    def _sell(self, user_id, item_id=None):
        '''
        :param int user_id: user who is buying (needed for no-reselling)
        :param int item_id: item in the previous observation (if any)
        :return Tuple(item_id, p):  item chosen and the probability we had to get this item
                                    (-1, 1) if the user has already bought all items...
        '''        
        weights = self._item_probability_weights[item_id]
        if item_id is not None:
            sold = set([i_t[0] for i_t in self.observations.get(user_id, [(None, None)])])
            weights = np.roll(weights, -item_id)  # if it's a new user, who hasn't bought item, item is going to be the most probable one...
            # item_probability_weights[item_id] gives the prob given the past item
            # after having put to 0 all sold items, shift so that the most probable is the next one....
            weights = [w if i not in sold else 0 for i, w in enumerate(self._item_probability_weights[item_id])]
                
        if sum(weights) == 0:
            return (-1, 1)
        else:
            try:
                item = random.choices(population=range(self.n_items), weights=weights)[0]
                self.update_item_weights(item)
                p = weights[item] / np.linalg.norm(weights)
                return (item, p)
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
                steps = int(n_observations/10)
                if obs_done % steps==0 and obs_done != 0:
                    pc = obs_done // steps
                    print("%s0 percent of the observations produced (%s)" % (pc, obs_done))

            # some warning if there are overbuyers...
            if len(over_buyers) == 1 and n_warning == 0:
                print("Warning: user %s has bought all items" % list(over_buyers)[0])
                n_warning = 1
            elif len(over_buyers) == int(self.n_users*0.5) and n_warning == 1:
                print("Warning: 50% of users have bought all items")
                n_warning = 2
            
            (user, p_u) = self._random_user(user)  # new user given the previous user
            (new_item, p_i)  = self._sell(user, item)  # given the item bought in the last iteration... (even if not by the same user)            
            if new_item == -1:  # buyer bought all items
                over_buyers.add(user)
                continue
            item = new_item
            # update items' probability: more frequent an item, more probably will be bought!
            
            self.observations.setdefault(user, []).append((item, obs_done*self._time_unites))
            self.observations_list.append(((user, p_u), (item, p_i), obs_done*self._time_unites))
            self.__hash = tuple(self.observations_list).__hash__()  # new observations!
            obs_done += 1        

        if  len(over_buyers) == self.n_users:
            print("All buyers have bought all items...")


    def best_kl(self):
        '''The best possible KL divergence a recommender can get. This means a predictor
        which produces, for each user and observation, the actual probability
        distribution (weights), and tries to minimize the KL of this distribution
        with the one-hot of the label (the actual item in the obs)
        '''    
        pass
    
            
    def max_information(self):
        '''
        The actual information considering what we really know --that each observation depends only
        on the previous observation.

        NB The model used is quite simplicistic, so it'd make no sense to try to model a real dataset
        with that, but --ideally-- giving to a NN a good number of train:observation and
        label:next_observation it should get this result if we ask for a dimensionality reduction
        compatible with the item/user_features
        '''        
        if self.__hash != self.__last_max_information:
            max_uncertainty = math.log(self.n_items) + math.log(self.n_users)
            print("max possible uncertainty ", max_uncertainty)
            tot_surprise = 0.0
            for o in self.observations_list:
                tot_surprise += -math.log(o[0][1]) -math.log(o[1][1])
            print("actual uncertainty ", tot_surprise / len(self.observations_list))
            self.__max_information = max_uncertainty - tot_surprise / len(self.observations_list)
            self.__last_max_information = self.__hash

        return self.__max_information
    

    def recommender_information(self):
        '''
        How much can we lower our uncertainty when knowing how the
        dataset was built?

        For each set of observation, the uncertainty (average surprise, which
        is constant because we expect all books to have the same probability 1/N)
        is $log(N)$.

        This is the same for the "next buying" of each user.

        But in reality, once a user buys an item, the probability that the same user
        will buy a certain items is now bigger than 1/N, and smaller for others –not
        because we know that books' probability follows a power-law, but also because
        each user buys an item similar to the one bought one step earlier by a similar
        user, which did the same till the user in the previous step was the original
        user themselves.

        So, the probability of user u_i buying item b_k given their previous buying:
        
        P(observation=n, user=u_i, item=b_k | observation=n-j, user=u_i, item=b_l)

        where observation n-j is the last buying by user u_i. Of course, following the
        pattern P(n | n-1, n-2, ... , n-j) is possible in theory, but not at all straightforward
        (at least for Mario).

        We therefore model P(n | n-j) with something like:

        $$ P(n | powerlaw(b_l) * ( (distance(n_k, b_l)*bias)-1)*exp(-par*(j-1)) + 1.0) $$

        with the idea that there is a random walk from one user to the other
        (free meanpath is sqrt(j) independently on dimeensions --see freemeanpath.py--
        without considering the different P(user)).

        In case it's the first buying we'll have P(powerlaw(b_l)), like saying j -> inf.

        That's the best a recommender should be able to get (as it will get as input the previous
        buyings of the user and not from the previous step).

        NOTE:

        1. par deve essere ottimizzato anche nella NN! per ora lo mettiamo fisso...
        2. ha senso avere una fz simile nella NN? con powerlaw e bias?
        
        '''
        pass


    def print_observations(self, filename, separator="\t"):
        with open(filename, 'w') as f:
            f.write('user' + separator + str('item') + separator + 'timestamp' + "\n")
            for o in self.observations_list:
                f.write(str(o[0][0]) + separator + str(o[1][0]) + separator + str(o[2]) + "\n")


    def print_users(self, filename, separator="\t"):
        with open(filename, 'w') as f:
            n_features = len(self.__user_features)
            f.write('user' + separator + separator.join(['feature_'+str(i) for i in range(n_features)]) + "\n")
            for ui, u in enumerate(self.users):
                f.write(str(ui) + separator + separator.join([str(f) for f in u]) + "\n")
        
        
    @staticmethod            
    def make_population(n: int, features: List[int]) -> List[Tuple]:
        '''
        Values of features are uniformly distributed between -1 and 1.

        NO TWO INDIVIDUALS WILL HAVE THE SAME SET OF FEATURES! The idea
        is that features are, hiddenly, the minimum set of dimension I can use
        to describe individuals, therefore each individual is unique.

        The idea is that users with some of the features having the same value
        have similar taste and similar items should be bought after a buying.

        :param int n: Number of individuals (items/users)
        :param List[int] features: [feature1_n_values, feature2... ]
        :return Dict: {"feature": Tuple, "similarity": List(similarity with other individual)}
        '''
        population = []
        if reduce(mul, features, 1) < n:
            raise ValueError("Not enought features to make different individuals")
        while len(population) < n:
            # values uniformly distributed in [-1, +1] for each feature
            f = [2.0*random.randint(0, f-1)/(f-1)-1 for f in features]
            population.append({"features": tuple(f), "similarities": dict([(i, None) for i in range(n)])})
            present_i = len(population) - 1
            for i in range(present_i+1):
                sim = Simulator.get_similarity(population[i]["features"], population[present_i]["features"])
                population[i]["similarities"][present_i] = sim
                population[present_i]["similarities"][i] = sim
        return population
