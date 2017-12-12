import random
from typing import List, Tuple, Dict
import math
import numpy as np
from functools import reduce
from operator import mul

# Created by Mario Alemi 29 November 2017

class Simulator:

    def __init__(self, n_users: int, user_features: List[int],
                    n_items: int, item_features: int,
                    bias: int,
                    cross_influence: bool=True,
                    timestamp: bool=True) -> (List[Tuple], List[Tuple]):

        '''Produce a list of observations --users who "buy" items.
        e.g.

```
s = Simulator(n_users=1000, user_features=[3, 15, 100], n_items=100000, item_features=[50, 10000, 10], bias=10)  # takes long time
s.run()

        :param int n_users: Number of users
        :param List[int] user_features: [feature1_n_values, feature2... ]
        :param int n_items: Number of items
        :param List[int] item_features: as for users
        :param int bias: how similarity influences. If 0, at all. If 1, p(item after an item sim=-1)=0
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
        if self.bias >= 0.0 and self.bias <= 1.0:
            self.bias = np.float16(bias)
        else:
            raise ValueError("Bias must be in [0.0, 1.0]")
        print("INFO: creating users")
        self.users = self.make_population(n_users, user_features, msg="user")
        print("INFO: creating items")
        self.items = self.make_population(n_items, item_features, msg="item")

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
        probability_weights[None] = np.array([n/(i+1) for i in range(n)]).astype(np.float16)  # first one is n, second n/2 etc
        for p in range(0, len(population)):
            # ETA...
            if len(population) > 100:
                steps = int(len(population)/10)
                if p % steps==0 and p != 0:
                    pc = p // steps 
                    print("INFO: %s0 percent of the probability weights produced (%s)" % (pc, p))

            probability_weights[p] = probability_weights[None]
            # P(p_i | p_j) depends on the distance between p_i and p_j:
            for p2 in range(n):
                try:
                    sim = population[p]["similarities"][p2]
                except KeyError:
                    sim = np.float16(Simulator.get_similarity(population[p]["features"], population[p2]["features"]))
                    population[p]["similarities"][p2] = sim
                    population[p2]["similarities"][p] = sim
                probability_weights[p][p2] += probability_weights[p][p2]*sim*self.bias  # up if similar (>1), down if dissimilar (<1), nothing if 0
        return probability_weights


    def update_item_weights(self, item_id):
        '''When an item is bought, its probability increases.
        Because we use a zipf distribution which starts with n, n/2, n/3 .... 1,
        we increase by 1 the bin of the correspondent item
        
        '''
        for i in range(len(self.items)):
            self._item_probability_weights[i][item_id] += 1
            

    
    @staticmethod
    def get_similarity(f1: List, f2: List, fast=True) -> np.float64:
        # Do 1 & 2 have some feature in common?
        if not fast:
            norm1 = np.float16(np.linalg.norm(f1))
            norm2 = np.float16(np.linalg.norm(f2))
            if norm1 == 0.0 or norm2 == 0.0:
                return np.float16(0.0)
            else:
                return np.float16(np.dot(f1, f2) / (norm1*norm2))
        else:
            return np.float16(np.dot(f1, f2))

    def _random_user(self, u=None):
        '''Get next user.
        :par int u: user who was in the previous observation
        :return Tuple(user_id, p): new user_id and the probability we had to get it
        '''
        weights = self._user_probability_weights[u]
        user = random.choices(range(self.n_users), weights=weights)[0]
        p = np.float16(weights[user] / np.linalg.norm(weights))
        return (user, p)


    # Get a sold item, but not if already sold that user
    def _sell_and_tout(self, user_id, item_id=None, tout=True):
        '''Sell an item and, if successful, increase the item probability of being sold
        :param int user_id: user who is buying (needed for no-reselling)
        :param int item_id: item in the previous observation (if not item_id=None)
        :return Tuple(item_id, p):  item chosen and the probability we had to get this item
                                    (-1, 1) if the user has already bought all items...
        '''        
        weights = self._item_probability_weights[item_id]
        if item_id is not None:
            sold = ([i_t[0] for i_t in self.observations.get(user_id, [(None, None)])])
            weights = np.roll(weights, -item_id)  # if it's a new user, who hasn't bought item, item is going to be the most probable one...
            # item_probability_weights[item_id] gives the prob given the past item
            # after having put to 0 all sold items, shift so that the most probable is the next one....
            weights = [w if i not in sold else 0 for i, w in enumerate(self._item_probability_weights[item_id])]
                
        if sum(weights) == 0:
            return (-1, 1)
        else:
            try:
                item = random.choices(population=range(self.n_items), weights=weights)[0]
                if tout:
                    self.update_item_weights(item)
                p = np.float16(weights[item] / np.linalg.norm(weights))
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
            (new_item, p_i)  = self._sell_and_tout(user, item, tout=True)  # given the item bought in the last iteration... (even if not by the same user)            
            if new_item == -1:  # buyer bought all items
                over_buyers.add(user)
                continue
            item = new_item
            self.observations.setdefault(user, []).append((item, obs_done*self._time_unites))
            self.observations_list.append(((user, p_u), (item, p_i), obs_done*self._time_unites))
            
            obs_done += 1        

        self.__hash = tuple(self.observations_list).__hash__()  # new observations!            
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
        How much can we lower our uncertainty when knowing how the dataset was built?

        For each set of observation, the uncertainty (average surprise, which
        is constant because we expect all books to have the same probability 1/N)
        is $log(N)$.

        This is the same for the "next buying" of each user.

        But in reality, once a user buys an item, the probability that the same user
        will buy a certain items is now bigger than 1/N, and smaller for others –not
        just because we know that books' probability follows a power-law, but also because
        each user buys an item similar to the one bought one step earlier by a similar
        user, which did the same till the user in the previous step was the original
        user themselves.

        So, the probability of user u_i buying item b_k given their previous buying:
        
        P(observation=n, user=u_i, item=b_k) = f(observation=n-j, user=u_i, item=b_l)

        --where observation n-j is the last buying by user u_i-- is hard to compute
        (NB there is "j" because u_i bought j-time ago, with many other transactions by other users
        in-between).
        One could follow the probability of each step in the pattern P(n | n-1, n-2, ... , n-j),
        but that's not at all straightforward (at least for Mario).

        We therefore model P(n | n-j) with something like:

        $$ f(powerlaw(b_l)(1 + distance(b_k, b_l)*bias)*1/sqrt(j) ) $$

        The factor 1/j comes from two considerations:

        1. Euristic, books bought long time ago are still influencial to what we read today.
        2. The average meanpath goes with sqrt(j). That means that (assuming that distance
        among the items is uniformly distributed in bins of width d_m=max_distance/n) the number of
        possible items which originated the one we saw is the ones with distance [0, d_m],
        then [d_m, d_m*sqrt(2)], [d_m*s(2), d_m*s(3)], ... which goes of course down like 1/sqrt(i)

        That's the best a recommender should be able to get (as it will get as input the previous
        buyings of the user and not from the previous step).

        ==> ha senso avere una fz simile nella NN? con powerlaw e bias?
        
        '''
        pass


    def export(self, filenames=["is.csv", "us.csv", "os.csv"], separator="\t"):
        
        with open(filenames[2], 'w') as f:
            f.write('user' + separator + str('item') + separator + 'timestamp' + "\n")
            for o in self.observations_list:
                f.write(str(o[0][0]) + separator + str(o[1][0]) + separator + str(o[2]) + "\n")

        with open(filenames[1], 'w') as f:
            n_features = len(self.__user_features)
            f.write('user' + separator + separator.join(['feature_'+str(i) for i in range(n_features)]) + "\n")
            for ui, u in enumerate(self.users):
                f.write(str(ui) + separator + separator.join([str(f) for f in u]) + "\n")
        
        with open(filenames[0], 'w') as f:
            n_features = self.__item_features
            f.write('item' + separator + separator.join(['feature_'+str(i) for i in range(n_features)]) + "\n")
            for ui, u in enumerate(self.items):
                f.write(str(ui) + separator + separator.join([str(f) for f in u]) + "\n")
        
        
    @staticmethod            
    def make_population(n: int, features, msg="generic") -> List[Tuple]:
        '''
        Values of features are uniformly distributed between -1 and 1.

        The idea is that features are, hiddenly, the minimum set of dimension I can use
        to describe individuals.

        Users with some of the features having the same value
        have similar taste and similar items should be bought after a buying.

        :param int n: Number of individuals (items/users)
        :param (List[int] OR int) features: [feature1_n_values, feature2... ] or Number of features. If list: only feature*_n_values [-1, 1] are generated for each feature (users!). In the second case (items!) a uniform float [-1, 1] for each features will be generated.
        :return Dict: {"feature": Tuple, "similarity": List(similarity with other individual)}
        '''
        population = []
        while len(population) < n:
            if n > 1000:
                steps = int(n/10)
                if len(population) % steps==0 and len(population) != 0:
                    pc = len(population) // steps 
                    print("INFO: %s0 percent of the %s population produced (%s)" % (pc, msg, len(population)))

            # values uniformly distributed in [-1, +1] for each feature
            try:
                f = np.array([2.0*random.randint(0, f-1)/(f-1)-1 for f in features]).astype(np.float16)
            except TypeError:  # it's int...
                f = np.random.uniform(-1, 1, features).astype(np.float16)
            population.append({"features": f, "similarities": {}})
        return population
