import random
from typing import List, Dict, Tuple
import math
import numpy as np
from synthetic_data.observation import Observation
import os
import pickle
import time


# Created by Mario Alemi 29 November 2017

def xlogx(x):
    return -x * math.log(float(x)) if x > 0 else 0


def xlogx_vec(x):
    xlogx_vectorized = np.vectorize(xlogx)
    return np.array([xlogx_vectorized(a) for a in x])


def entropy2(a, b):
    return -xlogx(a + b) + xlogx(a) + xlogx(b)


def entropy4(a, b, c, d):
    return -xlogx(a + b + c + d) + xlogx(a) + xlogx(b) + xlogx(c) + xlogx(d)


def loglikelihood_ratio(k11, k10, k01, k00):
    assert (k11 >= 0 and k10 >= 0 and k01 >= 0 and k00 >= 0)
    row_entropy = entropy2(k11 + k10, k01 + k00)
    column_entropy = entropy2(k11 + k01, k10 + k00)
    matrix_entropy = entropy4(k11, k10, k01, k00)
    if row_entropy + column_entropy < matrix_entropy:
        # round off error
        return 0.0
    return 2.0 * (row_entropy + column_entropy - matrix_entropy)


def root_loglikelihood_ratio(k11, k10, k01, k00):
    result = k11.copy()
    result.fill(0.0)
    for i in range(len(result)-1):
        for j in range(len(result[i])-1):
            result[i][j] = np.sqrt(loglikelihood_ratio(k11[i][j], k10[i][j], k01[i][j], k00[i][j]))
            if (k11[i][j] + k10[i][j]) > 0 and (k01[i][j] + k00[i][j]) > 0:
                if k11[i][j] / (k11[i][j] + k10[i][j]) < k01[i][j] / (k01[i][j] + k00[i][j]):
                    result[i][j] = -result[i][j]
    return result


class Simulator:

    def __init__(self, n_users: int, user_features: List[int],
                 n_items: int, item_features: int,
                 bias: int,
                 users_distribution: str = "zipf",
                 items_distribution: str = "zipf",
                 read_cache_dir: str = None,
                 save_cache_dir: str = None,
                 timestamp: bool=True,
                 tout: bool=True) -> None:

        """Produce a list of observations --users who "buy" items.
        e.g.

```
s = Simulator(n_users=101, user_features=0, n_items=1500, item_features=10, bias=1.0)
s.run()
```

        :param int n_users: Number of users
        :param List[int] user_features: [feature1_n_values, feature2... ]
        :param int n_items: Number of items
        :param List[int] item_features: as for users
        :param int bias: how similarity influences. If 0, at all. If 1, p(item after an item sim=-1)=0
        :param int timestamp: unix-like timestamp (in seconds)
        :return List[Tuple3]: list of observations (user_id, item_id, timestamp)

        """
        self.user_buying_dict = {}  # {user: [(item, timestamp), (), ...], ...}
        self.observations_list = []
        self._user_features = user_features
        self._item_features = item_features
        self.n_users = n_users
        self.n_items = n_items
        self._reset_cooccurrences_matrices()
        self._reset_sequentials_matrices()
        self.tout = tout
        assert read_cache_dir is None or save_cache_dir is None, \
            "saving and reading the cache at the same time does not make sense"
        self.read_cache_dir = read_cache_dir
        self.save_cache_dir = save_cache_dir
        if bias >= 0:
            self.bias = np.float32(bias)
        else:
            raise ValueError("Bias must be equal or bigger than 0")

        # creating users
        self.users = self.make_population(n=n_users,
                                          features=user_features,
                                          population_name="user")
        # creating items
        self.items = self.make_population(n=n_items,
                                          features=item_features,
                                          population_name="item")

        print("INFO: creating user probability weights")
        # probability of getting a user given a previous user
        self._user_probability_weights = self.get_probability_weights(population=self.users,
                                                                      cache_name="user",
                                                                      distribution=users_distribution)

        print("INFO: creating item probability weights")
        # probability of getting a item given a previous item
        self._item_probability_weights = self.get_probability_weights(population=self.items,
                                                                      cache_name="item",
                                                                      distribution=items_distribution)

        # track times
        self._cooccurrence_time = 0
        self._sequential_time = 0
        self._observations_time = 0

        # to be updated each time we change observations
        self.__hash = tuple(self.observations_list).__hash__()
        self.__max_information = None
        self.__recommender_information = None

        # avoid computing twice the info for the same observations
        self.__last_max_information = self.__hash
        self.__last_recommender_information = self.__hash
                
        if timestamp:
            self._time_unites = 86400  # one day
        else:
            self._time_unites = 1

    def _reset_cooccurrences_matrices(self):
        # user has item
        self.user_item_present = np.zeros(self.n_users * self.n_items).reshape((self.n_users, self.n_items))
        # user does not have item
        self.user_item_absent = np.ones(self.n_users * self.n_items).reshape((self.n_users, self.n_items))
        # co-occurrence matrices
        self.items_cooccurrence11 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_cooccurrence10 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_cooccurrence01 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_cooccurrence00 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_llr = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.users_cooccurrence = np.zeros(self.n_users * self.n_users).reshape((self.n_users, self.n_users))

    def _reset_sequentials_matrices(self):
        self.items_sequentials11 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_sequentials01 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_sequentials10 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_sequentials00 = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))
        self.items_sequential_llr = np.zeros(self.n_items * self.n_items).reshape((self.n_items, self.n_items))

    def get_probability_weights(self, population, cache_name, distribution):
        """Just a wrapper to _make_probability_weights in case we don't use cache"""
        if self.read_cache_dir is not None:
            probability_weights = pickle.load(open(str(self.read_cache_dir) + "/" + str(population_name) +
                                              "/probability_weights.pickle", 'rb'))
            assert len(probability_weights) == len(population)
            return probability_weights
        else:
            return self._make_probability_weights(population, cache_name, distribution)

    def _make_probability_weights(self, population, cache_name, distribution):
        """Given an individual (eg users or items), get the probability of getting any
        other one according to a Zipf or uniform distribution.
        :par population: P(P1 | P2)
        :par cache_name: the name
        :par distribution: 'zipf' or 'uniform'
        :return List(List): [i][j] probability of j given i
        """

        n = len(population)

        if distribution == "zipf":
            # fill the blueprint for probability weights according to a powerlaw n->1/n (not normalized)
            probability_weights = {
                None: np.array([n/(i+1) for i in range(n)]).astype(np.float32)
            }
        elif distribution == "uniform":
            probability_weights = {
                None: np.ones(n, np.float32)
            }
        else:
            raise ValueError("'distribution' must be 'zipf' or 'uniform'.")

        for p in range(n):
            # ETA...
            if n > 100:
                steps = int(n/10)
                if p % steps == 0 and p != 0:
                    pc = p // steps 
                    print("INFO: %s0 percent of the probability weights produced (%s)" % (pc, p))

            probability_weights[p] = probability_weights[None].copy()
            # P(p_i | p_j) depends on the distance between p_i and p_j:
            for p2 in range(n):
                try:
                    sim = population[p]["similarities"][p2]
                except KeyError:
                    sim = np.float32(Simulator.get_similarity(population[p]["features"], population[p2]["features"]))
                    population[p]["similarities"][p2] = sim
                    population[p2]["similarities"][p] = sim
                # weights up for similar ones
                # big bias must force only very similar ones.
                # negative similarity gives no hope of buying
                # * len(pop) because when sell_and_tout it's incremented by 1
                if sim <= 0:
                    probability_weights[p][p2] = 0.0
                else:
                    # if bias is 0 nothing changes
                    # if bias is 1 prob goes from 0 (sim=0) to the original one (sim=1)
                    probability_weights[p][p2] = probability_weights[p][p2] * (1 - self.bias) + \
                                                 probability_weights[p][p2] * self.bias * sim
                if probability_weights[p][p2] < 0:
                    probability_weights[p][p2] = 0.0

        if self.save_cache_dir is not None:
            try:
                os.mkdir(self.save_cache_dir)
            except FileExistsError:
                raise FileExistsError("A cache directory with this name already exists, pls delete it")
            cache_file = str(self.save_cache_dir) + "/" + str(cache_name) + "/" + "probability_weights.pickle"
            pickle.dump(probability_weights, open(cache_file, 'wb'))

        return probability_weights

    def update_item_weights(self, item_id):
        """When an item is bought, its probability increases.
        We increase by 1 the bin of the correspondent item
        """
        for i in range(len(self.items)):
            self._item_probability_weights[i][item_id] += 1
        self._item_probability_weights[None][item_id] += 1
    
    @staticmethod
    def get_similarity(f1: List, f2: List) -> np.float64:
        """Return cosine similarity:
        1 = very similar
        0 = not at all (cosine is -1)
        """
        # Do 1 & 2 have some feature in common?
        norm1 = np.float32(np.linalg.norm(f1))
        norm2 = np.float32(np.linalg.norm(f2))
        if norm1 == 0.0 or norm2 == 0.0:
            return np.float32(0.0)
        else:
            sim = np.float32(np.dot(f1, f2) / (norm1*norm2))
            # floating errors makes abs(similarity) > 1.0
            sim = np.sign(sim) * min(abs(sim), 1)
            return (sim + 1.0) / 2.0

    def _random_user(self, previous_user=None) -> Tuple:
        """Get a random user who will buy something, considering that this
        user might be influenced by the user who previously bought something (through similarity
        in the weights)

        :par int previous_user: user who was in the previous observation (None if none)
        :return Tuple(user_id, p): new user_id and the probability we had to get it
        """
        weights = self._user_probability_weights[previous_user]
        user = random.choices(range(self.n_users), weights=weights)[0]
        p = np.float32(weights[user] / np.linalg.norm(weights))
        return user, p

    def _sell_and_tout(self, user_id, previous_item=None):
        """Sell an item and, if successful, increase the item probability of being sold
        #TODO/1 [weights] could be recomputed as a linear combination
        #TODO/1 of all weights taken not just from one previous_item,
        #TODO/1 but from all previous items, weighted with time (older weight less).
        #TODO/2 Il tout deve aumentare la probabilità del libro solo per gli utenti simili...
        :param int user_id: user who is buying (needed for no-reselling)
        :param int previous_item: item in the previous observation (if not item_id=None)
        :return Tuple(item_id, p):  item chosen and the probability we had to get this item
                                    (-1, 1) if the user has already bought all items...
        """
        weights = self._item_probability_weights[previous_item]  # get the probabilities for the items
        # sold items get weight=0
        if previous_item is not None:  # not first item ever sold
            sold = ([i_t[0] for i_t in self.user_buying_dict.get(user_id, [(None, None)])])
            # item_probability_weights[item_id] gives the prob given the past item
            # after having put to 0 all sold items, shift so that the most probable is the next one....
            weights = [w if i not in sold else 0 for i, w in enumerate(weights)]

        if sum(weights) == 0:  # user has bought every item
            return -1, 1
        else:
            try:
                new_item = random.choices(population=range(self.n_items), weights=weights)[0]
                if self.tout:
                    self.update_item_weights(new_item)
                p = np.float32(weights[new_item] / np.linalg.norm(weights))
                return new_item, p
            except:
                print("ERROR: Weights not valid for random.choices:\n", weights)
                raise BaseException

    def run(self, n_observations: int) -> None:
        """Create observations.

        * If a user hasn't bought anything yet, and is buying smt because
          influenced by the previous_user, it will buy an item similar
          to the one bought by such previous_user

        :param n_observations:
        :return: List[Dict]
        """
        # reset dict and list, otherwise we'll have inconsistent timestamps...
        # (but leave the probabilities)
        self.user_buying_dict = {}
        self.observations_list = []
        obs_done = 0
        over_buyers = set()  # buyers who bought all items
        n_warning = 0
        user = None
        # go till all users have bought all items or n_obs
        while obs_done <= n_observations and len(over_buyers) < self.n_users:
            # ETA...
            if n_observations > 100:
                steps = int(n_observations/10)
                if obs_done % steps == 0 and obs_done != 0:
                    pc = obs_done // steps
                    print("%s0 percent of the observations produced (%s)" % (pc, obs_done))

            # some warning if there are buyers who bought all items...
            if len(over_buyers) == 1 and n_warning == 0:
                print("Warning: user %s has bought all items" % list(over_buyers)[0])
                n_warning = 1
            elif len(over_buyers) == int(self.n_users*0.5) and n_warning == 1:
                print("Warning: 50% of users have bought all items")
                n_warning = 2
            elif len(over_buyers) >= self.n_users:
                print("All users have bought all items. Stopping here.")
                break

            # new user given the previous user:
            previous_user = user
            (user, p_u) = self._random_user(previous_user=user)
            # if user has already bought something, get the item considering the similarity
            # with the last bought item by this user.
            previous_item = self.user_buying_dict.get(user, [(None, None)])[-1][0]
            # If it's the first item and previous_user != None, be influenced by that user
            if previous_user is not None and previous_item is None:
                previous_item = self.user_buying_dict.get(previous_user, [(None, None)])[-1][0]
            (new_item, p_i) = self._sell_and_tout(user, previous_item)
            if new_item == -1:  # buyer bought all items
                over_buyers.add(user)
                continue
            item = new_item
            self.user_buying_dict.setdefault(user, []).append((item, obs_done*self._time_unites))
            self.observations_list.append(Observation(item_id=item,
                                                      item_p=p_i,
                                                      user_id=user,
                                                      user_p=p_u,
                                                      timestamp=obs_done*self._time_unites))
            
            obs_done += 1        

        self.__hash = tuple(self.observations_list).__hash__()  # new observations!
        self._observations_time = time.time()

    def best_kl(self):
        """The best possible KL divergence a recommender can get. This means a predictor
        which produces, for each user and observation, the actual probability
        distribution (weights), and tries to minimize the KL of this distribution
        with the one-hot of the label (the actual item in the obs)
        """
        pass

    def max_information(self):
        """TODO
        The actual information considering what we really know --that each observation depends only
        on the previous observation.

        Information is: uncertainty(no model) - uncertainty(knowing the model)

        uncertainty(no model) is log(n_items) + log(n_users), or the average surprise considering that
        each item/user has a probability 1/n of appearing

        in uncertainty(knowing the model), we take into account that we could compute the probability
        of the user being the one that bought an item, and the probability of that user buying that item.
        The surprise for each observation is therefore -log(user_p) -log(item_p). We sum and normalize...

        NB The model used is quite simplicistic, so it'd make no sense to try to model a real dataset
        with that, but --ideally-- giving to a NN a good number of train:observation and
        label:next_observation it should get this result if we ask for a dimensionality reduction
        compatible with the item/user_features
        """
        if self.__hash != self.__last_max_information:
            max_uncertainty = math.log(self.n_items) + math.log(self.n_users)
            print("max possible uncertainty ", max_uncertainty)
            tot_surprise = 0.0
            for o in self.observations_list:
                tot_surprise += -math.log(o["user_p"]) - math.log(o["item_p"])
            print("actual uncertainty ", tot_surprise / len(self.observations_list))
            self.__max_information = max_uncertainty - tot_surprise / len(self.observations_list)
            self.__last_max_information = self.__hash

        return self.__max_information

    def recommender_information(self):
        """TODO
        How much can we lower our uncertainty when knowing how the dataset was built?

        This is different from max_information, because in the recommender we only put
        when and what the user bought in their history.

        We know that for each set of observations, the uncertainty (average surprise, which
        is constant because we expect all books to have the same probability 1/N)
        is $log(N)$.

        What is the surprise for each buying when we know not what is the previous observation,
        but only previous observations _with the same user_?

        Once a user buys an item, the items' buying probability for that user in the future
        is bigger or smaller than 1/N, depending on a) how far this future is and b) how similar
        each book is to the one just bought.

        So, the probability of user u_i buying item b_k given their last buying:
        
        P(observation=n, user=u_i, item=b_k) = f(observation=n-j, user=u_i, item=b_l)

        --where observation n-j is the last buying by user u_i-- is hard to compute
        (NB there is "j" because u_i bought j-time ago, with many other transactions by other users
        in-between).

        (One could follow the probability of each step in the pattern P(n | n-1, n-2, ... , n-j),
        but that's not at all straightforward.)

        We therefore model P(n | n-j) with something like:

        $$ f(powerlaw(b_l)(1 + distance(b_k, b_l)*bias)*1/sqrt(j) ) $$

        The factor 1/sqrt(j) comes from two considerations:

        1. Heuristic, books bought long time ago are still influential to what we read today.
        2. The average mean-path goes with sqrt(j). That means that (assuming that distance
        among the items is uniformly distributed in bins of width d_m=max_distance/n) the
        possible items which originated the one we saw are the ones with distance [0, d_m],
        then [d_m, d_m*sqrt(2)], [d_m*s(2), d_m*s(3)], ... which goes of course down like 1/sqrt(i)

        That's the best a recommender should be able to get (as it will get as input the previous
        buyings of the user and not from the previous step).

        """
        if self.__hash != self.__last_recommender_information:
            print("INFO: Computing from scratch")
            max_uncertainty = math.log(self.n_items) + math.log(self.n_users)
            print("max recommender's possible uncertainty ", max_uncertainty)
            user_surprise = {}  # surprise for each user
            for user, item_times in self.user_buying_dict.items():
                print("USER ", user)
                for i, i_t in enumerate(item_times):
                    weights = self._item_probability_weights[None]
                    item = i_t[0]
                    p = weights[item] / np.linalg.norm(weights)
                    if user_surprise.get(user) is None:
                        assert i == 0, "Check this, it should be the first buy..."
                        user_surprise[user] = -math.log(p)
                    else:
                        assert i > 0, "Check this, it should not be the first buy..."
                        # p = f(powerlaw(b_l)(1 + distance(b_k, b_l)*bias)*1/sqrt(j) )
                        # print("P BEFORE ", p, Simulator.get_similarity(self.items[item_times[i-1][0]]["features"],
                        #                                                self.items[i_t[0]]["features"]),
                        #       ((i_t[1] - item_times[i-1][1])/self._time_unites))
                        p *= (1 + self.bias * Simulator.get_similarity(self.items[item_times[i-1][0]]["features"],
                                                                       self.items[i_t[0]]["features"])
                                            * 1./math.sqrt((i_t[1] - item_times[i-1][1])/self._time_unites))
                        user_surprise[user] -= math.log(p)
            min_uncertainty = np.sum(list(user_surprise.values())) / len(self.observations_list)
            print("min recommender's possible uncertainty ", min_uncertainty)
            self.__recommender_information = max_uncertainty - min_uncertainty
            self.__last_recommender_information = self.__hash
        else:
            print("INFO: Returning cached result")
        return self.__recommender_information

    def export(self, filenames=tuple(["is.csv", "us.csv", "os.csv"]), separator="\t", print_timestamp=True):
        """Save into three files (items, users, observations)

        :param filenames:
        :param separator:
        :param print_timestamp:
        :return:
        """
        with open(filenames[2], 'w') as f:
            f.write('user' + separator + str('item') + separator + 'timestamp' + "\n")
            for o in self.observations_list:
                if print_timestamp:
                    f.write(str(o["user_id"]) + separator + str(o["item_id"]) + separator + str(o["timestamp"]) + "\n")
                else:
                    f.write(str(o["user_id"]) + separator + str(o["item_id"]) + "\n")

        with open(filenames[1], 'w') as f:
            try:
                n_features = len(self._user_features)
            except TypeError:
                n_features = self._user_features
            f.write('user' + separator + separator.join(['feature_'+str(i) for i in range(n_features)]) + "\n")
            for ui, u in enumerate(self.users):
                try:
                    f.write(str(ui) + separator + separator.join([str(f) for f in u['features']]) + "\n")
                except TypeError:  # u['feature'] not iterable (0)
                    f.write(str(ui) + "\n")
        
        with open(filenames[0], 'w') as f:
            try:
                n_features = len(self._item_features)
            except TypeError:
                n_features = self._item_features
            f.write('item' + separator + separator.join(['feature_'+str(i) for i in range(n_features)]) + "\n")
            for ui, u in enumerate(self.items):
                f.write(str(ui) + separator + separator.join([str(f) for f in u['features']]) + "\n")

    def get_popular_items(self, maxi=20):
        pop = dict((i, 0) for i in range(self.n_items))
        for o in self.observations_list:
            pop[o["item_id"]] += 1
        return sorted(pop.items(), key=lambda t: t[1], reverse=True)[:maxi]

    def get_popular_users(self, maxi=20):
        pop = dict((i, 0) for i in range(self.n_users))
        for o in self.observations_list:
            pop[o["user_id"]] += 1
        return sorted(pop.items(), key=lambda t: t[1], reverse=True)[:maxi]

    def get_similar_items(self, item, maxi=20):
        sims = self.items[item]['similarities'].copy()
        del(sims[item])
        return sorted(sims.items(), key=lambda t: t[1], reverse=True)[:maxi]

    def get_similar_users(self, user):
        sims = self.users[user]['similarities'].copy()
        del(sims[user])
        return sorted(sims.items(), key=lambda t: t[1], reverse=True)

    def compute_cooccurrences(self):
        self._reset_cooccurrences_matrices()
        for o in self.observations_list:
            self.user_item_present[o['user_id']][o['item_id']] += 1
            self.user_item_absent[o['user_id']][o['item_id']] -= 1
        self.items_cooccurrence11 = np.matmul(self.user_item_present.transpose(), self.user_item_present)
        self.items_cooccurrence10 = np.matmul(self.user_item_present.transpose(), self.user_item_absent)
        self.items_cooccurrence01 = np.matmul(self.user_item_absent.transpose(), self.user_item_present)
        self.items_cooccurrence00 = np.matmul(self.user_item_absent.transpose(), self.user_item_absent)
        self.items_llr = root_loglikelihood_ratio(self.items_cooccurrence11,
                                                  self.items_cooccurrence01,
                                                  self.items_cooccurrence10,
                                                  self.items_cooccurrence00)
        self._cooccurrence_time = time.time()

    def compute_sequentials(self):
        self._reset_sequentials_matrices()

        start_time = time.time()
        counter = 0
        loops = len(self.observations_list)*self.n_items**2
        time_warned = False
        for u in self.user_buying_dict:
            if counter > 1000 and not time_warned:
                exp_time = loops * (time.time() - start_time) / counter
                print("Expected time in seconds: ", round(exp_time, 0))
                time_warned = True

            for j, item_time in enumerate(self.user_buying_dict[u]):
                item = item_time[0]
                if j > 0:
                    self.items_sequentials11[self.user_buying_dict[u][j-1][0]][item] += 1
                    for item2 in range(self.n_items):
                        if item2 != item:
                            self.items_sequentials10[self.user_buying_dict[u][j-1][0]][item2] += 1
                        if item2 != self.user_buying_dict[u][j-1][0]:
                            self.items_sequentials01[item2][item] += 1
                        for item3 in range(self.n_items):
                            if {item2, item3}.intersection({item, self.user_buying_dict[u][j-1][0]}) == set():
                                self.items_sequentials00[item2][item3] += 1
                            counter += 1
        print("Actual time in seconds: ", round(time.time() - start_time, 0))

        self.items_sequential_llr = root_loglikelihood_ratio(self.items_sequentials11,
                                                             self.items_sequentials01,
                                                             self.items_sequentials10,
                                                             self.items_sequentials00)

        self._sequential_time = time.time()

    def get_cooccurred_items(self, item, maxi=20):
        if self._observations_time > self._cooccurrence_time:
            self.compute_cooccurrences()
        cooc_items = self.items_cooccurrence11[item]
        oo = [(it, oc) for it, oc in enumerate(cooc_items)]
        return sorted(oo, key=lambda t: t[1], reverse=True)[:maxi]

    def get_llr_items(self, item, maxi=20):
        if self._observations_time > self._cooccurrence_time:
            self.compute_cooccurrences()
        llr_item = self.items_llr[item]
        oo = [(it, oc) for it, oc in enumerate(llr_item)]
        return sorted(oo, key=lambda t: t[1], reverse=True)[:maxi]

    def get_sequentials_llr_items(self, item, maxi=20):
        if self._observations_time > self._sequential_time:
            self.compute_sequentials()
        llr_item = self.items_sequential_llr[item]
        oo = [(it, oc) for it, oc in enumerate(llr_item)]
        return sorted(oo, key=lambda t: t[1], reverse=True)[:maxi]

    def export_similars(self, filenames=tuple(["similar_items.csv", "similar_users.csv"]), maxi=20):
        """

        :param filenames:
        :return:
        """
        with open(filenames[0], 'w') as f:
            for ni in range(self.n_items):
                f.write(str(ni) + "\t" +
                        ", ".join([str(i) for i in self.get_similar_items(ni)][:maxi]) + "\n"
                        )

        with open(filenames[1], 'w') as f:
            for ni in range(self.n_users):
                f.write(str(ni) + "\t" +
                        ", ".join([str(i) for i in self.get_similar_users(ni)][:maxi]) + "\n"
                        )

    @staticmethod
    def make_population(n: int, features, population_name) -> List[Dict]:
        """
        Values of features are distributed either continuously in [0, 1]
        or discreetly in [-1, 1].

        When discreet (users), each feature is independent on the others:
        features are like age, geo, gender etc.

        When continuous (items), features are shuffled randomly then the
        values' distribution follows the powerlaw. This because are seen
        like categories of books: each book is predominantly inside a category
        with other flavours.

        Users with some of the features having the same value
        have similar taste and similar items should be bought after a buying.

        :param int n: Number of individuals (items/users)
        :param (List[int] OR int) features:  or Number of features.
                List: [feature1_n_values, feature2... ]. feature_n_values in [-1, 1] are generated for each feature
                Int: N. N features uniformly distributed in [-1, 1]
        :param population_name "user" or "item" just for messaging
        :return List[Dict]: {"feature": Tuple, "similarity": {}}  # similarity with other individual will be filled later
        """
        population = []
        while len(population) < n:
            if n > 10000:
                steps = int(n/10)
                if len(population) % steps == 0 and len(population) != 0:
                    pc = len(population) // steps 
                    print("INFO: %s0 percent of the %s population produced (%s)" %
                          (pc, population_name, len(population)))

            if features == 0:
                f = 1
            else:
                # discreet, distributed in [-1, +1]:
                try:
                    f = np.array([2.0*random.randint(0, f-1)/(f-1)-1 for f in features]).astype(np.float32)
                # continuous:
                except TypeError:  # it's int...
                    # probability that a feature is chosen as principal (feature 0 is the most probable and
                    # down with Pareto
                    p_dist = np.array([1 / i for i in range(1, features+1)]) / sum([1 / i for i in range(1, features+1)])
                    # order of importance of features
                    features_order = np.random.choice(features, features, False, p_dist)
                    # some random values to give to each feature
                    features_values = np.random.random(features)
                    f = np.zeros(features)
                    for i, r in enumerate(f):
                        # now the most important feature, which is the first element
                        # in features_order, gets a bigger value and so on
                        f[features_order[i]] = features_values[i] / (i + 1)
            population.append({"features": f, "similarities": {}})
        return population
