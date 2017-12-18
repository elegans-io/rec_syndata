__author__ = 'mal'


class Observation(dict):
    def __init__(self, item_id, item_p, user_id, user_p, timestamp):
        super().__init__()
        self["item_id"] = item_id
        self["item_p"] = item_p
        self["user_id"] = user_id
        self["user_p"] = user_p
        self["timestamp"] = timestamp
        self.__hash__ = hash((item_id, item_p, user_id, user_p, timestamp))

    def __hash__(self):
        return self.__hash__

