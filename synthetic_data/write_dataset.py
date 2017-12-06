from typing import List, Tuple

# Created by Mario Alemi 29 November 2017

def write_dataset(users: List, n_items: int, n_categories, observations: List[Tuple],
                  ucsv: str = "us.csv", icsv: str = "is.csv", ocsv: str = "os.csv", separator: str = "\t") -> None:
    """
    u.csv:  user_id  | user_feature1  | ...  | user_featureN 
    i.csv:  item_id  | category
    o.csv:  user_id  | item_id  | timestamp
    """
    with open(ucsv, 'w') as f:
        n_features = len(users[0])
        f.write('user' + separator + separator.join(['feature_'+str(i) for i in range(n_features)]) + "\n")
        for ui, u in enumerate(users):
            f.write(str(ui) + separator + separator.join([str(f) for f in u]) + "\n")

    with open(icsv, 'w') as f:
        f.write('item' + separator + 'category' + "\n")
        for i in range(0, n_items):
            f.write(str(i) + separator + str(i % n_categories) + "\n")

    with open(ocsv, 'w') as f:
        f.write('user' + separator + str('item') + separator + 'timestamp' + "\n")
        for u in observations.keys():
            for o in observations[u]:
                f.write(str(u) + separator + str(o[0]) + separator + str(o[1]) + "\n")

    
