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
            for ui, u in enumerate(users):
                    f.write(str(ui) + "\t" + separator.join([str(f) for f in u]) + "\n")

    with open(icsv, 'w') as f:
            for i in range(0, n_items):
                    f.write(str(i) + separator + str(i % n_categories) + "\n")

    with open(ocsv, 'w') as f:
            for o in observations:
                    f.write(str(o[0]) + "\t" + str(o[1]) + "\t" + str(o[2]) + "\n")                                

    
