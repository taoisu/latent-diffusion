import pickle

pickle_path = "/database/TrainSuperRes/flist.pkl"
with open(pickle_path, 'rb') as f:
    flist = pickle.load(f)
    count = 0
    for path_name in flist:
        count += 1
    print(f"count={count}")