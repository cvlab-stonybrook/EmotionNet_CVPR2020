import pickle as pkl

def save2pickle(save_file, save_data):
    with open(save_file, "wb") as of_:
        pkl.dump(save_data, of_, pkl.HIGHEST_PROTOCOL)


def loadpickle(save_file):
    with open(save_file, "rb") as of_:
        return  pkl.load(of_)

def loadpickle_python2_compatible(save_file):
    with open(save_file, "rb") as of_:
        return pkl.load(of_, encoding='latin1')