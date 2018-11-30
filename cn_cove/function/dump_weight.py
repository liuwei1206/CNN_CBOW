__author__ = "liuwei"

"""
dump_weight to h5py files
note that we only dump the weight that is trainable, untrainable parameters will not
be saved
"""
import h5py
from ..data.vocab import words_from_vocab

def dump_weights(model, file_path):
    """
    dump the weights of the model to h5py files
    Args:
        model: the obj of the model
        file_path: file path to save the model weights
    """
    # get all params, and then Traversing all the param
    model_name = type(model).__name__
    params = model.state_dict()

    with h5py.File(file_path, 'w') as fout:
        # each model corresponding to a group
        group = fout.create_group(model_name)

        for k, v in params.items():
            print(k)
            shape = list(v.size())
            dset = group.create_dataset(k, shape, dtype="float32")
            values = v.cpu().numpy()
            dset[...] = values


def load_weights(file_path):
    """
    load the model weight from hdf5 file
    Args:
        file_path: the path of the hdf5 file
    """
    with h5py.File(file_path, 'r') as fout:
        for key in fout.keys():
            print(fout[key].name)

            group = fout[fout[key].name]
            for k in group.keys():
                print(group[k].name)
                print(group[k].value.shape)


def dump_embedding(model, words, embed_file):
    """
    dump the word embedding from the word, to do some test, create a txt file
    Args:
        model: the trained model
        embed_file: the file path of embedding
    :return:
    """
    params = model.state_dict()

    for k, v in params.items():
        if "_embed._embed" in k:
            values = v.cpu().numpy()
            shape = values.shape
            print("word embedding's shape is: " + str(shape))

            write_to_txt(values, words, embed_file)

def write_to_txt(values, words, file):
    """
    write values into text file. note that when we transform words to ids, we add 1 for
    all word_idx to do mask, so we need - 1 to let word coressponding to idx
    Args:
        values: the embedding values
        words: the word of str
        type: the write type, where 1 for write word embedding
    """
    shape = values.shape
    assert shape[0] == len(words)

    with open(file, 'w') as f:
        for i in range(shape[0]):
            """
            each word and corresponding one line
            """
            f.write(words[i] + ' ')

            word_embed = [str(item) for item in values[i]]
            word_embed = ",".join(word_embed)
            f.write(word_embed)
            f.write("\n")





