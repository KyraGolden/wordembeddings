import io
import gzip

from gensim.models.fasttext import FastText
import gensim
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt




def find_synonyms(fname, word):
    # lines = [x.decode('utf8').strip() for x in fname.readlines()]
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    # print(fin.readline().split())
    data = {}
    # for line in fin:
    counter = 0
    while counter < 5000:
        line = fin.readline()
        line = line.strip()  # wichtig, da sonst line break zur letzen ziffer ist und dies die Zahlen verfälscht
        # print(line)
        # tokens = line.rstrip().split(' ')
        tokens = str(line).split(' ')
        # print('Tokens: ')
        # print(tokens)
        # das untere geht viel länger als das obere
        # data[tokens[0]] = map(float, tokens[1:])
        data[tokens[0]] = tokens[1:]
        # print('Data: ')
        # print(data)
        counter += 1
        # print("Cosine similarity to " + word + ": ")
    distances = {}
    X = np.empty((0, 300))
    for item in data:
        # print(item)
        if item.lower() == word.lower():
            pass
        else:
            # Bsp. Königx
         try:
            distance = metrics.pairwise.cosine_distances([data[item]], [data[word]])
            if 0.3 < distance < 0.5:
                distances[item] = distance
                # save the n vectors
                vec = np.array(data[item], dtype=float)
                X = np.append(X, [vec], axis=0)
         except ValueError:
             pass


    #add original vector to array
    help2 = np.array(data[word], dtype=float)
    X = np.append(X,[help2],axis = 0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(X)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)
    X = np.append(X, [help2], axis=0)
    listofkeys = list(distances.keys())
    listofkeys.append(word)
    #for label, x, y in zip(listofkeys.format(y), x_coords, y_coords):
    for i, txt in enumerate(listofkeys):
        plt.annotate(txt, (x_coords[i], y_coords[i]))
        #plt.annotate(label,
        #             xy=(x, y),
        #            xytext=(0, 0),
        #           textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
    plt.savefig('vectors.png')

    key_min = min(distances.keys(), key=(lambda k: distances[k]))
    print('Min. Distanz')
    print(key_min)
    print(distances[key_min])
    return (key_min)


if __name__ == "__main__":
    find_synonyms("../../../short.cc.de.300.vec", 'König')
