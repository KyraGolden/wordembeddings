#!/usr/bin/python3

import cgi
import cgitb
import io
import json
import pickle
import gzip
#from gensim.models.fasttext import FastText
from sklearn import metrics
import numpy as np
cgitb.enable(display=0, logdir="/logs")

def application(environ, start_response):
    params = cgi.FieldStorage(fp=environ['wsgi.input'], environ=environ)
    word = params["w"].value
    nr =  params["nr"].value
    searchType = params["st"].value
    if searchType == 'bigSearch':
        synonym = find_synonyms("/home/pcl3_03/public_html/wordembeddings/50000.cc.de.300.vec", word, int(nr))
    else:
        synonym = find_synonyms("/home/pcl3_03/public_html/wordembeddings/10000.cc.de.300.vec", word, int(nr))
    output = {}
    output["syn1"] = str(synonym)
    output_js = json.dumps(output)
    output = "<h2> Here are " + nr + " synonyms for " + word + ":</h2><br><p>" + synonym + "</p>"
    status = "200 OK"
    output_js = output_js.encode()
    response_headers = [("Content-type", "application/json; charset=UTF-8"),
                        ("Content-Length", str(len(output_js)))]
    start_response(status, response_headers)
    return [output_js]

def find_synonyms(fname, word, anzahl):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        # line = fin.readline()
        tokens = str(line).split(' ')
        data[tokens[0]] = tokens[1:]
    #word_vecs = open('long_word_vecs.pkl', 'rb')
    #data = pickle.load(word_vecs)
        #with open('word_vecs.pkl', 'rb') as word_vecs:
        #    data = pickle.load(word_vecs)
    # die pickle-Datei sollte eigentlich schon zur Verfügung stehen, sollte das nicht der Fall sein, wird sie hier gleich
    # erstellt für den nächsten Aufruf.

    distances = {}
    out_list = []
    for item in data:
        if item == word:
            pass
        else:
            try:
                distances[item] = metrics.pairwise.cosine_distances([data[item]], [data[word]])
            except KeyError:
                #print("Dieses Wort befindet sich nicht in der Datenbank")
                return "Dieses Wort befindet sich nicht in der Datenbank"
            except ValueError:
                pass
    ctr = 0
    while ctr < anzahl:
        key_min = min(distances.keys(), key=(lambda k: distances[k]))
        out_list.append(key_min)
        del distances[key_min]
        ctr += 1
    seperator = ', '
    print(seperator.join(out_list))
    return seperator.join(out_list)
    #return(key_min)

if __name__ == "__main__":
    find_synonyms("../../../short.cc.de.300.vec", 'Blau', 2)