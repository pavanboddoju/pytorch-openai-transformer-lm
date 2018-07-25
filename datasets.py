import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def get_embeddings(path):
    with open(path) as f:
        f = csv.reader(f)
        embeddings_list = []
        embeddings = []        
        for row in f:
            a = [float(s) for s in list(row)[2:]]
            embeddings.append(a)
            if len(embeddings) == 3:
                embeddings_list.append(embeddings)
                embeddings = []
        return embeddings_list

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))

    embeddings = get_embeddings(os.path.join(data_dir, 'cloze_elmo_val_compiled.csv'))
    teELMo = get_embeddings(os.path.join(data_dir, 'cloze_elmo_test_compiled.csv'))

    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys, tr_elmo, va_elmo = train_test_split(storys, comps1, comps2, ys, embeddings, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    trELMo = []

    for s, c1, c2, y, elmo in zip(tr_storys, tr_comps1, tr_comps2, tr_ys, tr_elmo):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)
        trELMo.append(elmo)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    vaELMo = []

    for s, c1, c2, y, elmo in zip(va_storys, va_comps1, va_comps2, va_ys, va_elmo):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
        vaELMo.append(elmo)

    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    
    trELMo = np.asarray(trELMo, dtype=np.float32)
    vaELMo = np.asarray(vaELMo, dtype=np.float32)
    teELMo = np.asarray(teELMo, dtype=np.float32)

    return (trX1, trX2, trX3, trY, trELMo), (vaX1, vaX2, vaX3, vaY, vaELMo), (teX1, teX2, teX3, teELMo)
