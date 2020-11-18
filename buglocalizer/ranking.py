import json
from datasets import DATASET
import numpy as np

def main():
    # file_ = 'module3_' + str(alpha) + '.json'
    # with open(DATASET.root / 'rvsm_similarity.json', 'rb') as file:
    with open(DATASET.root / 'vsm_similarity.json', 'rb') as file:
        module = json.load(file)
    out = []
    for m in module:
        temp = np.argsort(m).tolist()
        temp.reverse()
        out.append(temp)
    
    with open(DATASET.root / 'ranking.json', 'w') as file:
        json.dump(out, file)

def main_normal(alpha):
    file_ = 'module3_normal_' + str(alpha) + '.json'
    with open(DATASET.root / file_, 'rb') as file:
        module = json.load(file)
    
    out = []
    for m in module:
        temp = np.argsort(m).tolist()
        temp.reverse()
        out.append(temp)
    
    file_name = 'ranking_normal_' + str(alpha) +  '.json'
    with open(DATASET.root / file_name, 'w') as file:
        json.dump(out, file)

if __name__ == '__main__':
    main()

    # myList = [1, 2, 3, 100, 5]
    # print(np.argsort(myList).tolist().reverse())