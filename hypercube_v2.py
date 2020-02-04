import time 
import numpy as np
import auxiliary as aux
from multiprocessing import Pool

if __name__ == '__main__':  
    start_time = time.time()
    mut_hashmap, letter_codes = aux.generate_mutations_hash_map_and_letter_codes()

    start_time = time.time()
    mut = []
    with open('S7_small.txt', 'r') as f:
        f.readline()
        for line in f:
            mut.append(line.split('\t')[0])
        pos = aux.get_positions(mut)
        pos_order = aux.get_position_order_dict(pos)

    genotypes_vectors = []
    for a in mut:
        genotypes_vectors.append(aux.encode_genotype(a, pos_order))
        
    print('generate hypercubes for dimension 1')
    print()
    hp = aux.get_hypercubes(genotypes_vectors, 1)
    decoded = list(map(lambda x: aux.decode_hypercube(x, mut_hashmap, letter_codes, pos_order), hp))
    print('decoded')
        
    dims = 2

    while True:
        print('generate hypercubes for dimension {0}'.format(dims))
        
        new_hp = np.array_split(np.array(list(hp))[:,1], 100)
        next_hypercubes  = []
        
        pool = Pool(1)
        next_hypercubes = pool.map(aux.get_hypercubes, new_hp)
        pool.close()
        pool.join()

        next_hypercubes = [hp for diagonal in next_hypercubes for hp in diagonal] 
        
        if len(next_hypercubes) == 0:
            print('No hypercubes for dimension {0}'.format(dims))
            print('Done')
            break
            
        dims += 1
        hp = sorted(set(next_hypercubes))
        decoded = list(map(lambda x: aux.decode_hypercube(x, mut_hashmap, letter_codes, pos_order), hp))
        print('decoded')
        
    print(time.time() - start_time)