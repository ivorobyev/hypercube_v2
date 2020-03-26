import time 
import numpy as np
import auxiliary as aux
from multiprocessing import Pool
import argparse
import os

if __name__ == '__main__':  
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--genotypes', help='the filename with genotypes')
    parser.add_argument('-d', '--folder', help='name of non-existing folder to store intermediate and result files, '
                                               '"hypercubes" by default', default='hypercubes')
    args = parser.parse_args()

    if args.folder.strip() == '':
        print('ERROR: Folder "{0}" contains only whitespaces, give me valid name'.format(args.folder))
        exit(1)
    if os.path.exists(args.folder):
        print('ERROR: Folder/file "{0}" exists, run again with another folder name'.format(args.folder))
        exit(1)
    try:
        os.mkdir(args.folder)
    except:
        print('ERROR: Unable to create folder {0}'.format(args.folder))
        exit(1)

    mut_hashmap, letter_codes = aux.generate_mutations_hash_map_and_letter_codes()

    start_time = time.time()
    mut = []
    with open(args.genotypes, 'r') as f:
        f.readline()
        for line in f:
            mut.append(line.split('\t')[0])
        pos = aux.get_positions(mut)
        pos_order = aux.get_position_order_dict(pos) 

    genotypes_vectors = []
    for ind, a in enumerate(mut):
        a = '0Z' if a == '' else a
        try:
            aux.check_input(a, ind)
        except NameError as err:
            print(err)
            exit()
        genotypes_vectors.append(aux.encode_genotype(a, pos_order))
    
    print('Generate hypercubes for dimension 1')
    hp = aux.get_next_hypercubes_dim_one(genotypes_vectors, args.folder)
    decoded = list(map(lambda x: aux.decode_hypercube(x, mut_hashmap, letter_codes, pos_order), hp))
    aux.write_to_file(1, decoded, args.folder)
    print('Done, hypercubes:{0}'.format(len(hp)))
    dims = 2
    while True:
        print('Generate hypercubes for dimension {0}'.format(dims))
        hp = aux.get_next_hypercubes(hp)
        decoded = list(map(lambda x: aux.decode_hypercube(x, mut_hashmap, letter_codes, pos_order), hp))
        aux.write_to_file(dims, decoded, args.folder)
        del decoded
        print('Done, hypercubes:{0}'.format(len(hp)))
        if len(hp) == 0:
            break
        dims += 1
    print('Elapsed time: {0}'.format(time.time() - start_time))