import numpy as np
import os
from multiprocessing import Pool

def check_input(line, rownumber):
    """Check input rows for valid format"""
    amino_acids_list = ['G', 'A', 'V', 'L', 
                        'I', 'M', 'F', 'W', 
                        'P', 'S', 'T', 'C', 
                        'Y', 'N', 'Q', 'D', 
                        'E', 'K', 'R', 'H', 'Z']
    frames = line.split(':')

    for a in frames:
        if a[:-1].isdigit() & (str(a[-1]).upper() in amino_acids_list):
            continue
        else:
            raise NameError('ERROR: invalid input format at line: {0}'.format(rownumber+2))

def get_hash(letter):
    return int(str(hash(letter)).replace('-','')[:5])

def generate_mutations_hash_map_and_letter_codes():
    mut_hashmap = {}
    letter_codes = {}
    letters = 'ARNDCEQGHILKMFPSTWYV'
    ln = len(letters)
    
    letter_codes[get_hash(0)] = 'Z'
    for letter in letters:
        letter_codes[get_hash(letter)] = letter
        
    for i in letters:
        for j in letters:
            mut = sorted((i,j))
            mut_hashmap[np.bitwise_xor(get_hash(i),get_hash(j))] = mut
        
        z_mut = sorted((i,'Z'))
        mut_hashmap[abs(get_hash(i) - get_hash(0))] = z_mut
            
    return mut_hashmap, letter_codes
        
def get_positions(mut_list):
    positions = set()
    for a in mut_list:
        a = '0Z' if a == '' else a
        genotype = a.split(':')
        genotype_pos = list(map(lambda x: int(x[:-1]), genotype))
        for a in genotype_pos:
            positions.add(a)
    return positions

def get_position_order_dict(positions):
    pos_order = {}
    for index, key in enumerate(sorted(positions)):
        pos_order[index] = key
        
    return pos_order
        
def encode_genotype(genotype, position_order):
    genotype_s = genotype.split(':')
    genotype_splitted = {int(x[:-1]) : x[-1] for x in genotype_s}
    genotype_positions = list(map(lambda x: int(x[:-1]), genotype_s))
    genotype_enc = []
    for pos in sorted(position_order.values()):
        if pos not in genotype_splitted.keys():
            genotype_enc.append(0)
        else:
            if genotype_splitted[pos] != 'Z':
                genotype_enc.append(get_hash(genotype_splitted[pos]))
            else:
                genotype_enc.append(0)
    
    return genotype_enc

def decode_hypercube(hypercube, mut_hashmap, letter_codes, pos_order):
    decoded_array = []
    l = int(len(hypercube) / 2)
    for index, value in enumerate(np.bitwise_xor(hypercube[:l], hypercube[l:])):
        states = sorted([mut_hashmap[value][0], mut_hashmap[value][1]])
        if value != 0:
            decoded_array.append('{0}{1}{2}'.format(states[0], pos_order[index], states[1]))
    
    first_genotype = []
    for index, a in enumerate(hypercube[:l]):
        if a != 0:
            first_genotype.append(str(pos_order[index])+letter_codes[a])
            
    last_genotype = []
    for index, a in enumerate(hypercube[l:]):
        if a != 0:
            last_genotype.append(str(pos_order[index])+letter_codes[a])
    
    diagonal = ':'.join(sorted(decoded_array))
    
    first_genotype = ':'.join(first_genotype) if len(first_genotype) > 0 else '0Z'
    last_genotype = ':'.join(last_genotype) if len(last_genotype) > 0 else '0Z'
    
    return ' '.join((diagonal, first_genotype, last_genotype))

def init(vv, cc):
    global vectors, c
    vectors = vv
    c = cc

def get_distance_matrix_multiplication(j):
     c[:,j] = np.sum(vectors[j]!=vectors, axis = 1)

def get_next_hypercubes_dim_one(vectors, folder, cores):
    vectors = np.array(vectors)
    l = len(vectors[0])
    c = np.memmap(folder+'/hypercubes_tmp', dtype='int32', mode='w+', shape = (len(vectors), len(vectors)))

    p = Pool(processes=cores, initializer=init, initargs=(vectors, c))
    p.map(get_distance_matrix_multiplication, range(len(vectors)))

    inds = np.where(c == 1)
    del c
    os.remove(folder+'/hypercubes_tmp')

    unar_dist = {tuple(sorted([inds[0][i], inds[1][i]])) for i in range(len(inds[0]))}
    next_hp = []
    for a in unar_dist:
        next_hp.append(tuple(np.concatenate((vectors[a[0]], vectors[a[1]]))))
        
    return np.array(sorted(next_hp, key = lambda s: tuple(np.bitwise_xor(s[:l],s[l:]))))

def get_next_hypercubes(vectors):
    vectors = np.array(vectors)
    l = int(len(vectors[0])/2)
    
    next_hp = set()
    inds = []
    for i in range(len(vectors) - 1):
        inds.append(i)
        
        if tuple(np.bitwise_xor(vectors[i][:l], vectors[i][l:])) == tuple(np.bitwise_xor(vectors[i+1][:l], vectors[i+1][l:])):
            continue
            
        if len(inds) != 0:
            diag_vectors = []
            for i in inds:
                diag_vectors.append(vectors[i])
            
            diag_vectors = np.array(diag_vectors)
                
            s1 = np.sum(diag_vectors[:,l:][...,np.newaxis]!=diag_vectors[:,l:].T[np.newaxis,...], axis = 1)
            s1 = np.where(s1!=1, 0, s1)

            s2 = np.sum(diag_vectors[:,:l][...,np.newaxis]!=diag_vectors[:,:l].T[np.newaxis,...], axis = 1)
            s2 = np.where(s2!=1, 0, s2)

            c = s1 + s2
            indx = np.where(c == 2)
            del s1, s2, c

            unar_dist = {tuple(sorted([indx[0][i], indx[1][i]])) for i in range(len(indx[0]))}
            
            for a in set(unar_dist):
                hypercube = sorted([tuple(diag_vectors[a[0]][:l]), 
                                    tuple(diag_vectors[a[0]][l:]), 
                                    tuple(diag_vectors[a[1]][:l]), 
                                    tuple(diag_vectors[a[1]][l:])])

                short_hp = hypercube[-1] + hypercube[0]
                next_hp.add(tuple(short_hp))
            inds = []
    return np.array(sorted(next_hp, key = lambda s: tuple(np.bitwise_xor(s[:l],s[l:]))))

def write_to_file(dims, hypercubes, folder):
    f = open(folder+"/hypercubes"+str(dims)+".txt", "w")
    for hp in hypercubes:
        f.write(str(hp) +'\n')
    f.close()
