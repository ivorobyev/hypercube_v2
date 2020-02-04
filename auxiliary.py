import numpy as np

def get_hash(letter):
    '''
    Return hash of amino acid letter
    '''
    return int(str(hash(letter)).replace('-','')[:5])

def generate_mutations_hash_map_and_letter_codes():
    '''
    Generate dictionaries of amino acid letters codes and mutations
    '''
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
            mut_hashmap[abs(get_hash(i) -  get_hash(j))] = mut
        
        z_mut = sorted((i,'Z'))
        mut_hashmap[abs(get_hash(i) -  get_hash(0))] = z_mut
            
    return mut_hashmap, letter_codes
        
def get_positions(mut_list):
    '''
    Get all possible positions with mutations
    '''
    positions = set()
    for a in mut_list:
        genotype = a.split(':')
        genotype_pos = list(map(lambda x: int(x[:-1]), genotype))
        for a in genotype_pos:
            positions.add(a)
    return positions

def get_position_order_dict(positions):
    '''
    Return dictionary with positions and numerical order
    '''
    pos_order = {}
    for index, key in enumerate(sorted(positions)):
        pos_order[index] = key
        
    return pos_order
        
def encode_genotype(genotype, position_order):
    '''
    Convert genotype string to vector
    '''
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

def get_hypercubes(genotypes, dimensions = 0):
    '''
    Get all possible hypercubes for given genotypes
    '''
    hypercubes = []
    ln = int(len(genotypes[0]) / 2) if dimensions != 1 else len(genotypes[0])
    genotypes_count = len(genotypes)
    distance_value = dimensions if dimensions == 1 else 2
    
    for i in range(genotypes_count):
        diagonal = abs(np.array(genotypes[i][:ln]) - np.array(genotypes[i][-ln:]))
        for j in range(i+1, genotypes_count):

            prev_diagonal = abs(np.array(genotypes[j][:ln]) - np.array(genotypes[j][-ln:]))

            if not np.array_equal(diagonal, prev_diagonal):
                break
                
            dst = abs(np.array(genotypes[j]) - np.array(genotypes[i]))  
            if (np.count_nonzero(dst) == distance_value) and np.array_equal(dst[:ln],dst[-ln:]):
                hypercube = sorted([genotypes[j][:ln],
                                    genotypes[j][-ln:],
                                    genotypes[i][:ln],
                                    genotypes[i][-ln:]])
                hp = hypercube[-1] + hypercube[0]
                hypercubes.append((tuple(abs(np.array(dst[:ln]) - np.array(prev_diagonal))),tuple(hp)))
    return sorted(set(hypercubes))

def decode_hypercube(hypercube, mut_hashmap, letter_codes, pos_order):
    '''
    Decode genotypes from vectors to string
    '''
    decoded_array = []
    ln = int(len(hypercube[1]) / 2)
    for index, value in enumerate(hypercube[0]):
        states = sorted([mut_hashmap[value][0], mut_hashmap[value][1]])
        if value != 0:
            decoded_array.append('{0}{1}{2}'.format(states[0], pos_order[index], states[1]))
    
    first_genotype = []
    for index, a in enumerate(hypercube[1][:ln]):
        if a != 0:
            first_genotype.append(str(pos_order[index])+letter_codes[a])
            
    last_genotype = []
    for index, a in enumerate(hypercube[1][-ln:]):
        if a != 0:
            last_genotype.append(str(pos_order[index])+letter_codes[a])
    
    diagonal = ':'.join(sorted(decoded_array))
    
    first_genotype = ':'.join(first_genotype) if len(first_genotype) > 0 else '0Z'
    last_genotype = ':'.join(last_genotype) if len(last_genotype) > 0 else '0Z'
    
    return ' '.join((diagonal, first_genotype, last_genotype))
    