

import numpy as np
import random
from compression import CSR, BCSR, List_of_Lists, COO


def generate_sparse_matrix(size, density):
    matrix = np.zeros((size, size))
    nnzs = int(size * size * density)
    positions = random.sample(range(size * size), nnzs)

    for pos in positions:
        row = pos // size
        col = pos % size
        matrix[row][col] = 2
    
    return matrix

matrices = []
densities = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
size = 1024

for density in densities:
    matrix = generate_sparse_matrix(size, density)
    matrices.append(matrix)

for matrix in matrices:
    csr = CSR(matrix.tolist())
    bcsr = BCSR(matrix.tolist(), block_size=8)
    lil = List_of_Lists(matrix.tolist())
    coo = COO(matrix.tolist())

    data_csr, csr_steps = csr.decompress()
    data_bcsr, bcsr_steps = bcsr.decompress()
    data_lil, lil_steps = lil.decompress()
    data_coo, coo_steps = coo.decompress()

    assert np.array_equal(np.array(data_csr), matrix), "CSR incorrect"
    assert np.array_equal(np.array(data_bcsr), matrix), "BCSR incorrect"
    assert np.array_equal(np.array(data_lil), matrix), "LIL incorrect"
    assert np.array_equal(np.array(data_coo), matrix), "COO incorrect"


storage_overheads = {
    'CSR': [],
    'BCSR': [],
    'LIL': [],
    'COO': []
}

decompression_times = {
    'CSR': [],
    'BCSR': [],
    'LIL': [],
    'COO': []
}

for matrix in matrices:
    csr_overheads = []
    bcsr_overheads = []
    lil_overheads = []
    coo_overheads = []
    csr_times = []
    bcsr_times = []
    lil_times = []
    coo_times = []

    for i in range(0, size, 64):
        submatrix = matrix[i:i+64, i:i+64]

        non_zero_count = np.count_nonzero(submatrix)
        if non_zero_count == 0:
            continue

        csr = CSR(submatrix.tolist())
        bcsr = BCSR(submatrix.tolist(), block_size=8)
        lil = List_of_Lists(submatrix.tolist())
        coo = COO(submatrix.tolist())

        data_csr, csr_steps = csr.decompress()
        data_bcsr, bcsr_steps = bcsr.decompress()
        data_lil, lil_steps = lil.decompress()
        data_coo, coo_steps = coo.decompress()

        csr_times.append(csr_steps)
        bcsr_times.append(bcsr_steps)
        lil_times.append(lil_steps)
        coo_times.append(coo_steps)

        csr_overheads.append(csr.get_storage_overhead())
        bcsr_overheads.append(bcsr.get_storage_overhead())
        lil_overheads.append(lil.get_storage_overhead())
        coo_overheads.append(coo.get_storage_overhead())

    storage_overheads['CSR'].append(np.mean(csr_overheads))
    storage_overheads['BCSR'].append(np.mean(bcsr_overheads))
    storage_overheads['LIL'].append(np.mean(lil_overheads))
    storage_overheads['COO'].append(np.mean(coo_overheads))
    decompression_times['CSR'].append(np.mean(csr_times))
    decompression_times['BCSR'].append(np.mean(bcsr_times))
    decompression_times['LIL'].append(np.mean(lil_times))
    decompression_times['COO'].append(np.mean(coo_times))

def read_file(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()

        # print(filename)
        num_rows, num_cols, nnzs = map(int, line.strip().split())

        if num_rows % 64 != 0:
            num_rows += 64 - (num_rows % 64)
        if num_cols % 64 != 0:
            num_cols += 64 - (num_cols % 64)

        matrix = np.zeros((num_rows, num_cols))
        for line in f:
            if filename == 'barth.mtx' or filename == 'crack.mtx':
                r, c = line.strip().split()
                val = 1.0
            else:   
                r, c, val = line.strip().split()
            r = int(r) - 1
            c = int(c) - 1
            val = float(val)
            matrix[r][c] = val

        return matrix
    

filenames = ['rbsb480.mtx', 'init_adder1.mtx', 'barth.mtx', 'kineticBatchReactor_4.mtx', 'mark3jac020.mtx', 'crack.mtx', 'poli3.mtx', 'mixtank_new.mtx', 'invextr1_new.mtx', 'g7jac140sc.mtx']
# filenames = ['rbsb480.mtx', 'init_adder1.mtx', 'barth.mtx', 'kineticBatchReactor_4.mtx', 'mark3jac020.mtx', 'crack.mtx', 'poli3.mtx']

suitsparse_matrices = []
for filename in filenames:
    matrix = read_file(filename)
    suitsparse_matrices.append(matrix)

for matrix in suitsparse_matrices:
    csr_overheads = []
    bcsr_overheads = []
    lil_overheads = []
    coo_overheads = []
    csr_times = []
    bcsr_times = []
    lil_times = []
    coo_times = []

    for i in range(0, matrix.shape[0], 64):
        submatrix = matrix[i:i+64, i:i+64]

        non_zero_count = np.count_nonzero(submatrix)
        if non_zero_count == 0:
            continue


        csr = CSR(submatrix.tolist())
        bcsr = BCSR(submatrix.tolist(), block_size=8)
        lil = List_of_Lists(submatrix.tolist())
        coo = COO(submatrix.tolist())

        data_csr, csr_steps = csr.decompress()
        data_bcsr, bcsr_steps = bcsr.decompress()
        data_lil, lil_steps = lil.decompress()
        data_coo, coo_steps = coo.decompress()

        csr_times.append(csr_steps)
        bcsr_times.append(bcsr_steps)
        lil_times.append(lil_steps)
        coo_times.append(coo_steps)

        csr_overheads.append(csr.get_storage_overhead())
        bcsr_overheads.append(bcsr.get_storage_overhead())
        lil_overheads.append(lil.get_storage_overhead())
        coo_overheads.append(coo.get_storage_overhead())

    storage_overheads['CSR'].append(np.mean(csr_overheads))
    storage_overheads['BCSR'].append(np.mean(bcsr_overheads))
    storage_overheads['LIL'].append(np.mean(lil_overheads)) 
    storage_overheads['COO'].append(np.mean(coo_overheads))
    decompression_times['CSR'].append(np.mean(csr_times))
    decompression_times['BCSR'].append(np.mean(bcsr_times))
    decompression_times['LIL'].append(np.mean(lil_times))
    decompression_times['COO'].append(np.mean(coo_times))



import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))
plt.subplot(2, 2, 1)
plt.plot(densities, storage_overheads['CSR'][:10], marker='o', label='CSR')
plt.plot(densities, storage_overheads['BCSR'][:10], marker='o', label='BCSR')
plt.plot(densities, storage_overheads['LIL'][:10], marker='o', label='LIL')
plt.plot(densities, storage_overheads['COO'][:10], marker='o', label='COO')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Density')
plt.ylabel('Storage Overhead')
plt.title('Figure 1(a): Storage Overhead vs. Density Synthetic')
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 2)
plt.plot(densities, decompression_times['CSR'][:10], marker='o', label='CSR')
plt.plot(densities, decompression_times['BCSR'][:10], marker='o', label='BCSR')
plt.plot(densities, decompression_times['LIL'][:10], marker='o', label='LIL')
plt.plot(densities, decompression_times['COO'][:10], marker='o', label='COO')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Density')
plt.ylabel('# Steps')
plt.title('Figure 2(a): Decompression Latency vs. Density Synthetic')
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot(filenames, storage_overheads['CSR'][10:], marker='o', label='CSR')
plt.plot(filenames, storage_overheads['BCSR'][10:], marker='o', label='BCSR')
plt.plot(filenames, storage_overheads['LIL'][10:], marker='o', label='LIL')
plt.plot(filenames, storage_overheads['COO'][10:], marker='o', label='COO')
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.xlabel('SuiteSparse Matrices')
plt.ylabel('Storage Overhead')
plt.title('Figure 1(b): Storage Overhead vs. Density SuiteSparse')
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(filenames, decompression_times['CSR'][10:], marker='o', label='CSR')
plt.plot(filenames, decompression_times['BCSR'][10:], marker='o', label='BCSR')
plt.plot(filenames, decompression_times['LIL'][10:], marker='o', label='LIL')
plt.plot(filenames, decompression_times['COO'][10:], marker='o', label='COO')
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.xlabel('SuiteSparse Matrices')
plt.ylabel('# Steps')
plt.title('Figure 2(b): Decompression Latency vs. Density SuiteSparse')
plt.legend()
plt.grid(True)
plt.savefig('result.png')
plt.show()