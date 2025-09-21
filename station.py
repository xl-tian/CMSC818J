# stationary systolic array simulation

import numpy as np

# matrix_a = np.array([[1, 2], [3, 4]])
# matrix_b = np.array([[5, 6, 2], [7, 8, 9]])

# matrix_b = np.array([[5, 6], [7, 8]])

rng = np.random.default_rng()

n, m, p = 9, 7, 2

matrix_a = rng.integers(low=1, high=10, size=(n, m))
matrix_b = rng.integers(low=1, high=10, size=(m, p))

verbose = False

# Perform matrix multiplication
result_matrix = matrix_a @ matrix_b

if verbose:
    print("Matrix A:")
    print(matrix_a)
    print("Matrix B:")
    print(matrix_b)
    print("Result Matrix (A @ B):") 
    print(result_matrix)

class MAC:
    def __init__(self):
        self.b = 0
        self.a = 0
        self.save_a = 0
        self.weight = 0
        self.row = 0
        self.col = 0
        self.partial_result = 0
        self.save_partial_result = 0
        self.right = None
        self.down = None

    def set_position(self, row, col):
        self.row = row
        self.col = col

    def set_neighbors(self, right, down):
        self.right = right
        self.down = down

    def set_input(self, a):
        self.save_a = self.a
        self.a = a

    def set_weight(self, weight):
        self.save_weight = self.weight
        self.weight = weight

    def set_partial_result(self, partial_result):
        self.save_partial_result = self.partial_result
        self.partial_result = partial_result

    def load_data(self):
        self.down.set_weight(self.save_weight)

    def compute(self):
        if self.right is not None:
            self.right.set_input(self.save_a)

        if verbose:
            print(f"MAC({self.row},{self.col}) - compute: {self.save_partial_result} + {self.a} * {self.weight} = {self.save_partial_result + self.a * self.weight}")
        self.down.set_partial_result(self.save_partial_result + self.a * self.weight)


class SystolicArray:
    def __init__(self, n, m, p):
        self.n = n
        self.m = m
        self.p = p
        self.result = [[] for _ in range(self.p)]
        self.array = [[MAC() for _ in range(p)] for _ in range(m+1)]

        for i in range(m):
            for j in range(p):
                self.array[i][j].set_position(i, j)
                right = self.array[i][j + 1] if j + 1 < p else None
                down = self.array[i + 1][j]
                self.array[i][j].set_neighbors(right, down)

    def simulate(self, matrix_a, matrix_b):
        # load weights
        for timestep in range(self.m):
            for j in range(self.p):
                self.array[0][j].set_weight(matrix_b[self.m-timestep-1][j])
            for i in range(0, self.m-1):
                for j in range(self.p):
                    self.array[i][j].load_data()
        
        # compute
        for timestep in range(self.n + self.p + self.m - 2):
            for i in range(self.m):
                if timestep - i >= 0 and timestep - i < self.n:
                    self.array[i][0].set_input(matrix_a[timestep - i][i])
                else:
                    self.array[i][0].set_input(0)
            for i in range(self.m):
                for j in range(self.p):
                    self.array[i][j].compute()

            for j in range(self.p):
                if timestep >= self.m + j -1 and timestep < self.m + j -1 + self.n:
                    self.result[j].append(self.array[self.m][j].partial_result)

            if verbose:
                print(f"After timestep {timestep+1}:")
                for row in self.array:
                    print([mac.partial_result for mac in row])

        return np.array(self.result).T




arr = SystolicArray(n, m, p)
result = arr.simulate(matrix_a, matrix_b)

if verbose:
    print("Resultant Matrix:")
    print(result)

print(np.array_equal(result, result_matrix))