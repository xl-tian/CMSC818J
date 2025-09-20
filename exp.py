

import numpy as np

# Define two matrices as NumPy arrays
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6, 2], [7, 8, 9]])

rng = np.random.default_rng()

n, m, p = 4, 7, 6

matrix_a = rng.integers(low=0, high=10, size=(n, m))
matrix_b = rng.integers(low=0, high=10, size=(m, p))

# Perform matrix multiplication
result_matrix = matrix_a @ matrix_b

verbose = False

if verbose:
    print("Matrix A:")
    print(matrix_a)
    print("Matrix B:")
    print(matrix_b)
    print("Result Matrix (A @ B):")
    print(result_matrix)

class MAC:
    def __init__(self, row, col, m):
        self.accumulator = 0
        # first MAC is 0, 0
        self.row = row
        self.col = col
        self.time = 0
        self.m = m
        self.start_compute = False
        self.off_load = False
        self.right = None
        self.down = None

    def set_neighbors(self, right, down):
        self.right = right
        self.down = down

    def set_input(self, a):
        self.a = a

    def set_weight(self, b):
        self.b = b

    def step(self):
        if (self.row + self.col) == self.time:
            self.start_compute = True

        if (self.m == 0):
            self.off_load = True

        if self.start_compute and self.m > 0:
            if verbose:
                print(f"MAC({self.row},{self.col}) at timestep {self.time} computing: {self.a} * {self.b}")
            self.accumulator += self.a * self.b
            self.m -= 1
            self.save_a = self.a
            self.save_b = self.b
            # self.right.set_input(self.a)
            # self.down.set_weight(self.b)
        
        if self.off_load:
            self.save = self.accumulator
            self.accumulator = self.b

        self.time += 1

        return self.accumulator
    
    def flow(self):
        if self.start_compute and not self.off_load:
            self.right.set_input(self.save_a)
            self.down.set_weight(self.save_b)
        if self.off_load:
            if verbose:
                print(f"MAC({self.row},{self.col}) at timestep {self.time-1} offloading: {self.save}")
            self.down.set_weight(self.save)

# class Send_Buff:
#     def __init__(self, data, mac, count_down):
#         self.data = data
#         self.mac = mac
#         self.count_down = count_down

#     def step(self):
#         if self.count_down == 0:
#             self.mac.set_input(self.data)
#         self.count_down -= 1

class Systolic_Arr:
    def __init__(self, n, m, p):
        self.n = n
        self.m = m
        self.p = p
        self.array = [[MAC(row, col, self.m) for col in range(self.p+1)] for row in range(self.n+1)]
        self.result = [[] for _ in range(self.p)]

        for i in range(self.n):
            for j in range(self.p):
                self.array[i][j].set_neighbors(self.array[i][j+1], self.array[i+1][j])


    def simulate(self, A, B):
        for timestep in range(self.n + self.p + self.m - 2 + self.n):
            for i in range(self.n):
                if timestep - i >= 0 and timestep - i < self.m:
                    self.array[i][0].set_input(A[i][timestep - i])
            for j in range(self.p):
                if timestep - j >= 0 and timestep - j < self.m:
                    self.array[0][j].set_weight(B[timestep - j][j])
            for i in range(self.n):
                for j in range(self.p):
                    self.array[i][j].step()
                    if verbose:
                        print(f"MAC({i},{j}) at time {timestep}: {self.array[i][j].accumulator}")

            for i in range(self.n):
                for j in range(self.p):
                    self.array[i][j].flow()

            for i in range(self.p):
                if self.array[self.n-1][i].off_load and len(self.result[i]) < self.n:
                    self.result[i].append(self.array[self.n-1][i].save)

        if verbose:
            print("Final result:")
            print(np.flip(np.array(self.result), axis=1).T)

        return np.flip(np.array(self.result), axis=1).T

            
arr = Systolic_Arr(n, m, p)
result = arr.simulate(matrix_a, matrix_b)

print(np.array_equal(result, result_matrix))