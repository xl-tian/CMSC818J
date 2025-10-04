class CSR:
    def __init__(self, matrix):
        self.matrix = matrix
        self.offsets = []
        self.col_indices = []
        self.values = []

        self.num_rows = len(matrix)
        self.num_cols = len(matrix[0])


        offset = 0
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if matrix[r][c] != 0:
                    self.col_indices.append(c)
                    self.values.append(matrix[r][c])
                    offset += 1
            self.offsets.append(offset)

        self.B0 = self.offsets
        self.B1 = self.col_indices
        self.B2 = self.values

    
    def get_storage_overhead(self):
        metadata_size = len(self.B0) + len(self.B1)
        data_size = len(self.B2)

        ratio = metadata_size /  data_size

        return ratio
    

    def decompress(self, verbose=False):
        data = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        num_steps = 0

        row = 0
        offset_index = 0
        read_index = 0
        start = 0

        while offset_index < len(self.B0):
            end = self.B0[offset_index]
            if verbose:
                print(f"Step {num_steps}: Reading end offset: {end} from B0")
            num_steps += 1
            
            length = end - start
            if verbose:
                print(f"Step {num_steps}: Calculating length: {length}")
            num_steps += 1

            for i in range(length):
                col = self.B1[read_index+i]
                val = self.B2[read_index+i]
                data[row][col] = val
                if verbose:
                    print(f"Step {num_steps}: Reading column index: {col} into B1 and value: {val} into B2")
                num_steps += 1

        
            start = end
            offset_index += 1
            read_index += length
            row += 1

        return data, num_steps
    

class BCSR:
    def __init__(self, matrix, block_size):
        self.matrix = matrix
        self.offsets = []
        self.col_indices = []
        self.values = []
        self.block_size = block_size

        self.num_rows = len(matrix)
        self.num_cols = len(matrix[0])

        offset = 0
        num_blocks_row = 0
        num_blocks_total = 0
        for r in range(0, self.num_rows, block_size):
            for c in range(0, self.num_cols, block_size):
                num_zeros = 0
                for r1 in range(block_size):
                    for c1 in range(block_size):
                        if matrix[r + r1][c + c1] != 0:
                            num_zeros += 1


                if num_zeros != 0:
                    self.values.append([])
                    self.col_indices.append(c)
                    for r1 in range(block_size):
                        for c1 in range(block_size):
                            self.values[num_blocks_total].append(matrix[r + r1][c + c1])
                    num_blocks_row += 1
                    num_blocks_total += 1
            offset += num_blocks_row
            num_blocks_row = 0
            self.offsets.append(offset)

        self.B0 = self.offsets
        self.B1 = self.col_indices
        self.B_List = self.values


    def get_storage_overhead(self):
        metadata_size = len(self.B0) + len(self.B1)
        data_size = sum([len(block) for block in self.B_List])

        ratio = metadata_size /  data_size

        return ratio
    
    

    def decompress(self, verbose=False):

        data = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        num_steps = 0
        row = 0
        offset_index = 0
        col_index = 0
        val_index = 0
        start = 0

        while offset_index < len(self.B0):
            end = self.B0[offset_index]
            
            num_steps += 1

            length = end - start
            num_steps += 1

            for i in range(length):
                col = self.B1[col_index+i]

                for j in range(self.block_size * self.block_size):
                    val = self.B_List[val_index+i][j]
   
                    r = j // self.block_size
                    c = j % self.block_size
                    data[row + r][col + c] = val
                num_steps += 1
            col_index += length
            val_index += length
            start = end
            offset_index += 1
            row += self.block_size

        return data, num_steps


class List_of_Lists:
    def __init__(self, matrix):
        self.matrix = matrix

        self.num_rows = len(matrix)
        self.num_cols = len(matrix[0])

        self.row_indices = [[] for _ in range(self.num_cols)]
        self.values = [[] for _ in range(self.num_cols)]

        local = 0
        max = 0
        for c in range(self.num_cols):
            for r in range(self.num_rows):
                if matrix[r][c] != 0:
                    self.row_indices[c].append(r)
                    self.values[c].append(matrix[r][c])
            local = len(self.row_indices[c])
            if local > max:
                max = local
        
        for c in range(self.num_cols):
            while len(self.row_indices[c]) < max:
                self.row_indices[c].append(-1)
                self.values[c].append(0)
        


        self.B_List1 = self.values
        self.B_List2 = self.row_indices

    def get_storage_overhead(self):
        num_invalid = 0
        for c in range(self.num_cols):
            for r in range(len(self.B_List2[c])):
                if self.B_List2[c][r] == -1:
                    num_invalid += 1

        ratio = num_invalid / (len(self.B_List2[0]) * self.num_cols)

        return ratio

    def decompress(self, verbose=False):
        data = [[0 for _ in range(self.num_rows)] for _ in range(self.num_cols)]
        num_steps = 0

        row_indices = [0 for _ in range(self.num_cols)]
        value_indices = [0 for _ in range(self.num_cols)]
        indices = [0 for _ in range(self.num_cols)]
        done = False

        while not done:
            row_indices = [self.row_indices[c][indices[c]] if indices[c] < len(self.row_indices[c]) else -1 for c in range(self.num_cols)]
            value_indices = [self.values[c][indices[c]] if indices[c] < len(self.values[c]) else 0 for c in range(self.num_cols)]
            num_steps += 1

            min_r = min([r for r in row_indices if r != -1], default=-1)
            if min_r == -1:
                done = True
            else:
                for c in range(self.num_cols):
                    if row_indices[c] == min_r:
                        data[min_r][c] = value_indices[c]
                        indices[c] += 1
        return data, num_steps-1
    

class COO:
    def __init__(self, matrix):
        self.matrix = matrix
        self.tuples = []
        self.num_rows = len(matrix)
        self.num_cols = len(matrix[0])

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if matrix[r][c] != 0:
                    self.tuples.append((r, c, matrix[r][c]))

    def get_storage_overhead(self):

        return (1/3)

    def decompress(self):
        data = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        num_steps = 0

        for r, c, val in self.tuples:
            data[r][c] = val
            num_steps += 1
        
        return data, num_steps