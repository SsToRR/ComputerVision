# Input matrix
matrix = [
    [3, 0, 5, 2],
    [1, 4, 2, 6],
    [7, 1, 3, 0],
    [2, 8, 1, 5]
]

# Maxpooling parameters
pool_size = 2
stride = 2

output = [
    [
        max(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]),
        max(matrix[0][2], matrix[0][3], matrix[1][2], matrix[1][3])
    ],
    [
        max(matrix[2][0], matrix[2][1], matrix[3][0], matrix[3][1]),
        max(matrix[2][2], matrix[2][3], matrix[3][2], matrix[3][3])
    ]
]

print("Maxpooling output:")
for row in output:
    print(row)
