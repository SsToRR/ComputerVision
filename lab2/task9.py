# Input patch
patch = [
    [1, 2, 0],
    [3, 1, 1],
    [2, 0, 4]
]

# Sobel X kernel
kernel = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

result = (
    patch[0][0] * kernel[0][0] +
    patch[0][1] * kernel[0][1] +
    patch[0][2] * kernel[0][2] +
    patch[1][0] * kernel[1][0] +
    patch[1][1] * kernel[1][1] +
    patch[1][2] * kernel[1][2] +
    patch[2][0] * kernel[2][0] +
    patch[2][1] * kernel[2][1] +
    patch[2][2] * kernel[2][2]
)

print("Convolution result:", result)
