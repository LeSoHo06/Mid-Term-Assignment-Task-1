import random
import time
import csv
import numpy as np



# Function to generate a random matrix of a given size
def generate_random_matrix(size):
    matrix = []
    for n in range(size):
        row = [random.randint(1, 10) for m in range(size)] # Generate a row with random integers between 1 and 10
        matrix.append(row) # Append the row to the matrix
    
    return matrix # Return the generated matrix





# Perform matrix multiplication using nested loops (naive approach)
def matrix_multiplication(matrix_a, matrix_b):
    rows_a, columns_a = len(matrix_a), len(matrix_a[0])
    rows_b, columns_b = len(matrix_b), len(matrix_b[0])

    # Initialize result matrix with zeros
    result = [[ 0 for n in range(columns_b)] for n in range(rows_a)]

    # Perform multiplication by iterating through rows and columns
    for i in range(rows_a):
        for j in range(columns_b):
            for k in range(columns_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j] # Accumulate the product of corresponding elements

    return result # Return the resulting matrix
    




# Perform matrix multiplication using NumPy's dot function
def matrix_multiplication_numpy(matrix_a, matrix_b):
    np_matrix_a = np.array(matrix_a) # Convert list to NumPy array
    np_matrix_b = np.array(matrix_b) # Convert list to NumPy array
    result = np.dot(np_matrix_a, np_matrix_b) # Use NumPy's dot function for matrix multiplication
    return result.tolist() # Convert result back to list






# Function to measure and print the runtime of different matrix multiplication methods
def runtime_matrix_multiplication():
    sizes = range(2, 1001, 50) # Iterate over matrix sizes from 2 to 1000 with a step of 50
    for size in sizes:
        matrix_a = generate_random_matrix(size) # Generate random matrix A
        matrix_b = generate_random_matrix(size) # Generate random matrix B

        # Measure runtime of naive matrix multiplication
        start = time.time()
        result_manual = matrix_multiplication(matrix_a, matrix_b)
        end = time.time()

        # Measure runtime of NumPy matrix multiplication
        start_numpy = time.time()
        result_numpy = matrix_multiplication_numpy(matrix_a, matrix_b)
        end_numpy = time.time()
        

        # Measure runtime of Strassen's non-recursive multiplication (only for 2x2 matrices)
        if size == 2:
            start_strassen_non_recursive = time.time()
            result_strassen_non_recursive = strassen_without_recursion(matrix_a, matrix_b)
            end_strassen_non_recursive = time.time()
        else:
            result_strassen_non_recursive = None
            end_strassen_non_recursive = start_strassen_non_recursive = 0


        # Measure runtime of Strassen's recursive multiplication (only for 2x2 matrices)
        if size == 2:
            matrix_a_np = np.array(matrix_a) # Convert matrix A to NumPy array
            matrix_b_np = np.array(matrix_b) # Convert matrix B to NumPy array
            start_strassen_recursive = time.time()
            result_strassen_recursive = strassen_with_recursion(matrix_a_np, matrix_b_np)
            end_strassen_recursive = time.time()
        else:
            result_strassen_recursive = None
            end_strassen_recursive = start_strassen_recursive = 0


        # Save results to CSV if matrix size is greater than 52
        if size > 52:
            save_matrix_to_csv(matrix_a, f"matrix_a_{size}.csv")
            save_matrix_to_csv(matrix_b, f"matrix_b_{size}.csv")
            save_matrix_to_csv(result_manual, f"result_martrix_{size}.csv")
            
            save_matrix_to_csv(result_numpy, f"result_numpy_matrix_{size}.csv")
            
            print(f"Matrix size: {size}x{size}, Manual Time taken: {end - start} seconds, NumPy Time taken: {end_numpy - start_numpy: .6f} seconds")
            print(f"Full matrices saved to CSV files: matrix_a_{size}.csv, matrix_b_{size}.csv, result_manual_matrix_{size}.csv, result_numpy_matrix_{size}.csv\n")

        # Print matrices and runtime for small sizes (<= 52)
        else:
            print_matrix(matrix_a, f"Matrix A ({size}x{size})")
            print_matrix(matrix_b, f"Matrix B ({size}x{size})")
            print_matrix(result_manual, f"Result Matrix (Manual) ({size}x{size})")
            print_matrix(result_numpy, f"Result Matrix (Numpy) ({size}x{size})")

            if result_strassen_non_recursive is not None:
                print_matrix(result_strassen_non_recursive, f"Result Matrix (Strassen Non-Recursive) ({size}x{size})")
                print(f"Strassen Non-Recursive Time taken: {end_strassen_non_recursive - start_strassen_non_recursive: .6f} seconds")
            
            if result_strassen_recursive is not None:
                print_matrix(result_strassen_recursive, f"Result Matrix (Strassen Recursive) ({size}x{size})")
                print(f"Strassen Recursive Time taken: {end_strassen_recursive - start_strassen_recursive: .6f} seconds")

            print(f"Matrix size: {size}x{size}, Manual Time taken: {end - start: .6f} seconds, NumPy Time taken: {end_numpy - start_numpy: .6f} seconds\n")





# Function to save a matrix to a CSV file
def save_matrix_to_csv(matrix, filename):
    file = open(filename, 'w', newline = '') # Open file for writing
    writer = csv.writer(file) # Create a CSV writer object
    writer.writerows(matrix) # Write the matrix to the CSV file
    file.close() # Close the file





# Function to print a matrix
def print_matrix(matrix, name):
    print(f"{name}: ")
    for row in matrix:
        print(row) # Print each row of the matrix
    print()






# Strassen's algorithm without recursion for 2x2 matrices
def strassen_without_recursion(matrix_a, matrix_b):
    # Extract elements from the input matrices
    a = matrix_a[0][0]
    b = matrix_a[0][1]
    c = matrix_a[1][0]
    d = matrix_a[1][1]

    e = matrix_b[0][0]
    f = matrix_b[0][1]
    g = matrix_b[1][0]
    h = matrix_b[1][1]

    # Calculate M1 to M7 using Strassen's formula
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + d) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)

    # Calculate the resulting elements of the product matrix
    p11 = m1 + m4 - m5 + m7
    p12 = m3 + m5
    p21 = m2 + m4
    p22 = m1 - m2 + m3 + m6

    # Return the resulting 2x2 matrix
    return [[p11, p12], [p21, p22]]






# Strassen's algorithm with recursion for 2D matrices
def strassen_with_recursion(matrix_a, matrix_b):
    n = len(matrix_a)
    if n <= 2:
        return np.dot(matrix_a, matrix_b) # Base case: use NumPy's dot for small matrices

    mid = n // 2 # Find the midpoint to split the matrices

    # Partition matrix A into four submatrices
    a11 = matrix_a[:mid, :mid]
    a12 = matrix_a[:mid, mid:]
    a21 = matrix_a[mid:, :mid]
    a22 = matrix_a[mid:, mid:]

    # Partition matrix B into four submatrices
    b11 = matrix_b[:mid, :mid]
    b12 = matrix_b[:mid, mid:]
    b21 = matrix_b[mid:, :mid]
    b22 = matrix_b[mid:, mid:]

    # Calculate M1 to M7 using recursive calls
    m1 = strassen_with_recursion(a11, b12 - b22)
    m2 = strassen_with_recursion(a11 + a12, b22)
    m3 = strassen_with_recursion(a21 + a22, b11)
    m4 = strassen_with_recursion(a22, b21 - b11)
    m5 = strassen_with_recursion(a11 + a22, b11 + b22)
    m6 = strassen_with_recursion(a12 - a22, b21 +b22)
    m7 = strassen_with_recursion(a11 - a21, b11 + b12)

    # Calculate the resulting submatrices of the product matrix
    p11 = m5 + m4 - m2 + m6
    p12 = m1 + m2
    p21 = m3 + m4
    p22 = m5 + m1 - m3 - m7

    # Combine the submatrices to form the final product matrix
    P = np.vstack((np.hstack((p11, p12))), np.hstack((p21, p22)))
    return P 




# Execute the runtime measurement for matrix multiplication
runtime_matrix_multiplication()






# Sources:
# - ChatGPT 4.0: Helped me with finding some general problems with the code and with implementing its quality.
# - Youtube: Helped me explain algorithms like the NumPy or Strassens.
# - Task 1.c: https://www.digitalocean.com/community/tutorials/numpy-matrix-multiplication
# - Task 1.e (Strassen with recursion): https://www.geeksforgeeks.org/strassen-algorithm-in-python/
# - Task 1.e (Strassen without recursion): https://www.tutorialspoint.com/data_structures_algorithms/strassens_matrix_multiplication_algorithm.htm






