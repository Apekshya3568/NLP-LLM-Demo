def min_edit_distance(source, target):
    n = len(source)
    m = len(target)

    # Step 1: Create distance matrix (n+1 x m+1)
    D = [[0 for _ in range(m+1)] for _ in range(n+1)]

    # Step 2: Initialize first row and first column
    for i in range(1, n+1):
        D[i][0] = i  # cost of deleting i characters
    for j in range(1, m+1):
        D[0][j] = j  # cost of inserting j characters

    # Step 3: Fill the matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            if source[i-1] == target[j-1]:
                cost = 0  # no substitution needed
            else:
                cost = 1  # substitution cost

            D[i][j] = min(
                D[i-1][j] + 1,     # deletion
                D[i][j-1] + 1,     # insertion
                D[i-1][j-1] + cost # substitution
            )

    # Optional: print the DP matrix
    print("DP Matrix:")
    for row in D:
        print(row)

    # Step 4: Return minimum edit distance
    return D[n][m]

# Example usage
source = "intention"
target = "execution"
distance = min_edit_distance(source, target)
print(f"Minimum edit distance between '{source}' and '{target}' is {distance}")
