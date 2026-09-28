from client import JacobiEigen

def main():
    A = [[4.0, 2.0], [2.0, 1.0]]
    eigs, V = JacobiEigen.eigenvalues(A)
    print("Eigenvalues:", [round(e, 4) for e in eigs])
    print("Eigenvector matrix V:", V)

if __name__ == "__main__":
    main()
