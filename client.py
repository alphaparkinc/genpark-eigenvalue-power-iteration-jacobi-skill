"""Jacobi Eigenvalue & Eigenvector Algorithm.
100% Python Standard Library.
"""

import math

class JacobiEigen:
    """Jacobi rotation method for all eigenvalues and eigenvectors of real symmetric matrices."""

    @staticmethod
    def eigenvalues(A: list, max_iter: int = 100, eps: float = 1e-9) -> tuple:
        n = len(A)
        V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        D = [row[:] for row in A]

        for _ in range(max_iter):
            max_val = 0.0
            p, q = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(D[i][j]) > max_val:
                        max_val = abs(D[i][j])
                        p, q = i, j
            if max_val < eps:
                break
            theta = 0.5 * math.atan2(2 * D[p][q], D[p][p] - D[q][q])
            c = math.cos(theta)
            s = math.sin(theta)
            D_pp = c*c * D[p][p] + 2*s*c * D[p][q] + s*s * D[q][q]
            D_qq = s*s * D[p][p] - 2*s*c * D[p][q] + c*c * D[q][q]
            D[p][q] = D[q][p] = 0.0
            D[p][p] = D_pp
            D[q][q] = D_qq
            for k in range(n):
                if k != p and k != q:
                    d_pk = c * D[p][k] + s * D[q][k]
                    d_qk = -s * D[p][k] + c * D[q][k]
                    D[p][k] = D[k][p] = d_pk
                    D[q][k] = D[k][q] = d_qk
            for k in range(n):
                v_kp = c * V[k][p] + s * V[k][q]
                v_kq = -s * V[k][p] + c * V[k][q]
                V[k][p] = v_kp
                V[k][q] = v_kq

        eigenvalues = [D[i][i] for i in range(n)]
        return eigenvalues, V
