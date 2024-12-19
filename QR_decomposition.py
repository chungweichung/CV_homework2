
def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for k in range(n):

        x = R[k:, k]
    
        norm_x = np.linalg.norm(x)

        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x + np.sign(x[0]) * norm_x * e1
        v = v / np.linalg.norm(v)

        H = np.eye(m)
        H[k:, k:] -= 2.0 * np.outer(v, v)


        R = H @ R
        Q = Q @ H

    return Q, R
if __name__ == "__main__":
  A=np.array([[10,10,1],[2,5,4],[10,8,9]])
  Q, R=householder_qr(A)

  print('Q:')
  print(Q)
  print('R:')
  print(R)

  print('Q mult Q transpot:')
  print(Q@Q.T)
  print('Q transpot mult Q:')
  print(Q.T@Q)

  print('inverse of A:')
  A_inverse=np.linalg.inv(A)
  print(A_inverse)
  Q_inverse,R_inverse=householder_qr(A_inverse)
  print('Q_inverse:')
  print(Q_inverse)
  print('R_inverse:')
  print(R_inverse)
  print('Q_inverse mult Q_inverse transpot:')
  print(Q_inverse@Q_inverse.T)
  print('R_inverse')
  print(np.linalg.inv(R_inverse))
  print('Q_inverse')
  print(np.linalg.inv(Q_inverse))

  print(np.linalg.inv(R)@np.linalg.inv(Q))
