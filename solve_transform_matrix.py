import numpy as np

def compute_projection_matrix_dlt(orig_points, dst_points):
    def normalize_points(points):
        points = np.array(points)
        mean = np.mean(points, axis=0)
        std = np.std(points)
        transform = np.array([
            [1 / std, 0, -mean[0] / std],
            [0, 1 / std, -mean[1] / std],
            [0, 0, 1]
        ])
        normalized_points = np.dot(transform, np.vstack((points.T, np.ones(points.shape[0]))))
        return normalized_points[:2].T, transform

    orig_points_norm, T_orig = normalize_points(orig_points)
    dst_points_norm, T_dst = normalize_points(dst_points)

    A = []
    for (x, y), (x_prime, y_prime) in zip(orig_points_norm, dst_points_norm):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape((3, 3))  
    H = np.dot(np.linalg.inv(T_dst), np.dot(H_norm, T_orig))

    H = H / H[-1, -1]

    return H

def project_point(H, point):

    x, y = point
    vec = np.array([x, y, 1])
    projected = H @ vec
    projected /= projected[-1]
    return projected[:2]

def inverse_project_point(H, point):
    H_inv = np.linalg.inv(H)

    x_prime, y_prime = point
    vec = np.array([x_prime, y_prime, 1])
    

    original = H_inv @ vec
    
    original /= original[-1]
    
    return original[0], original[1]


if __name__ == "__main__":

    src_points = [(45, 104), (169, 579), (297, 124), (433, 463)]  # 原始點
    dst_points = [(0, 0), (0, 489), (287, 0), (287, 489)]          # 目標點

    H = compute_projection_matrix_dlt(src_points, dst_points)
    print("投影矩陣 (H):")
    print(H)


    original_point = (140, 300) 
    destination_point = project_point(H, original_point)
    print(f"(b) 原始點 {original_point} 映射到目標平面: {destination_point}")

    destination_point = (0, 400)
    original_point = inverse_project_point(H, destination_point)
    print(f"(c) 目標點 {destination_point} 映射回原始平面: {original_point}")

    print(np.linalg.inv(H))