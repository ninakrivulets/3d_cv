import numpy as np
import torch
import matplotlib.pylab as plt
from tqdm import tqdm
import open3d as o3d


def homogenize_points(points):
    last_coordinate = torch.ones_like(points[..., -1:])
    return torch.cat((points, last_coordinate), dim=-1)


projections = torch.load('/Users/ninakrivulets/Desktop/HSE/3 course/3d_cv/data.pth')
rgbs = torch.load('/Users/ninakrivulets/Desktop/HSE/3 course/3d_cv/rgbs.pth')

def get_fundamental_matrix(points_l, points_r):
    #raise NotImplementedError
    equations = torch.einsum('bi,bj->bij', points_r, points_l).view(-1, 9)
    u, s, v = torch.linalg.svd(equations)
    F = v[-1].view(3, 3)
    return F

def compute_residuals(F, points_l, points_r):
    #raise NotImplementedError
    return torch.einsum('bi,ij,bj->b', points_r, F, points_l)

def compute_errors(F, points_l, points_r):
    # optional
    raise NotImplementedError

# параметры объектива указывают в миллиметрах, а размер сенсора соответствет ширине кадра плёнки
f = 50
sensor_size = 36
# по ним строим матрицу внутренних параметров (intrinsic parameters)
K = torch.tensor([
    [2 * f / sensor_size, 0.0, 0.0],
    [0.0, 2 * f / sensor_size, 0.0],
    [0.0,                 0.0, 1.0]
])

def get_common_coords(proj_0, proj_1):
    ids0, coords0 = proj_0['ids'], proj_0['coordinates']
    ids1, coords1 = proj_1['ids'], proj_1['coordinates']

    common_ids = torch.tensor(list(set(ids0.tolist()) & set(ids1.tolist())))

    idx0 = torch.nonzero(torch.isin(ids0, common_ids)).squeeze(1)
    idx1 = torch.nonzero(torch.isin(ids1, common_ids)).squeeze(1)

    return coords0[idx0], coords1[idx1], common_ids

def get_common_projections(proj_a, proj_b):
    ids_a, coords_a = proj_a['ids'], proj_a['coordinates']
    ids_b, coords_b = proj_b['ids'], proj_b['coordinates']

    # Преобразуем в словари для быстрого доступа
    map_a = {idx.item(): i for i, idx in enumerate(ids_a)}
    map_b = {idx.item(): i for i, idx in enumerate(ids_b)}

    # Находим общие id
    common_ids = list(set(map_a.keys()) & set(map_b.keys()))

    indices_a = torch.tensor([map_a[idx] for idx in common_ids])
    indices_b = torch.tensor([map_b[idx] for idx in common_ids])

    common_coords_a = coords_a[indices_a]
    common_coords_b = coords_b[indices_b]

    return {
        'ids': torch.tensor(common_ids),
        'coordinates_a': common_coords_a,
        'coordinates_b': common_coords_b
    }


def get_essential_matrix(F):
    E = K.T @ F @ K
    return E

def get_rotation_translation(E, is_last=0):
    U, S, Vt = torch.linalg.svd(E, full_matrices=True)

    W = [[0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]]

    W = torch.tensor(W)

    W = W.float()
    S = S.float()
    t_x = U @ W @ torch.diag(S) @ U.T
    t = torch.tensor([[(t_x[2, 1] - t_x[1, 2]) / 2,
                     (t_x[0, 2] - t_x[2, 0]) / 2,
                     (t_x[1, 0] - t_x[0, 1]) / 2]]).T
    
    
    if is_last==0:
        R = U @ W @ Vt
    else:
        R = U @ W.T @ Vt

    return t, R
    
def get_proj_matrix(t_x, R):
    zero_col = torch.zeros(K.shape[0], 1)
    H = torch.cat([K, zero_col], dim=1)
    ones = torch.eye(4)
    H_l = H @ ones
    # делаю столбец [t, 1]
    #t_x = torch.tensor([0.4684, 1.6618, 2.3047])
    new_element = torch.tensor([1.0]).unsqueeze(1)
    #print('NEW_el shape', new_element.shape)
    t_x_1 = torch.cat((t_x, new_element), dim=0)
    zero_row = zero_col.T
    R_0 = torch.cat([R, zero_row], dim=0)
    print('SHAPES', R_0.shape, t_x_1.shape)
    full = torch.cat([R_0, t_x_1], dim=1)
    H_r = H @ full
    return H_l, H_r
    
def dehomogenize_points(points):
    return points[..., :-1] / (points[..., -1:] + 1e-8 * torch.sign(points[..., -1:]))

def linear_triangulation(Hl, Hr, xl, xr):
    # Строим систему уравнений AX = 0
    A = [
        xl[0] * Hl[2] - Hl[0],
        xl[1] * Hl[2] - Hl[1],
        xr[0] * Hr[2] - Hr[0],
        xr[1] * Hr[2] - Hr[1]
    ]
    A = torch.stack(A)

    # Решаем через SVD
    _, _, Vt = torch.linalg.svd(A)
    X = Vt[-1]  # Последняя строка V^T
    return X # Возвращаем (X, Y, Z)

def full_pipeline(proj_one, proj_two):
    coords0, coords1, common_01 = get_common_coords(proj_one, proj_two)
    
    colours = rgbs[common_01]  

    hom_coords0_1 = homogenize_points(coords1)
    hom_coords1_0 = homogenize_points(coords0)
    F = get_fundamental_matrix(hom_coords0_1, hom_coords1_0)
    E = get_essential_matrix(F)
    t_x, R = get_rotation_translation(E)
    H_l, H_r = get_proj_matrix(t_x, R)
    points = []
    for i in tqdm(range(len(hom_coords0_1))):
        points.append(linear_triangulation(H_l, H_r,hom_coords0_1[i], hom_coords1_0[i])) #hom_coords0_1[i], hom_coords1_0[i]))
    points = torch.stack(points)
    points = dehomogenize_points(points)
    return {'ids': torch.as_tensor(common_01), 'coordinates': torch.as_tensor(np.asarray(points))}
    #return points, common_01


def merge_point_clouds(cloud_a, cloud_b):
    
    # общие точки между двумя облаками
    overlap = get_common_projections(cloud_a, cloud_b)
    shared_ids = overlap['ids']
    points_a = homogenize_points(overlap['coordinates_a'])
    points_b = homogenize_points(overlap['coordinates_b'])

    # Находим преобразование из системы координат A в B
    transform = torch.linalg.pinv(points_a) @ points_b

    # Определяем маску уникальных точек из cloud_a, которых нет в cloud_b
    shared_id_set = set(shared_ids.tolist())
    unique_mask = torch.tensor([idx.item() not in shared_id_set for idx in cloud_a['ids']])

    # Преобразуем уникальные точки cloud_a в систему координат камеры B
    transformed_coords = dehomogenize_points(homogenize_points(cloud_a['coordinates']) @ transform)

    # Объединяем координаты и идентификаторы
    merged_coords = torch.cat([transformed_coords[unique_mask], cloud_b['coordinates']], dim=0)
    merged_ids = torch.cat([cloud_a['ids'][unique_mask], cloud_b['ids']], dim=0)

    return {'ids': merged_ids, 'coordinates': merged_coords}


points_01 = full_pipeline(projections[1], projections[0])
points_02 = full_pipeline(projections[0], projections[2])
points_12 = full_pipeline(projections[1], projections[2])

intersection1 = merge_point_clouds(points_01, points_02)
intersection2 = merge_point_clouds(intersection1, points_12)

#print(points_02)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(intersection2['coordinates'])
colours = rgbs[intersection2['ids']] 
pcd.colors = o3d.utility.Vector3dVector(colours)
o3d.visualization.draw_geometries([pcd])

#torch.save(intersection2, '/Users/ninakrivulets/Desktop/HSE/3 course/3d_cv/solution.pth')
#torch.save(intersection2['coordinates'], '/Users/ninakrivulets/Desktop/HSE/3 course/3d_cv/solution_coordinates.pth')