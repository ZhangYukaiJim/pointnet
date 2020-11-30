import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def cropout_point_cloud(batch_data, max_trans_dist, random_trans_dist=True, close=True):
    batch_size = batch_data.shape[0]
    num_points = batch_data.shape[1]
    if random_trans_dist:
        trans_dist = np.random.rand(batch_size)*max_trans_dist
    else:
        trans_dist = np.ones(batch_size)*max_trans_dist
    translation = np.zeros((batch_size, 3))
    translation[:, 2] = trans_dist          #translation distance initized onto z

    #rotate the translation vectors to random direction
    for k in range(batch_size):
        angle_x = np.random.uniform()*2*np.pi
        angle_y = np.random.uniform()*2*np.pi
        cos_x = np.cos(angle_x)
        sin_x = np.sin(angle_x)
        cos_y = np.cos(angle_y)
        sin_y = np.sin(angle_y)
        rotation_x = np.array([[1, 0, 0],
                            [0, cos_x, -sin_x],
                            [0, sin_x, cos_x]])
        rotation_y = np.array([[cos_y, 0, sin_y],
                            [0, 1, 0],
                            [-sin_y, 0, cos_y]])
        translation[k, :] =  np.dot(np.dot(translation[k,:],rotation_x),rotation_y)

    #apply translation
    batch_data_t = batch_data + np.expand_dims(translation,1)
    batch_dist = np.sqrt(np.sum(np.square(batch_data_t), 2))

    if(close): 
        out_idx = np.where(batch_dist>1)
        batch_data_t[out_idx[0],out_idx[1],:] = \
            batch_data_t[out_idx[0],out_idx[1],:]/np.expand_dims(batch_dist[out_idx],1)
    else:
        for k in range(batch_size):
            #mask out points outside the boundary
            out_idx = np.where(batch_dist[k,:]>1)
            out_num = len(out_idx[0])
            mask = np.ones(num_points, dtype=bool)
            mask[out_idx] = False
            pcd_data = np.delete(batch_data_t[k,:,:], out_idx, axis=0)
            #replace the deleted points with existing points
            replace_idx = np.random.choice(np.arange(num_points-out_num), out_num)
            replace_points = pcd_data[replace_idx,:]
            pcd_data = np.concatenate((pcd_data, replace_points), axis=0)
            batch_data_t[k,:,:] = pcd_data
    
    return batch_data_t


def bubble_cropout(batch_data, max_bubble_radius, random_bubble_radius=True, close=True):
    batch_size = batch_data.shape[0]
    num_points = batch_data.shape[1]

    #pick one point from each point cloud as bubble center
    bubble_centers_idx = np.random.choice(np.arange(num_points), batch_size)  #[32*1]
    bubble_centers = batch_data[np.arange(batch_size), bubble_centers_idx, :] + 0.001 #[32*3]
    bubble_centers = np.expand_dims(bubble_centers, 1) #[32,1,3]

    #apply translation
    batch_data_t = batch_data - bubble_centers
    batch_dist = np.sqrt(np.sum(np.square(batch_data_t), 2))

    if(close): 
        out_idx = np.where(batch_dist<max_bubble_radius)
        batch_data_t[out_idx[0],out_idx[1],:] = \
            batch_data_t[out_idx[0],out_idx[1],:]/np.expand_dims(batch_dist[out_idx],1) * max_bubble_radius
    else:
        # for k in range(batch_size):
        #     #mask out points outside the boundary
        #     out_idx = np.where(batch_dist[k,:]<max_bubble_radius)
        #     out_num = len(out_idx[0])
        #     mask = np.ones(num_points, dtype=bool)
        #     mask[out_idx] = False
        #     pcd_data = np.delete(batch_data_t[k,:,:], out_idx, axis=0)
        #     #replace the deleted points with existing points
        #     replace_idx = np.random.choice(np.arange(num_points-out_num), out_num)
        #     replace_points = pcd_data[replace_idx,:]
        #     pcd_data = np.concatenate((pcd_data, replace_points), axis=0)
        #     batch_data_t[k,:,:] = pcd_data
        out_idx = np.where(batch_dist<max_bubble_radius)
        batch_data_t[out_idx[0],out_idx[1],:] = 0
    
    batch_data = batch_data_t + bubble_centers
    return batch_data