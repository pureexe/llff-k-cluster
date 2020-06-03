import os 
import numpy as np
from load_llff import load_llff_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import copy 
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

LLFF_DIR = "C:/Datasets/nerf_llff_data/"
DATASET = "fern"
K_CLUSTER = 4

def get_images_data(dataset, split_val = 8):
    dataset_dir = os.path.join(LLFF_DIR,dataset)
    image_dir = os.path.join(dataset_dir,'images')
    _, poses, bds, _, _ = load_llff_data(dataset_dir, factor=None)
    # split into train data and validation data
    validation_ids = np.arange(poses.shape[0])
    validation_ids[::split_val] = -1 #validation every [0,7, ... ]
    validation_ids = validation_ids < 0
    # pick only pose in train
    images_path = [os.path.join('images', f) for f in sorted(os.listdir(image_dir))]
    images_train = []
    images_valid = []
    for image_id in range(poses.shape[0]):
        R = poses[image_id][:3,:3]
        t = poses[image_id][:3,3].reshape([3,1])
        # LLFF need to inverse rotation and translation to match our format
        R[:3,0] *= -1
        R = np.transpose(R)
        t = np.matmul(R,t)
        H = poses[image_id,0,-1]
        W = poses[image_id,1,-1]
        focal = poses[image_id,2,-1]
        img_obj = {
            "path": images_path[image_id],
            "r": R,
            "t": t,
            "R": R.T,
            "center": -R @ t,
            "planes": bds[image_id],
            "camera": {
                "width": int(W),
                "height": int(H),
                "fx": float(focal),
                "fy": float(focal),
                "px": float(W / 2.0),
                "py": float(H / 2.0)
            },
        }
        if not validation_ids[image_id]:
            images_train.append(img_obj)
        else:
            images_valid.append(img_obj)
    return images_train, images_valid

def visualization(centers, cluster_member, cluster_reference, title = ""):
    # plot visual preview
    color = ['y','m','c','g','b']
    fig = plt.figure()
    axes_shape = int(np.ceil(np.sqrt(K_CLUSTER)))
    for i in range(K_CLUSTER):
        color_label = [color[i % len(color)] if cluster_member[i][j] else 'k' for j in range(centers.shape[0])]
        color_label[cluster_reference[i]] = 'r'
        ax = fig.add_subplot(axes_shape, axes_shape, i+1, projection='3d', elev=48, azim=90)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=color_label)
        ax.set_title("{} cluster {:d}".format(title,i+1))
    plt.show()

def find_cluster(images_train):
    centers = np.array(list(map(lambda x: x['center'],images_train)))[:,:,0]
    kmean = KMeans(n_clusters=K_CLUSTER)
    kmean.fit(centers)
    labels = kmean.labels_
    cluster_reference = []
    cluster_member = []

    # find reference view
    max_distance = []
    for i in range(K_CLUSTER):
        # find reference cam (closest camera to cluster center)
        current_cluster = centers[labels == i]
        current_center = kmean.cluster_centers_[i],
        dists = np.sum(np.square(current_cluster - current_center), -1)
        ref_cam = np.argmin(dists)
        refcam_id = np.where(np.all(centers == current_cluster[ref_cam],axis=1))
        cluster_reference.append(refcam_id[0][0])
        # find max radius
        dists = np.sum(np.square(current_cluster - current_cluster[ref_cam]), -1)
        max_distance.append(np.max(dists))
    max_distance = max(max_distance)
    for i in range(K_CLUSTER):
        ref_cam = centers[cluster_reference[i]]
        all_dists = np.sum(np.square(centers - ref_cam), -1)
        cluster_member.append(all_dists <= max_distance)

    return cluster_member, cluster_reference

def del_cam_planes(image_object):
    del image_object['camera']
    del image_object['planes']
    return image_object

def write_cluster_config(name, images_train, images_valid, reference, members):
    output = {}
    # validation set
    validation_set = []
    val_camera = copy.copy(images_valid[0]['camera'])
    validation_set.append({
        "camera": val_camera,
        "images": list(map(del_cam_planes, images_valid))
    })
    training_set = []
    for i, member in enumerate(members):    
        current_cluster = [img for j,img in enumerate(images_train) if member[j]]
        ref_view = images_train[reference[i]]
        ref_view = copy.deepcopy(ref_view)
        ref_view2 = copy.deepcopy(ref_view)
        current_cluster = copy.deepcopy(current_cluster)
        training_set.append({
            "camera": ref_view['camera'],
            "images": list(map(del_cam_planes, current_cluster)),
            "planes": ref_view['planes'],
            'refimg': del_cam_planes(ref_view2)
        })
    output['validation'] = validation_set
    output['training'] = training_set
    with open('output/{}.json'.format(name),'w') as f:
        output_json = json.dumps(output, cls=NumpyEncoder)
        f.write(output_json)

def main():
    for dataset in ['fern','flower','fortress','horns','leaves','orchids','room','trex']:
        images_train, images_valid = get_images_data(dataset)
        member, reference = find_cluster(images_train)
        centers = np.array(list(map(lambda x: x['center'],images_train)))[:,:,0]
        write_cluster_config(dataset, images_train, images_valid, reference, member)
    #visualization(centers, member, reference, DATASET)

if __name__ == "__main__":
    main()