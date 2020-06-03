import os 
import numpy as np
from load_llff import load_llff_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LLFF_DIR = "C:/Datasets/nerf_llff_data/"
DATASET = "trex"
K_CLUSTER = 4

def get_images_train():
    dataset_dir = os.path.join(LLFF_DIR, DATASET)
    image_dir = os.path.join(dataset_dir,'images')
    _, poses, bds, _, _ = load_llff_data(dataset_dir, factor=None)
    # split into train data and validation data
    validation_ids = np.arange(poses.shape[0])
    validation_ids[::8] = -1 #validation every [0,7, ... ]
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
        img_obj = {
            "path": images_path[image_id],
            "center": -R @ t,
            "depth": bds[image_id]
        }
        if not validation_ids[image_id]:
            images_train.append(img_obj)
        else:
            images_valid.append(img_obj)
    return images_train

def main():
    images_train = get_images_train()
    centers = np.array(list(map(lambda x: x['center'],images_train)))[:,:,0]
    kmean = KMeans(n_clusters=K_CLUSTER)
    kmean.fit(centers)
    labels = kmean.labels_
    all_ref = np.zeros((4,3))
    # find reference view
    for i in range(K_CLUSTER):
        current_cluster = centers[labels == i]
        current_center = kmean.cluster_centers_[i],
        dists = np.sum(np.square(current_cluster - current_center), -1)
        ref_cam = np.argmin(dists)
        all_ref[i,:] = current_cluster[ref_cam]
        refcam_id = np.where(np.all(centers == current_cluster[ref_cam],axis=1))
        labels[refcam_id[0]] += 4
    # plot visual preview
    fig = plt.figure(1)
    ax = Axes3D(fig, elev=48, azim=90)
    #ax.scatter(all_ref[:, 0], all_ref[:, 1], all_ref[:, 2], c='r', edgecolor='r')
    # create label
    color = ['y','m','c','g','r','r','r','r']
    color_label = [ color[i] for i in labels ]
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=color_label)
    plt.show()

if __name__ == "__main__":
    main()


    """
    # find reference view
    for i in range(K_CLUSTER):
        # find reference cam (closest camera to cluster center)
        current_cluster = centers[labels == i]
        current_center = kmean.cluster_centers_[i],
        dists = np.sum(np.square(current_cluster - current_center), -1)
        ref_cam = np.argmin(dists)
        refcam_id = np.where(np.all(centers == current_cluster[ref_cam],axis=1))
        cluster_reference.append(refcam_id[0][0])
        # find every camera in max radius
        dists = np.sum(np.square(current_cluster - current_cluster[ref_cam]), -1)
        max_distance = np.max(dists)
        all_dists = np.sum(np.square(centers - current_cluster[ref_cam]), -1)
        cluster_member.append(all_dists <= max_distance)
    """