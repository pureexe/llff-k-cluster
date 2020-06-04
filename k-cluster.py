import os 
import numpy as np
from load_llff import load_llff_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial import Delaunay
import copy 
import json
from PIL import Image,ImageDraw
import math

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#LLFF_DIR = "C:/Datasets/nerf_llff_data/"
LLFF_DIR = "/data/orbiter/datasets/nerf_llff_data/"
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
    images_path = ['images/{}'.format(f) for f in sorted(os.listdir(image_dir))]
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

def create_Delaunay(images_train,reference):
    centers = np.array(list(map(lambda x: x['center'],images_train)))[:,:,0]
    pca = PCA(.95)
    pca.fit(centers)
    train_c = pca.transform(centers)
    #print(reference)
    ref = np.array([train_c[i] for i in reference])
    #ref = [train_c[i] for i in reference]
    indices = list(range(len(ref)))
    tri = Delaunay(ref, incremental=True)

    def orderConvexHull(points, ch):
        mp = {}
        for c in ch:
            if c[0] not in mp:
                mp[c[0]] = []
            if c[1] not in mp:
                mp[c[1]] = []

            mp[c[0]].append(c[1])
            mp[c[1]].append(c[0])
        #print()

        ls = [ch[0][0], ch[0][1]]
        while ls[0] != ls[-1]:
            two = mp[ls[-1]]
            if two[0] == ls[-2]:
                ls.append(two[1])
            else:
                ls.append(two[0])

        # Find top point, opengl convention. I.e., max y
        maxy = -1e10
        maxyi = -1
        for i, p in enumerate(ls):
            if points[p][1] > maxy:
                maxy = points[p][1]
                maxyi = i

        if points[ls[(maxyi+1) % len(ls)]][0] > points[ls[maxyi]][0]:
            return ls
        else:
            return ls[::-1]

    convexList = orderConvexHull(tri.points, tri.convex_hull)
    convexList = convexList[:-1]
    pp = tri.points
    #print(tri.points)
    plt.plot(train_c[:,0],train_c[:,1], 'o')
    plt.triplot(pp[:,0], pp[:,1], tri.simplices.copy())
    plt.plot(pp[:,0],pp[:,1], 'rx')
    plt.savefig('output/xx.png')
    newpoints = []
    def func(v1):
      #print("v1", v1)
      v1 = np.array([v1[0], v1[1], 0])
      v2 = np.array([0, 0, -1])
      v3 = np.cross(v1, v2)
      #print("v3", v3)
      v3 = v3 / np.linalg.norm(v3)
      return v3
      
    for i in range(len(convexList)):
      p0 = convexList[i]
      #if indices[p0] == 255: continue

      pa = convexList[(i+1) % len(convexList)]
      pb = convexList[(len(convexList)+i-1) % len(convexList)]
      va = func(tri.points[pa] - tri.points[p0])
      vb = -func(tri.points[pb] - tri.points[p0])
      vs = (va + vb) * 0.5;
      vs = vs / np.linalg.norm(vs)
      newpoints.append(tri.points[p0] + vs[:2] * 0.8)
      indices += [indices[p0]]

    print(indices)
    tri.add_points(newpoints)
    
    pp = tri.points
    plt.triplot(pp[:,0], pp[:,1], tri.simplices.copy())
    plt.plot(pp[:,0],pp[:,1], 'o')
    plt.savefig('output/xxx.png')

    size = 512
    # Create empty black canvas
    imBary = Image.new('RGB', (size, size))
    drawBary = ImageDraw.Draw(imBary)

    imInd = Image.new('RGB', (size, size))
    drawInd = ImageDraw.Draw(imInd)

    imIndVis = Image.new('RGB', (size, size))
    drawIndVis = ImageDraw.Draw(imIndVis)

    maxcoordinate = np.max(np.abs(tri.points))
    scaler = size * 0.5 * 0.98 / maxcoordinate
    shifter = size / 2

    vlist = []
    for i in range(size):
        for j in range(size):
            vx = i
            vy = j
            vx = (vx - shifter) / scaler
            vy = (shifter - vy) / scaler
            vlist.append((vx, vy))

    ind = tri.find_simplex(vlist)

    simscaler = int(math.floor(255 / (len(tri.simplices) - 1)))
    for i in range(size):
        for j in range(size):
            val = ind[i * size + j]
            if val > -1:
                #print(tri.simplices[val])
                ids = [indices[tri.simplices[val][x]] for x in range(3)]
                #print(ids)
                drawInd.point([(i, j)], fill=(ids[0], ids[1], ids[2]))
                drawIndVis.point([(i, j)],
                                fill=(ids[0] * simscaler,
                                    ids[1] * simscaler,
                                    ids[2] * simscaler))

                bary = tri.transform[val, :2, :2] * np.matrix(vlist[i * size + j] - tri.transform[val, 2, :]).transpose()
                drawBary.point([(i, j)], fill=(bary[0] * 255, bary[1] * 255, (1-bary[0]-bary[1]) * 255))

    imBary.save('output/bary.png')
    imInd.save('output/indices.png')
    imIndVis.save('output/indicesVis.png')

    exit()

def del_cam_planes(image_object):
    del image_object['camera']
    del image_object['planes']
    del image_object['r']
    del image_object['R']
    del image_object['t']
    del image_object['center']
    return image_object

def write_cluster_config(name, images_train, images_valid, reference, members):
    output = {}
    # we don't need to handle validation set as this
    """
    # validation set
    validation_set = []
    val_camera = copy.copy(images_valid[0]['camera'])
    validation_set.append({
        "camera": val_camera,
        "images": list(map(del_cam_planes, images_valid))
    })
    output['validation'] = validation_set
    """
    training_set = []
    for i, member in enumerate(members):    
        current_cluster = [img for j,img in enumerate(images_train) if member[j]]
        ref_view = images_train[reference[i]]
        ref_view = copy.deepcopy(ref_view)
        ref_view2 = copy.deepcopy(ref_view)
        current_cluster = copy.deepcopy(current_cluster)
        training_set.append({
            #"camera": ref_view['camera'], #No longer need (?)
            "images": list(map(del_cam_planes, current_cluster)),
            "planes": ref_view['planes'],
            'refimg': del_cam_planes(ref_view2)
        })
    output['training'] = training_set
    with open('output/{}.json'.format(name),'w') as f:
        output_json = json.dumps(output, cls=NumpyEncoder, sort_keys=True,indent=2)
        f.write(output_json)

def main():
    #for dataset in ['fern','flower','fortress','horns','leaves','orchids','room','trex']:
    for dataset in ['fern']:
        images_train, images_valid = get_images_data(dataset)
        member, reference = find_cluster(images_train)
        create_Delaunay(images_train,reference)
        centers = np.array(list(map(lambda x: x['center'],images_train)))[:,:,0]
        write_cluster_config(dataset, images_train, images_valid, reference, member)
    #visualization(centers, member, reference, DATASET)

if __name__ == "__main__":
    main()