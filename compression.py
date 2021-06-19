import numpy as np
import matplotlib.pyplot as plt
from random import sample #Used for random initialization
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import imageio
from choose_k_random_centroids import choose_k_random_centroids
from k_means import k_means
from find_closest_centroids import find_closest_centroids

# load image
image_file = "img/"+input("Enter the name of the image: ")+".png"

X = imageio.imread(image_file)
X = X[:, :, :3]
X = X / 255.

# testing
# print(X)

img_size = X.shape
# print(img_size[1])
 
plt.imshow(X)
plt.show()

# K Means on image
X = X.reshape(-1, 3)

# no_of_clusters = 8
no_of_clusters = int(input("Enter the number of colors needed in the image: "))
no_of_iter = 10     #can be changed

print("Choosing", no_of_clusters, "random centroids...")
random_centroids = choose_k_random_centroids(X, no_of_clusters)

print("Running the K-means algorithm...")
cluster_id, centroid_history = k_means(X, random_centroids, K=no_of_clusters, max_iter=no_of_iter)

# Compressing image
print("Compressing the image...")
final_centroids = centroid_history[-1]

indices = find_closest_centroids(X, final_centroids)

# Now loop through the original image and form a new image
# that only has 16 colors in it
compressed_image = np.zeros((indices.shape[0], 3))

final_image = np.zeros((cluster_id.shape[0],3))
for i in range(compressed_image.shape[0]):
    final_image[i] = final_centroids[int(cluster_id[i])]

X_compressed = final_image.reshape(img_size)
X_compressed*=255
X_compressed = X_compressed.astype(np.uint8)
imageio.imwrite(image_file[:-4]+"_compressed_"+str(no_of_clusters)+".png", X_compressed)

# Reshape the original image and the new, final image and draw them
# To see what the "compressed" image looks like
plt.figure()
plt.subplot(121)
dummy1 = plt.imshow(X.reshape(img_size))
plt.subplot(122)
dummy2 = plt.imshow(X_compressed)
plt.show()

print("Image compressed and saved successfully! \n")