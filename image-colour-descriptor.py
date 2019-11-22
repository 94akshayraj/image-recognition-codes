from sklearn.cluster import KMeans
import cv2
def process_image_clusters(input_image,num_clusters):
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters = num_clusters)
    clt.fit(image)
    return clt.cluster_centers_

#Creditted to Adreian@pyimagesearch
