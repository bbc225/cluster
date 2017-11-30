###cluster.py
import scipy
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs  
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import spectral_clustering

# 层次聚类
def hierarchycluster(points,maxclust):
	# 1.生成距离矩阵，用欧式距离 
	#disMat = sch.distance.pdist(points,'euclidean') 
	# 2.层次聚类 
	#Z=sch.linkage(disMat,method='average') 
	#or
	model=AgglomerativeClustering(n_clusters=maxclust, affinity='euclidean', linkage='average')
	# 3.层次聚类结果树状图保存
	#P=sch.dendrogram(Z)
	#plt.savefig('plot_dendrogram.png')
	# 4.得到criterion为inconsistent的聚类结果 
	#criterion_1= sch.fcluster(Z, t=1, criterion='inconsistent') 
	# 5.得到criterion为maxclust的聚类结果 
	#y_pred= sch.fcluster(Z, t=maxclust, criterion='maxclust') 
	#or
	y_pred=model.fit(points).labels_
	plt.subplot(262)
	plt.scatter(points[:, 0], points[:, 1], c=y_pred)  
	plt.title("hierarchy")
# kmeans聚类
def kmeanscluster(points,maxclust,random_state):
	# 1.kmeans,第一维为聚类个数k.
	model=KMeans(n_clusters=maxclust, random_state=random_state)
	# 2.所有数据的label
	y_pred=model.fit(points).labels_ 
	# 3.添加子图
	plt.subplot(263)
	plt.scatter(points[:, 0], points[:, 1], c=y_pred)  
	plt.title("kmeans")
# GMM聚类
def GMMcluster(points,maxclust):
	# 1.gmm模型
	gmm=GaussianMixture(n_components=maxclust,max_iter=1000)
	gmm.fit(points)
	# 2.所有数据的label
	y_pred=gmm.predict(points)
	# 3.添加子图
	plt.subplot(264)
	plt.scatter(points[:, 0], points[:, 1], c=y_pred)  
	plt.title("GMM")
# SOM聚类
def SOMcluster(points,maxclust):
	import sompy.sompy 
	# 1.声明模型
	mapsize = [20,20]
	som = sompy.SOMFactory.build(points,mapsize, training='batch')  
	# mapsize,二维平面大小，如果是单一数字，表示节点数量
	# 2.训练
	som.train(n_job=1, verbose='info') 
	# 3.map聚类
	cl = som.cluster(n_clusters=maxclust)
	# 4.聚类标签
	y_map = som.find_bmu(points, njb=1)[0]
	y_pred=np.zeros_like(y_map)
	for i in range (y_pred.shape[0]):
		y_pred[i] = cl[int(y_map[i])]
	# 5.添加子图
	plt.subplot(265)
	plt.scatter(points[:, 0], points[:, 1], c=y_pred)  
	plt.title("som")
#AP聚类
def APcluster(points):
	# 1.AP模型声明 
	af = AffinityPropagation(preference=-50).fit(points)
	# 2.获取聚类结果
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_
	n_clusters_ = len(cluster_centers_indices)
	# 3.添加子图
	#from itertools import cycle
	#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	plt.subplot(266)
	plt.scatter(points[:, 0], points[:, 1], c=labels)  
	#for k, col in zip(range(n_clusters_), colors):
	#	class_members = labels == k
	#	cluster_center = points[cluster_centers_indices[k]]
		#plt.plot(points[class_members, 0], points[class_members, 1], col + '.')
		#plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
		#for x in points[class_members]:
		#	plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
	plt.title('AP')
def similarity_function(points):  
	from sklearn.metrics.pairwise import rbf_kernel  
	'''
	相似性函数，利用径向基核函数计算相似性矩阵，对角线元素置为０ 
	对角线元素为什么要置为０我也不清楚，但是论文里是这么说的 
	:param points: 
	'''
	res = rbf_kernel(points)  
	for i in range(len(res)):  
		res[i, i] = 0  
	return res  
#谱聚类
def spectralcluster(points,maxclust):
	from sklearn.feature_extraction import image
	# 1.Convert the image into a graph with the value of the gradient on the edges.
	W = similarity_function(points)
	# 2.谱聚类
	labels = spectral_clustering(W, n_clusters=maxclust)
	# 3.添加子图
	plt.subplot(267)
	plt.scatter(points[:, 0], points[:, 1], c=labels)
	plt.title('spectral')
#DBSCAN聚类
def dbscancluster(points):
	from sklearn.cluster import DBSCAN
	from sklearn.preprocessing import StandardScaler
	# 1.transform
	X = StandardScaler().fit_transform(points)
	# 2.Compute DBSCAN
	db = DBSCAN(eps=0.1, min_samples=10).fit(X)
	labels = db.labels_
	# 3.添加子图
	plt.subplot(268)
	plt.scatter(points[:, 0], points[:, 1], c=labels)
	plt.title('dbscan')
#FCM聚类
def fcmcluster(points,maxclust):
	import skfuzzy as fuzz
	# 1.模型声明
	cntr,u,u0,d,jm,p,fpc = fuzz.cmeans(points.T, maxclust, 2, error=0.005, maxiter=1000)
	# Plot assigned clusters, for each data point in training set
	# 2.计算聚类结果 
	cluster_membership = np.argmax(u, axis=0)
	# 3.添加子图
	plt.subplot(269)
	plt.scatter(points[:, 0], points[:, 1], c=cluster_membership)
	plt.title('fcm')
#birch聚类
def birchcluster(points,maxclust):
	from sklearn.cluster import Birch
	# 1.模型声明
	y_pred = Birch(n_clusters = maxclust,threshold = 0.1).fit_predict(points)
	# 2.添加子图
	plt.subplot(2,6,10)
	plt.scatter(points[:, 0], points[:, 1], c=y_pred)
	plt.title('birch')
#meanshift聚类
def meanshiftcluster(points):
	from sklearn.cluster import MeanShift, estimate_bandwidth
	# 1.The following bandwidth can be automatically detected using
	bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
	# 2.meanshift模型
	ms = MeanShift(bandwidth=bandwidth)
	ms.fit(points)
	labels = ms.labels_
	# 3.添加子图
	plt.subplot(2,6,11)
	plt.scatter(points[:, 0], points[:, 1], c=labels)
	plt.title('meanshift')
if __name__=="__main__":
	plt.figure(1)  
	n_samples = 1500  
	random_state = 170  
	# 1. 生成矩阵
	print "generate data\n"
	X, Y = make_blobs(n_samples=n_samples,  cluster_std=[1.0, 2.5, 0.5], random_state=random_state)  
	plt.subplot(261)
	plt.scatter(X[:, 0], X[:, 1], c=Y)  
	plt.title("origin")
	maxclust = 3 
	print "points count: ",X.shape
	# 2. 调用层次聚类
	print "hierarchy\n"
	hierarchycluster(X,maxclust)
	# 3. 调用kmeans
	print "kmeans\n"
	kmeanscluster(X,maxclust,random_state)
	# 4. 调用GMM
	print "GMM\n"
	GMMcluster(X,maxclust)
	# 5.调用som
	print "SOM\n"
	SOMcluster(X,maxclust)
	# 6.调用AP
	print "AP\n"
	APcluster(X)
	# 7.调用spectralcluster
	print "spectralcluster\n"
	spectralcluster(X,maxclust)
	# 8.调用dbscancluster
	print 'dbscancluster\n'
	dbscancluster(X)
	# 9.调用FCM
	print 'fcmcluster\n'
	fcmcluster(X,maxclust)
	# 10.调用birch
	print 'birchcluster\n'
	birchcluster(X,maxclust)
	# 11.调用meanshift
	print 'meanshiftcluster\n'
	meanshiftcluster(X)
	plt.show()