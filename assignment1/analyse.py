import math
import sys
import random
#single dimensional data

def Mean(data_points):
    assert len(data_points) != 0 , "No Data Points"
    return sum(data_points) / len(data_points)
    
def Variance(data_points):
    mean = Mean(data_points)
    return sum([(x-mean)**2 for x in data_points]) / len(data_points)

def StandardDeviation(data_points):
    return math.sqrt(Variance(data_points))

def GetData(file_path):
    try: file_data = open(file_path,"r").read()
    except IOError: print "File " + file_path + " not found"
    return [float(i) for i in file_data.split("\n") if len(i) > 0]

def distance(x,m,d_type,meta_info):
    if(d_type == "euclidean"):
        return abs(x-m)
    else :
        if(d_type == "mahalanobis"):
            return abs((x-m) / math.sqrt(meta_info[0]))
    
def cluster(data_points,m1,m2,oldm1,oldm2,var1,var2,d_type):
    if(oldm1 == m1 and oldm2 == m2) : return (m1,m2)
    else :
        data_split_1 = [x for x in data_points if distance(x,m1,d_type,[var1]) < distance(x,m2,d_type,[var2])]
        data_split_2 = [x for x in data_points if distance(x,m1,d_type,[var1]) > distance(x,m2,d_type,[var2])]
        equal_dist = [x for x in data_points if distance(x,m1,d_type,[var1]) == distance(x,m2,d_type,[var2])]
        data_split_1 += equal_dist[:len(equal_dist) / 2]
        data_split_2 += equal_dist[len(equal_dist)/2:]
        print "LOG : Clusters (Mean,Variance) are (%f,%f) , (%f,%f)" %(Mean(data_split_1),Variance(data_split_1),Mean(data_split_2),Variance(data_split_2))
        return cluster(data_points,Mean(data_split_1),Mean(data_split_2),m1,m2,Variance(data_split_1),Variance(data_split_2),d_type) 
        
def train(file_path,d_type,choose_random):
    data_points = GetData(file_path)
    assert len(data_points) != 0 , "No Data Points"
    mean = Mean(data_points)
    variance = Variance(data_points)
    if not random:
        centroids = cluster(data_points,mean+variance,mean-variance,mean+variance+1,mean-variance+1,variance,variance,d_type)
    else:
        rand1 = data_points[random.randrange(0,len(data_points))]
        rand2 = data_points[random.randrange(0,len(data_points))]
        centroids = cluster(data_points,rand1,rand2,rand1+1,rand2+1,variance,variance,d_type)               
    return centroids

def test(file_path,centroids,d_type):
    def confusion_matrix(p_class1,p_class2,threshold):
        c1_c1 = sum([p_class1[i] >= threshold for i in range(len(p_class1))])
        c1_c2 = sum([p_class1[i] < threshold for i in range(len(p_class1))])
        c2_c2 = sum([p_class2[i] <= threshold for i in range(len(p_class2))])
        c2_c1 = sum([p_class2[i] > threshold for i in range(len(p_class2))])
        return (c1_c1,c1_c2,c2_c2,c2_c1)
    def adjust_centroids(centroids,class1,class2):
        assert len(class1) == len(class2) , "Test Points are not equally distributed"
        #assumption class1 datapoints are close to centroids[0] then centroids[1]
        (class1_mean,claas1_variance) = (Mean(class1),Variance(class1))
        (class2_mean,claas2_variance) = (Mean(class2),Variance(class2))
        distance_class1_mu1 = [distance(x1,centroids[0],d_type,claas1_variance) for x1 in class1]
        distance_class2_mu1 = [distance(x2,centroids[0],d_type,claas2_variance) for x2 in class2]
        distance_class1_mu2 = [distance(x1,centroids[1],d_type,claas1_variance) for x1 in class1]
        distance_class2_mu2 = [distance(x2,centroids[1],d_type,claas2_variance) for x2 in class2]
        points_class1_near_mu1 = sum([distance_class1_mu1[i] < distance_class2_mu1[i] for i in range(len(distance_class2_mu2))])
        points_class1_near_mu2 = len(class1) - points_class1_near_mu1
        points_class2_near_mu1 = sum([distance_class2_mu1[i] < distance_class1_mu1[i] for i in range(len(distance_class2_mu2))])
        points_class2_near_mu2 = len(class2) - points_class2_near_mu1
        if(points_class1_near_mu1 > points_class2_near_mu1 and points_class1_near_mu2 < points_class2_near_mu2):
            return centroids
        if (points_class1_near_mu1 < points_class2_near_mu1 and points_class1_near_mu2 > points_class2_near_mu2):
            return (centroids[1],centroids[0])
        if (points_class1_near_mu1 > points_class2_near_mu1 and points_class1_near_mu2 > points_class2_near_mu2):
            if(points_class1_near_mu1 > points_class1_near_mu2) : return centroids
            else : return (centroids[1],centroids[0])
        if (points_class2_near_mu1 > points_class1_near_mu1 and points_class2_near_mu2 > points_class1_near_mu2):
            if(points_class2_near_mu2 > points_class2_near_mu1) : return centroids
            else : return (centroids[1],centroids[0])    
        return centroids
    test_points = GetData(file_path)
    assert len(test_points) % 2 == 0 , "Test Points are not equally distributed"
    (class1,class2) = (test_points[:len(test_points)/2],test_points[len(test_points)/2:])
    centroids = adjust_centroids(centroids,test_points[:len(test_points)/2],test_points[len(test_points)/2:])
    print "LOG : Modified centroid is "
    print centroids
    (class1_mean,claas1_variance) = (Mean(class1),Variance(class1))
    (class2_mean,claas2_variance) = (Mean(class2),Variance(class2))
    distance_class1_mu1 = [distance(x1,centroids[0],d_type,claas1_variance) for x1 in class1]
    distance_class2_mu1 = [distance(x2,centroids[0],d_type,claas2_variance) for x2 in class2]
    distance_class1_mu2 = [distance(x1,centroids[1],d_type,claas1_variance) for x1 in class1]
    distance_class2_mu2 = [distance(x2,centroids[1],d_type,claas2_variance) for x2 in class2]
    #assume m1 as positive
    probability_class1 = [distance_class1_mu1[i] / (distance_class1_mu1[i] + distance_class1_mu2[i]) for i in range(len(distance_class1_mu1))]
    probability_class2 = [distance_class2_mu1[i] / (distance_class2_mu1[i] + distance_class2_mu2[i]) for i in range(len(distance_class1_mu1))]
    prob_file = open("probability_class_1.txt","w")
    for i in probability_class1:
        prob_file.write(str(i) + " 0\n")
    for i in probability_class2:
        prob_file.write(str(i) + " 1\n")
    roc_file = open("roc.txt","w")
    precision = 1000
    confusion_list = [ confusion_matrix(probability_class1,probability_class2,t * 1.0/precision) for t in range(precision) ]
    for i in range(precision):
        roc_file.write(str(confusion_list[i][0]) + " " + str(confusion_list[i][1]) +  " " + str(confusion_list[i][2]) + " " + str(confusion_list[i][3]) + "\n")
    #print confusion_list
    class1_as_class1 = sum([distance_class1_mu1[i] < distance_class1_mu2[i] for i in range(len(class1))])
    class1_as_class2 = len(class1) - class1_as_class1
    class2_as_class2 = sum([distance_class2_mu2[i] < distance_class2_mu1[i] for i in range(len(class1))])
    class2_as_class1 = len(class2) - class2_as_class2
    print "LOG : Confustion Matrix : \n %d %d \n %d %d" %(class1_as_class1,class1_as_class2,class2_as_class1,class2_as_class2)

def main_driver():
    if len(sys.argv) < 2:
        print "Usage : python analyse.py euclidean/mahalanobis"
        exit(0)
    if sys.argv[1] != "euclidean" and sys.argv[1] != "mahalanobis":
        print "Usage : python analyse.py euclidean/mahalanobis <random>"
    else:
        if len(sys.argv) == 3:
            centroids = train("trainData.txt",sys.argv[1],True)
        else:
            centroids = train("trainData.txt",sys.argv[1],False)
        print "LOG : centroid is "
        print centroids
        test("testData.txt",centroids,"euclidean")
main_driver()
