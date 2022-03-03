import cv2
import numpy as np                      #导入python相关库
C = 2
M = 2
EPSILON = 0.001                       #定义变量
def get_init_fuzzy_mat(pixel_count):         #定义得到初始化数据，聚类中心的函数
    global C
    fuzzy_mat = np.zeros((C, pixel_count))
    for col in range(pixel_count):
        temp_sum = 0
        randoms = np.random.rand(C - 1, 1)
        for row in range(C - 1):
            fuzzy_mat[row, col] = randoms[row, 0] * (1 - temp_sum)
            temp_sum += fuzzy_mat[row, col]
        fuzzy_mat[-1, col] = 1 - temp_sum
    return fuzzy_mat
def get_centroids(data_array, fuzzy_mat):            #定义得到聚类中心的函数
    global M
    class_num, pixel_count = fuzzy_mat.shape[:2]
    centroids = np.zeros((class_num, 1))
    for i in range(class_num):
        fenzi = 0.
        fenmu = 0.
        for pixel in range(pixel_count):
            fenzi += np.power(fuzzy_mat[i, pixel], M) * data_array[0, pixel]
            fenmu += np.power(fuzzy_mat[i, pixel], M)
        centroids[i, 0] = fenzi / fenmu
    return centroids
def eculidDistance(vectA, vectB):                 #定义计算距离的函数
    return np.sqrt(np.sum(np.power(vectA - vectB, 2)))
def eculid_distance(pixel_1, pixel_2):             #定义距离计算函数
    return np.power(pixel_1 - pixel_2, 2)             #返回值为欧式距离表示
def cal_fcm_function(fuzzy_mat, centroids, data_array):   #定义fcm的计算式
    global M
    class_num, pixel_count = fuzzy_mat.shape[:2]
    target_value = 0.0
    for c in range(class_num):
        for p in range(pixel_count):
            target_value += eculid_distance(data_array[0, p], centroids[c, 0]) * np.power(fuzzy_mat[c, p], M)
    return target_value
def get_label(fuzzy_mat, data_array):            #定义计算标签函数
    pixel_count = data_array.shape[1]
    label = np.zeros((1, pixel_count))
    for i in range(pixel_count):
        if fuzzy_mat[0, i] > fuzzy_mat[1, i]:    #大于目标值则将其像素设置为0
            label[0, i] = 0
        else:
            label[0, i] = 255                #否则将其像素设置为1
    return label                            #返回标签
def cal_fuzzy_mat(data_array, centroids):         #定义新的fcm计算式，聚类中心计算
    global M
    pixel_count = data_array.shape[1]
    class_num = centroids.shape[0]
    new_fuzzy_mat = np.zeros((class_num, pixel_count))
    for p in range(pixel_count):
        for c in range(class_num):
            temp_sum = 0.
            Dik = eculid_distance(data_array[0, p], centroids[c, 0])
            for i in range(class_num):
                temp_sum += np.power(Dik / (eculid_distance(data_array[0, p], centroids[i, 0])), (1 / (M - 1)))
            new_fuzzy_mat[c, p] = 1 / temp_sum
    return new_fuzzy_mat
def fcm(init_fuzzy_mat, init_centroids, data_array):         #定义迭代函数
    global EPSILON
    last_target_function = cal_fcm_function(init_fuzzy_mat, init_centroids, data_array)
    print("迭代次数 = 1, 目标函数值 = {}".format(last_target_function))
    fuzzy_mat = cal_fuzzy_mat(data_array, init_centroids)
    centroids = get_centroids(data_array, fuzzy_mat)
    target_function = cal_fcm_function(fuzzy_mat, centroids, data_array)
    print("迭代次数 = 2, 目标函数值 = {}".format(target_function))
    count = 3
    while count < 20:                               #一共迭代20次，进行优化
        if abs(target_function - last_target_function) <= EPSILON:
            break
        else:
            last_target_function = target_function
            fuzzy_mat = cal_fuzzy_mat(data_array, centroids)
            centroids = get_centroids(data_array, fuzzy_mat)
            target_function = cal_fcm_function(fuzzy_mat, centroids, data_array)
            print("迭代次数 = {}, 目标函数值 = {}".format(count, target_function))
            count += 1
    return fuzzy_mat, centroids, target_function
image = cv2.imread(r"img1.jpg", cv2.IMREAD_GRAYSCALE)     #读取图像文件
rows, cols = image.shape[:2]                                  #得到行列
pixel_count = rows * cols
image_array = image.reshape(1, pixel_count)                     #图像矩阵转换得到初始模糊矩阵
# print(image_array[1])
init_fuzzy_mat = get_init_fuzzy_mat(pixel_count)                #初始化聚类中心
init_centroids = get_centroids(image_array, init_fuzzy_mat)
fuzzy_mat, centroids, target_function = fcm(init_fuzzy_mat, init_centroids, image_array)
label = get_label(-fuzzy_mat, -image_array)                    #上述三行是调用函数的计算过程
new_image = label.reshape(rows, cols)                        #将得到的结果返回为图像原始形状
cv2.imshow("result", new_image)                            #显示处理后的图像
cv2.imwrite("fcm_result.jpg", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
