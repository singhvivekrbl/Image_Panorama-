# 1. Only add your code inside the function (including newly improted packages).
#  You can design a new function and call the new function in the given functions.
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import json
# import matplotlib.pyplot as plt
# import math
def SSD(x, y):
    ssd = np.sqrt(np.sum(np.square(x - y)))
    return ssd


def homography_perspective(dst_pts, imgs, src_pts):
    row1, col1, cor1 = imgs[0].shape
    row2, col2, cor2 = imgs[1].shape

    # creating the homography matrix
    h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # transforming the image with perspective tne image 2
    trans_point = cv2.perspectiveTransform(
        np.float32(np.array([[0, 0], [0, row2], [col2, row2], [col2, 0]])).reshape((-1, 1, 2)), h)

    # concatenating the image shapes
    concat_point = np.concatenate(
        (np.float32(np.array([[0, 0], [0, row1], [col1, row1], [col2, 0]])).reshape((-1, 1, 2)), trans_point), axis=0)

    # min points for translation
    [xmn, ymn] = np.int32(concat_point.min(axis=0).flatten())
    homo_trans = np.array([[1, 0, -xmn], [0, 1, -ymn], [0, 0, 1]])

    res = homo_trans.dot(h)
    [xmx, ymx] = np.int32(concat_point.max(axis=0).flatten())

    result = cv2.warpPerspective(imgs[0], res, (xmx - xmn, ymx - ymn))
    result[abs(ymn):row2 + abs(ymn), abs(xmn): col2 + abs(xmn)] = imgs[1]
    return result

def stitch(imgmark, N, savepath=''):
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    sift = cv2.xfeatures2d.SIFT_create(600)
    # First Loop
    ###############################################################################
    # cv2.imwrite("image_x",imgs[0])
    kp_1, desc_1 = sift.detectAndCompute(imgs[0], None)
    kp_2, desc_2 = sift.detectAndCompute(imgs[1], None)
    kp_3, desc_3 = sift.detectAndCompute(imgs[2], None)
    kp_4, desc_4 = sift.detectAndCompute(imgs[3], None)

    count = 0
    id_desc1 = []
    id_desc2 = []
    match_array = {}
    file1 = open('problem 2: match_array_combination_1.txt', 'w')
    for i in range(len(desc_1)):
        for j in range(len(desc_2)):
            x = desc_1[i]
            y = desc_2[j]
            ssd = SSD(x, y)
            # print(" Distance between each descriptor", ssd)
            if ssd < 60:  # sorting out values where distance is less than 150
                count = count + 1
                # print("position of descriptor distance less than 70", i, k)
                id_desc1.append(i)
                id_desc2.append(j)
                match_array[kp_1[i]] = kp_2[j]
                # print(matching)
            # print(id_desc1,id_desc2)

    scr_key = np.float32([kp_src.pt for kp_src in match_array.keys()])
    src = scr_key.reshape(-1, 1, 2)

    des_val = np.float32([kp_des.pt for kp_des in match_array.values()])
    des = des_val.reshape(-1, 1, 2)

    for i in range(len(src)):
        str1: str = str(src[i]) + " : " + str(des[i]) + "\n"
        file1.write(str1)
    file1.close()

    result = homography_perspective(des, [imgs[0], imgs[1]], src)

    # second Loop
    ###############################################################################
    kp_5, desc_5 = sift.detectAndCompute(result, None)
    kp_3, desc_3 = sift.detectAndCompute(imgs[2], None)

    count = 0
    id_desc5 = []
    id_desc3 = []
    match_array = {}
    file1 = open('problem 2: match_array_combination_2.txt', 'w')
    for i in range(len(desc_5)):
        for j in range(len(desc_3)):
            x = desc_5[i]
            y = desc_3[j]
            ssd = SSD(x, y)
            # print(" Distance between each descriptor", ssd)
            if ssd < 100:  # sorting out values where distance is less than 100
                count = count + 1
                # print("position of descriptor distance less than 70", i, k)
                id_desc5.append(i)
                id_desc3.append(j)
                match_array[kp_5[i]] = kp_3[j]
                # print(matching)
            # print(id_desc1,id_desc2)

    scr_key = np.float32([kp_src.pt for kp_src in match_array.keys()])
    src = scr_key.reshape(-1, 1, 2)

    des_val = np.float32([kp_des.pt for kp_des in match_array.values()])
    des = des_val.reshape(-1, 1, 2)

    for i in range(len(src)):
        str1: str = str(src[i]) + " : " + str(des[i]) + "\n"
        file1.write(str1)
    file1.close()

    result1 = homography_perspective(des, [result, imgs[2]], src)
    # cv2.imshow("image", result1)
    # cv2.waitKey(0)
    # cv2.imwrite(savepath, result1)

    # third Loop
    ###############################################################################
    kp_6, desc_6 = sift.detectAndCompute(result1, None)
    kp_4, desc_4 = sift.detectAndCompute(imgs[3], None)
    count = 0
    id_desc6 = []
    id_desc4 = []
    match_array = {}
    file1 = open('problem 2: match_array_combination_3.txt', 'w')
    for i in range(len(desc_6)):
        for j in range(len(desc_4)):
            x = desc_6[i]
            y = desc_4[j]
            ssd = SSD(x, y)
            # print(" Distance between each descriptor", ssd)
            if ssd < 100:  # sorting out values where distance is less than 150
                count = count + 1
                # print("position of descriptor distance less than 70", i, k)
                id_desc6.append(i)
                id_desc4.append(j)
                match_array[kp_6[i]] = kp_4[j]
                # print(matching)
            # print(id_desc1,id_desc2)

    scr_key = np.float32([kp_src.pt for kp_src in match_array.keys()])
    src = scr_key.reshape(-1, 1, 2)

    des_val = np.float32([kp_des.pt for kp_des in match_array.values()])
    des = des_val.reshape(-1, 1, 2)

    for i in range(len(src)):
        str1: str = str(src[i]) + " : " + str(des[i]) + "\n"
        file1.write(str1)
    file1.close()

    result2 = homography_perspective(des, [result1, imgs[3]], src)

   # # If images are greater than 4
    # ###############################################################################
    # if len(imgs) >3:
    #     print("true")
    #     # forth Loop
    #     ###############################################################################
    #     kp_7, desc_7 = sift.detectAndCompute(result2, None)
    #     kp_5, desc_5 = sift.detectAndCompute(imgs[4], None)
    #     count = 0
    #     id_desc7 = []
    #     id_desc5 = []
    #     match_array = {}
    #     file1 = open('problem 2: match_array_combination_5.txt', 'w')
    #     for i in range(len(desc_7)):
    #         for j in range(len(desc_5)):
    #             x = desc_7[i]
    #             y = desc_5[j]
    #             ssd = SSD(x, y)
    #             # print(" Distance between each descriptor", ssd)
    #             if ssd < 100:  # sorting out values where distance is less than 150
    #                 count = count + 1
    #                 # print("position of descriptor distance less than 70", i, k)
    #                 id_desc7.append(i)
    #                 id_desc5.append(j)
    #                 match_array[kp_7[i]] = kp_5[j]
    #                 # print(matching)
    #             # print(id_desc1,id_desc2)
    #
    #     scr_key = np.float32([kp_src.pt for kp_src in match_array.keys()])
    #     src = scr_key.reshape(-1, 1, 2)
    #
    #     des_val = np.float32([kp_des.pt for kp_des in match_array.values()])
    #     des = des_val.reshape(-1, 1, 2)
    #
    #     for i in range(len(src)):
    #         str1: str = str(src[i]) + " : " + str(des[i]) + "\n"
    #         file1.write(str1)
    #     file1.close()
    #
    #     result3 = homography_perspective(des, [result2, imgs[4]], src)
    #
    #     # cv2.imshow("image", result3)
    #     # cv2.waitKey(0)
    #     # cv2.imwrite(savepath, result3)
    #
    #     # fifth Loop
    #     ###############################################################################
    #     kp_8, desc_8 = sift.detectAndCompute(result3, None)
    #     kp_6, desc_6 = sift.detectAndCompute(imgs[5], None)
    #     count = 0
    #     id_desc8 = []
    #     id_desc6 = []
    #     match_array = {}
    #     file1 = open('problem 2: match_array_combination_6.txt', 'w')
    #     for i in range(len(desc_8)):
    #         for j in range(len(desc_6)):
    #             x = desc_8[i]
    #             y = desc_6[j]
    #             ssd = SSD(x, y)
    #             # print(" Distance between each descriptor", ssd)
    #             if ssd < 100:  # sorting out values where distance is less than 150
    #                 count = count + 1
    #                 # print("position of descriptor distance less than 70", i, k)
    #                 id_desc8.append(i)
    #                 id_desc6.append(j)
    #                 match_array[kp_8[i]] = kp_6[j]
    #                 # print(matching)
    #             # print(id_desc1,id_desc2)
    #
    #     scr_key = np.float32([kp_src.pt for kp_src in match_array.keys()])
    #     src = scr_key.reshape(-1, 1, 2)
    #
    #     des_val = np.float32([kp_des.pt for kp_des in match_array.values()])
    #     des = des_val.reshape(-1, 1, 2)
    #
    #     for i in range(len(src)):
    #         str1: str = str(src[i]) + " : " + str(des[i]) + "\n"
    #         file1.write(str1)
    #     file1.close()
    #
    #     result4 = homography_perspective(des, [result3, imgs[5]], src)
    #     cv2.imshow("image", result4)
    #     cv2.waitKey(0)
    #     cv2.imwrite(savepath, result4)

    val = []
    dict_desc = {1: desc_1, 2: desc_2, 3: desc_3, 4: desc_4}
    
    # if condition if image is more than 4
    # change variable name
    # if len(imgs)>3:
    #     dict_desc = {1: desc_1, 2: desc_2, 3: desc_3, 4: desc_4, 5: desc_5, 6: desc_6}
    #     for ii in range(1, N):
    #         d1 = dict_desc[ii]
    #         for jj in range(1, N):
    #
    #             d2 = dict_desc[jj]
    #             count = 0
    #             for i in range(len(d1)):
    #                 for k in range(len(d2)):
    #
    #                     dis = SSD(d1[i], d2[k])
    #                     if dis < 100:
    #                         count = count + 1
    #             if count > 2:
    #                 m.append(1)
    #             else:
    #                 m.append(0)
    #     overlap_arr = np.array(m)
    #     return overlap_arr

    # overlap area matrix

    


    for i in range(1, N):
        # select range for first image
        d1 = dict_desc[i]
        for j in range(1, N):
            # select range for second image
            d2 = dict_desc[j]
            count = 0
            for k in range(len(d1)):  # looping over the range image 1
                for l in range(len(d2)):  # looping over the range image 2
                    dis = SSD(d1[i], d2[l])  # sum of square distance calculation
                    if dis < 100:
                        count = count + 1
            if count > 2:
                val.append(1)
            else:
                val.append(0)
    overlap_arr = np.array(val)
    cv2.imwrite(savepath, result2)
    return overlap_arr

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', N=5, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)