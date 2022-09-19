import cv2
import numpy as np
import matplotlib.pyplot as plt


def SSD(x, y):
    ssd = np.sqrt(np.sum(np.square(x - y)))
    return ssd


def image_perspective(img1, img2,h):

    row1, col1, cor1 = img1.shape
    row2, col2, cor2 = img2.shape

    wrapping = cv2.warpPerspective(img1, h, (col1 + col2, row1 + row2))

    for i in range(row2):
        for j in range(col2):

            if np.sum(wrapping[i][j]) <= 0:
                wrapping[i][j] = img2[i][j]
            else:
                cum_sum = np.sum(img2[i][j])

                if np.sum(wrapping[i][j]) <= cum_sum:
                    wrapping[i][j] = img2[i][j]
                else:
                    wrapping[i][j] = wrapping[i][j]
    return wrapping

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(img1, None)
    kp_2, desc_2 = sift.detectAndCompute(img2, None)
    count = 0
    id_desc1 = []
    id_desc2 = []
    match_array = {}
    file1 = open('problem 1:match_array.txt', 'w')
    for i in range(len(desc_1)):
        for j in range(len(desc_2)):
            x = desc_1[i]
            y = desc_2[j]
            ssd = SSD(x, y)
            # print(" Distance between each descriptor", ssd)
            if ssd < 170:  # sorting out values where distance is less than 150
                count = count + 1
                # print("position of descriptor distance less than 150", i, k)
                id_desc1.append(i)
                id_desc2.append(j)
                match_array[kp_1[i]] = kp_2[j]
                # print(matching)
            # print(idx_desc1,idx_desc2)

    scr_key = np.float32([kp_src.pt for kp_src in match_array.keys()])
    src = scr_key.reshape(-1, 1, 2)

    des_val = np.float32([kp_des.pt for kp_des in match_array.values()])
    des = des_val.reshape(-1, 1, 2)

    for i in range(len(src)):
        str1: str = str(src[i]) + " : " + str(des[i]) + "\n"
        file1.write(str1)
    file1.close()

    h, s = cv2.findHomography(src, des, cv2.RANSAC, 5.0)

    dst = image_perspective(img1, img2, h)
    cv2.imwrite(savepath, dst)

    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
