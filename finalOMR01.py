import cv2
import numpy as np

def users_directory():
    print("Welcome to UniqoXOMR")
    n=int(input("Number of questions (<36): "))  
    m=float(input("Mark per question: "))
    ne=float(input("Negative marking per question: "))
    set1=[2, 3, 4, 1, 4, 2, 1, 3, 1, 2, 4, 3, 4, 1, 1, 3, 3, 2, 1, 2, 2, 4, 3, 1, 3, 1, 2, 4, 1, 3, 1, 2, 2, 4, 2]
    set2=[4, 1, 1, 3, 3, 3, 1, 2, 2, 2, 1, 4, 3, 1, 2, 2, 4, 3, 1, 1, 3, 2, 4, 4, 2, 3, 4, 1, 2, 4, 2, 2, 1, 3, 1]
    set3=[2, 4, 3, 2, 3, 1, 3, 1, 4, 1, 2, 1, 2, 4, 2, 2, 3, 4, 1, 4, 3, 1, 1, 1, 3, 3, 2, 4, 4, 2, 1, 3, 2, 4, 2]
    set4=[2, 3, 4, 1, 2, 1, 4, 3, 1, 1, 4, 1, 3, 1, 2, 3, 4, 2, 4, 1, 2, 3, 1, 2, 3, 2, 4, 2, 3, 3, 4, 1, 2, 4, 1]
    
    # print("Answers of set 1: ")
    # for i in range(n):
    #     e = int(input())
    #     set1.append(e)
    #     i+=1
    # print("Answers of set 2: ")
    # for i in range(n):
    #     e = int(input())
    #     set2.append(e)
    #     i+=1
    # print("Answers of set 3: ")
    # for i in range(n):
    #     e = int(input())
    #     set3.append(e)
    #     i+=1

    if n>19:
        set1box1 = set1[:19]
        set1box2 = set1[19:]
        set2box1 = set2[:19]
        set2box2 = set2[19:]
        set3box1 = set3[:19]
        set3box2 = set3[19:]
        set4box1 = set4[:19]
        set4box2 = set4[19:]
    else:
        set1box1 = set1
        set2box1 = set2
        set3box1 = set3
        set1box2 = [0]
        set2box2 = [0]
        set3box2 = [0]

    return  m,n,ne,set1box1 ,set1box2 ,set2box1 ,set2box2 ,set3box1 ,set3box2 ,set4box1 ,set4box2


def reorder (biggest):
    biggest = biggest.reshape((4,2))
    # print("biggest\n",biggest)

    newbig = np.zeros((4,1,2),np.int32)

    add = biggest.sum(1)
    # print(add)

    newbig[0]= biggest[np.argmin(add)]
    newbig[3]= biggest[np.argmax(add)]

    diff = np.diff(biggest,axis=1)
    # print(diff)

    newbig[1] = biggest[np.argmin(diff)]
    newbig[2] = biggest[np.argmax(diff)]

    # print("new: ",newbig)

    return newbig


def crop_if_needed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny_img = cv2.Canny(gray, 150, 350)         # Detect edges

    kernel = np.ones((7,7),np.uint8)
    dilation_image = cv2.dilate(canny_img,kernel,iterations=2)
    erode_image = cv2.erode(dilation_image,kernel,iterations=1)

    contours, hirearchy = cv2.findContours(erode_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours

    # print("contours: ", len(contours))

    maxarea = 0
    largest_contour = np.array([]) 
    for cnt in contours: 
        area = cv2.contourArea(cnt)
        # print("AREA: ",area)
        cv2.drawContours(image,cnt,-1,(255,0,0),3)
        if area > 10000:
            peri = cv2.arcLength(cnt,True)
            approx =  cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxarea and len(approx)==4:
                largest_contour = approx
    # print("biggest: ",largest_contour)
    # print("Biggest Shape", largest_contour.shape)

    # print(cv2.contourArea(largest_contour)) 

    largest_contour = reorder(largest_contour)

    height = 700
    width = 640

    pts1=np.float32(largest_contour)
    pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    image_wrapped = cv2.warpPerspective(image,matrix, (width,height))
    
    return image_wrapped


def detect_squares(image_copy):
    # Scanning the document
    image = crop_if_needed(image_copy)
    # print(image.shape)
    image = image[5:700-5,5:640-5]
    # cv2.imshow("scanned image",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 50, 150)

    contours, hirearchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # print(len(contours))

    squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            squares.append(approx)
    
    # print(len(squares))

    sorted_squares = sorted(squares, key=cv2.contourArea, reverse=True)

    cv2.drawContours(image,sorted_squares,-1,(0,255,0),2)
    # cv2.imshow("scanned image",image)

    # Draw detected squares' corners on the image
    for square in squares:
        for point in square:
            cv2.circle(image, tuple(point[0]), 4, (255,0,0), -1)

    # cv2.imshow("Points", image)

    areaListSort = []
    for cnt in sorted_squares:
        areaListSort.append(cv2.contourArea(cnt))

    # print(areaListSort)

    ###################################### SET  BOX #####################################################################
    cv2.drawContours(image,sorted_squares[3],-1,(0,255,0),3)
    # cv2.imshow("SET BOX" , image)

    x, y, w, h = cv2.boundingRect(sorted_squares[3])

    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    n = 4  # Divide the square into n(number of sets) regions assuming the circles are evenly distributed

    circle_radius = w // 16  # The circle's diameter is about 1/8th the width of the square
    circle_centers = []
    for i in range(0, n):
        circle_centers.append((x+((2*i+1)*w)//8, y+h//2))
    # print(circle_centers)

    for point in circle_centers:
        cv2.circle(image, point, radius=5, color=(0, 255, 0), thickness=2)  # -1 thickness fills the circle
    # cv2.imshow('Marked Points', image)

    marked_circle_index = None
    min_intensity = 255         # Max possible intensity for a white pixel
    for i, center in enumerate(circle_centers):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, center, circle_radius, 255, -1)

        # cv2.imshow("mask",mask)

        # Calculate the average intensity within the circle
        circle_intensity = cv2.mean(gray, mask=mask)[0]
        # print(circle_intensity)

        # Check if this circle is the darkest (min-value) so far
        if circle_intensity < min_intensity:
            min_intensity = circle_intensity
            marked_circle_index = i

    print(f"SET no: {marked_circle_index + 1}")
    marked_set = marked_circle_index+1

    ####################################### ROLL BOX ######################################
    cv2.drawContours(image,sorted_squares[2],-1,(0,255,0),3)
    # cv2.imshow("ROLL BOX" , image)

    x, y, w, h = cv2.boundingRect(sorted_squares[2])

    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7
    
    # roll = roll_detection(image, x, y, w, h)


    ################################### 1st Box #########################################
    
    cv2.drawContours(image,sorted_squares[0],-1,(0,255,0),3)
    # cv2.imshow("1st Box" , image)

    x, y, w, h = cv2.boundingRect(sorted_squares[0])

    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    circle_sets1=[]
    for i in range(0, 19):
        circle_centers=[]
        for j in range(0, 4):
            circle_centers.append((x+((2*j+1)*w)//8, (2*y + (2*i+1)*h//19) // 2))
        circle_sets1.append(circle_centers)
    # print(circle_sets1)

    # for point_list in circle_sets1:
    #     for point in point_list:
    #        cv2.circle(image, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green circles

    # cv2.imshow("Marked Points", image)
        
    marked_index_1=[]
    for circle_center in circle_sets1:
        marked_circle_index = None
        min_intensity = 255    # Max possible intensity for a white pixel
        for i, center in enumerate(circle_center):
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, center, circle_radius, 255, -1)

            circle_intensity = cv2.mean(gray, mask=mask)[0]
            # print(circle_intensity)

            if circle_intensity < min_intensity:
                min_intensity = circle_intensity
                marked_circle_index = i
        marked_index_1.append(marked_circle_index+1)
    print("1 - 19 Selected Options: ", marked_index_1)

    ##################################### 2nd Box ###############################################

    cv2.drawContours(image,sorted_squares[1],-1,(0,255,0),3)
    # cv2.imshow("2nd Box" , image)

    x, y, w, h = cv2.boundingRect(sorted_squares[1])

    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    circle_sets2=[]
    for i in range(0, 16):
        circle_centers=[]
        for j in range(0, 4):
            circle_centers.append((x+((2*j+1)*w)//8, (2*y + (2*i+1)*h//16) // 2))
        circle_sets2.append(circle_centers)
    # print(circle_sets2)

    marked_index_2=[]
    for circle_center in circle_sets2:
        marked_circle_index = None
        min_intensity = 255  
        for i, center in enumerate(circle_center):

            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, center, circle_radius, 255, -1)

            circle_intensity = cv2.mean(gray, mask=mask)[0]
            # print(circle_intensity)

            # Check if this circle is the darkest (min-value) so far
            if circle_intensity < min_intensity:
                min_intensity = circle_intensity
                marked_circle_index = i
        marked_index_2.append(marked_circle_index+1)
    print("20 - 35 Selected Options: ", marked_index_2)



    return sorted_squares,areaListSort,image,marked_set,marked_index_1,circle_sets1,marked_index_2,circle_sets2


def mark(m,n,ne,marked_index_1,marked_index_2,setbox1,setbox2):
    total=0
    correct1=[]
    correct2=[]
    for elem1, elem2 in zip(marked_index_1, setbox1):
        if elem1 == elem2:
            correct1.append(1)
            total += m
        else:
            correct1.append(0)
            total -= ne

    for elem1, elem2 in zip(marked_index_2, setbox2):
        if elem1 == elem2:
            correct2.append(1)
            total += m
        else:
            correct2.append(0)
            total -= ne
    return total,correct1,correct2    


def digital_copy(image, correct1,correct2,sorted_squares,marked_index_1,circle_sets1,marked_index_2,circle_sets2,n,m,setbox1 ,setbox2):
    
    ############################################# Grade Box ###########################################
    x, y, w, h = cv2.boundingRect(sorted_squares[4])

    cv2.putText(image,f"{round(marks,2)}/{n*m}",(x+5,y+h-20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1,cv2.LINE_AA)    


    ############################################ 1st Box ####################################################
    x, y, w, h = cv2.boundingRect(sorted_squares[0])
    x+=4
    y+=4
    w-=7
    h-=7

    circle_radius = w//16

    for center,marked_index,is_correct,correct_index in zip(circle_sets1,marked_index_1,correct1,setbox1):
        for i, center in enumerate(center):
            colour=(0,255,0)
            x,y = center
            if i == marked_index-1:
                # print("yes")
                if is_correct:
                    colour = (0,255,0)
                    cv2.circle(image,center,circle_radius,colour,-1)
                else :
                    colour = (0,0,255)
                    cv2.circle(image,center,circle_radius,colour,-1)  
            elif i == correct_index-1:
                color = (0,255,0)          
                cv2.circle(image,center,circle_radius,colour,-1)      

    x, y, w, h = cv2.boundingRect(sorted_squares[1])
    x+=4
    y+=4
    w-=7
    h-=7

    for center,marked_index,is_correct,correct_index in zip(circle_sets2,marked_index_2,correct2,setbox2):
        for i, center in enumerate(center):
            colour=(0,255,0)
            x,y = center
            if i == marked_index-1:
                # print("yes")
                if is_correct:
                    colour = (0,255,0)
                    cv2.circle(image,center,circle_radius,colour,-1)
                else :
                    colour = (0,0,255)
                    cv2.circle(image,center,circle_radius,colour,-1)  
            elif i == correct_index-1:
                color = (0,255,0)
                cv2.circle(image,center,circle_radius,colour,-1)          
             
    return image



m,n,ne,set1box1 ,set1box2 ,set2box1 ,set2box2 ,set3box1 ,set3box2 ,set4box1 ,set4box2 = users_directory()

path = input("Give the path of your scanned OMR: ")

img = cv2.imread(path)
image_copy = img.copy()

sorted_squares,areaListSort,image,marked_set,marked_index_1,circle_sets1,marked_index_2,circle_sets2 = detect_squares(image_copy)

if marked_set == 1:
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set1box1,set1box2)
    image_digital = digital_copy(image, correct1,correct2,sorted_squares,marked_index_1,circle_sets1,marked_index_2,circle_sets2,n,m,set1box1 ,set1box2)
elif marked_set == 2:
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set2box1,set2box2)
    image_digital = digital_copy(image, correct1,correct2,sorted_squares,marked_index_1,circle_sets1,marked_index_2,circle_sets2,n,m,set2box1 ,set2box2)
elif marked_set == 3:
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set3box1,set3box2)
    image_digital = digital_copy(image, correct1,correct2,sorted_squares,marked_index_1,circle_sets1,marked_index_2,circle_sets2,n,m,set3box1 ,set3box2)
else: 
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set4box1,set4box2)
    image_digital = digital_copy(image, correct1,correct2,sorted_squares,marked_index_1,circle_sets1,marked_index_2,circle_sets2,n,m,set4box1 ,set4box2)


print("Mark: " , marks)

cv2.imshow("main",img)
cv2.imshow("digital",image_digital)

cv2.waitKey(0)
cv2.destroyAllWindows()


## E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Computer Vision\\Udemy Course\\OpenCV Project-8 (OMR)\\Pictures\\fullTest .jpg