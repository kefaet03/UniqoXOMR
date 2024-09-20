import cv2
import numpy as np


def roll_detection():
    print()

def crop_if_needed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest external contour is the OMR sheet
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the area coverage of the largest contour relative to the image size
        image_area = image.shape[0] * image.shape[1]
        contour_area = w * h
        coverage_ratio = contour_area / image_area
        # print(coverage_ratio)
        
        # If the largest contour covers a significant portion of the image, we assume no need to crop
        if coverage_ratio > 0.3:  # This threshold can be adjusted based on needs
            # Crop the image to the largest contour
            cropped_image = image[y+10:y+h-10, x+10:x+w-10]
            # print("Cropped!")
            return cropped_image
    
    return image


def detect_squares(image_copy):
    # Read the image
    image = crop_if_needed(image_copy)
    # cv2.imshow("Scanned",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    squares = []
    for cnt in contours:
        # Approximate the contour
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # If the approximated contour has 4 points, we consider it as a square
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 300:  # Can adjust this value according to needs
                squares.append(approx)
    # squares.sort()
    # print(squares)


    # Draw detected squares' corners on the image
    for square in squares:
        # print(square.shape)
        for point in square:
            cv2.circle(image, tuple(point[0]), 4, (255,0,0), -1)

    # cv2.imshow("xyz", image)
    # cv2.waitKey(0)

    # for point in squares[0]:
    #     cv2.circle(image, tuple(point[0]), 4, colors[color_index % len(colors)], -1)
        
    x=0
    y=0 
    w=0
    h=0 
    areaList = []
    areaListSort = []
    for i in range(0, len(squares)):
        x, y, w, h = cv2.boundingRect(squares[i])
        areaList.append(w*h)
        areaListSort.append(w*h)
    areaListSort.sort()
    # print(areaList)
    # print(areaListSort)

    # Assuming the circles are relatively small compared to the square and are placed in the middle
    # Calculate the bounding rectangle to get the square's position and size

    for i, num in enumerate(areaList):
        if num == areaListSort[len(areaListSort)-4]:
            x, y, w, h = cv2.boundingRect(squares[i]) # Select the SET Box
            break
    # print(squares[-3])
    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    # Divide the square into 4 regions assuming the circles are evenly distributed
    # Check each region for a filled circle
    circle_radius = w // 12  # The circle's diameter is about 1/8th the width of the square
    circle_centers = []
    for i in range(0, 4):
        circle_centers.append(((2*x + (2*i+1) * w // 4)//2, (2*y + h) // 2))
    # print(circle_centers)

    marked_circle_index = None
    min_intensity = 255  # Max possible intensity for a white pixel
    for i, center in enumerate(circle_centers):
        # Create a mask for the circle region
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, center, circle_radius, 255, -1)
        # cv2.circle(image, center, circle_radius, 200, -1)

        # Calculate the average intensity within the circle
        circle_intensity = cv2.mean(gray, mask=mask)[0]
        # print(circle_intensity)

        # Check if this circle is the darkest (min-value) so far
        if circle_intensity < min_intensity:
            min_intensity = circle_intensity
            marked_circle_index = i
    print(f"SET no: {marked_circle_index + 1}")
    marked_set = marked_circle_index+1

    # for i, num in enumerate(areaList):
    #     if num == areaListSort[len(areaListSort)-3]:
    #         x, y, w, h = cv2.boundingRect(squares[i]) # Select the roll Box
    #         break
    
    # roll = roll_detection(image, x, y, w, h)
    
    for i, num in enumerate(areaList):
        if num == areaListSort[len(areaListSort)-1]:
            x, y, w, h = cv2.boundingRect(squares[i]) # Select the biggest Box
            break
    # print(squares[0])
    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    circle_sets=[]
    circle_centers = []
    for i in range(0, 19):
        circle_centers=[]
        for j in range(0, 4):
            circle_centers.append(((2*x + (2*j+1) * w // 4)//2, (2*y + (2*i+1)*h//19) // 2))
        circle_sets.append(circle_centers)
    # print(circle_sets)
        
    marked_index_1=[]
    for circle_center in circle_sets:
        marked_circle_index = None
        min_intensity = 255  # Max possible intensity for a white pixel
        for i, center in enumerate(circle_center):
            # Create a mask for the circle region
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, center, circle_radius, 255, -1)
            # cv2.circle(image, center, circle_radius, 200, -1)

            # Calculate the average intensity within the circle
            circle_intensity = cv2.mean(gray, mask=mask)[0]
            # print(circle_intensity)

            # Check if this circle is the darkest (min-value) so far
            if circle_intensity < min_intensity:
                min_intensity = circle_intensity
                marked_circle_index = i
        marked_index_1.append(marked_circle_index+1)
    print("1 - 19 Selected Options: ", marked_index_1)

    for i, num in enumerate(areaList):
        if num == areaListSort[len(areaListSort)-2]:
            x, y, w, h = cv2.boundingRect(squares[i]) # Select the 2nd biggest Box
            break
    # print(squares[1])
    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    circle_sets=[]
    circle_centers = []
    for i in range(0, 16):
        circle_centers=[]
        for j in range(0, 4):
            circle_centers.append(((2*x + (2*j+1) * w // 4)//2, (2*y + (2*i+1)*h//16) // 2))
        circle_sets.append(circle_centers)
    # print(circle_sets)

    marked_index_2=[]
    for circle_center in circle_sets:
        marked_circle_index = None
        min_intensity = 255  # Max possible intensity for a white pixel
        for i, center in enumerate(circle_center):
            # Create a mask for the circle region
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, center, circle_radius, 255, -1)
            # cv2.circle(image, center, circle_radius, 200, -1)

            # Calculate the average intensity within the circle
            circle_intensity = cv2.mean(gray, mask=mask)[0]
            # print(circle_intensity)

            # Check if this circle is the darkest (min-value) so far
            if circle_intensity < min_intensity:
                min_intensity = circle_intensity
                marked_circle_index = i
        marked_index_2.append(marked_circle_index+1)
    print("20 - 35 Selected Options: ", marked_index_2)

    # Show the output image
    # cv2.namedWindow('Squares', cv2.WINDOW_NORMAL)
    # cv2.imshow('Squares', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return squares,areaList,areaListSort,image,marked_set,marked_index_1,marked_index_2


def users_directory():
    print()
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


def digital_copy(image, correct1,correct2,areaList,areaListSort,squares,marked_index_1,marked_index_2,marks,n,m,setbox1 ,setbox2):
    # print(areaListSort)
    for i, num in enumerate(areaList):
        if num == areaListSort[len(areaListSort)-5]:
            x, y, w, h = cv2.boundingRect(squares[i]) # Select the grade Box
            break

    cv2.putText(image,f"{round(marks,2)}/{n*m}",(x+5,y+h-20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1,cv2.LINE_AA)    

    for i, num in enumerate(areaList):
        if num == areaListSort[len(areaListSort)-1]:
            x, y, w, h = cv2.boundingRect(squares[i]) # Select the biggest Box
            break
    # print(squares[0])
    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    circle_radius = w//12

    circle_sets=[]
    circle_centers = []
    for i in range(0, 19):
        circle_centers=[]
        for j in range(0, 4):
            circle_centers.append(((2*x + (2*j+1) * w // 4)//2, (2*y + (2*i+1)*h//19) // 2))
        circle_sets.append(circle_centers)
    # print(circle_sets)

    
    for center,marked_index,is_correct,correct_index in zip(circle_sets,marked_index_1,correct1,setbox1):
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

    for i, num in enumerate(areaList):
        if num == areaListSort[len(areaListSort)-2]:
            x, y, w, h = cv2.boundingRect(squares[i]) # Select the 2nd biggest Box
            break
    # print(squares[1])
    # print(x, y, w, h)
    x+=4
    y+=4
    w-=7
    h-=7

    circle_sets=[]
    circle_centers = []
    for i in range(0, 16):
        circle_centers=[]
        for j in range(0, 4):
            circle_centers.append(((2*x + (2*j+1) * w // 4)//2, (2*y + (2*i+1)*h//16) // 2))
        circle_sets.append(circle_centers)
    # print(circle_sets)

    for center,marked_index,is_correct,correct_index in zip(circle_sets,marked_index_2,correct2,setbox2):
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



image = cv2.imread("E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Computer Vision\\Udemy Course\\OpenCV Project-8 (OMR)\Pictures\\fullTest .jpg")
image_copy = image.copy()

squares,areaList, areaListSort,image,marked_set,marked_index_1,marked_index_2 = detect_squares(image_copy)

m,n,ne,set1box1 ,set1box2 ,set2box1 ,set2box2 ,set3box1 ,set3box2 ,set4box1, set4box2= users_directory()

# print(marked_set)
# print(marked_index_1)
# print(set2box1)

if marked_set == 1:
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set1box1,set1box2)
    image_digital = digital_copy(image, correct1,correct2,areaList,areaListSort,squares,marked_index_1,marked_index_2,marks,n,m,set1box1 ,set1box2)
elif marked_set == 2:
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set2box1,set2box2)
    image_digital = digital_copy(image, correct1,correct2,areaList,areaListSort,squares,marked_index_1,marked_index_2,marks,n,m,set2box1 ,set2box2)
elif marked_set == 3:
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set3box1,set3box2)
    image_digital = digital_copy(image, correct1,correct2,areaList,areaListSort,squares,marked_index_1,marked_index_2,marks,n,m,set3box1 ,set3box2)
else: 
    marks,correct1,correct2 = mark(m,n,ne,marked_index_1,marked_index_2,set4box1,set4box2)
    image_digital = digital_copy(image, correct1,correct2,areaList,areaListSort,squares,marked_index_1,marked_index_2,marks,n,m,set4box1 ,set4box2)

# print(correct1)    
print("Marks: " ,marks)

cv2.imshow("main",image_copy)
cv2.imshow("digital",image_digital)
cv2.waitKey(0)
cv2.destroyAllWindows()


