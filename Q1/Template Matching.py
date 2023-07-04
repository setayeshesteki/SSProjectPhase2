import cv2
background=cv2.imread("background.jpg")
template=cv2.imread("template.jpg")
background_gray=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
matched_image=cv2.matchTemplate(background_gray,template_gray,cv2.TM_CCOEFF_NORMED)
(min_val, max_val , min_loc , max_loc) = cv2.minMaxLoc(matched_image)
loc_x=max_loc[0]
loc_y=max_loc[1]
height=template_gray.shape[0]
width = template_gray.shape[1]
cv2.rectangle(background,(loc_x,loc_y), (loc_x+width , loc_y+height) , (255 , 0 , 150) , 2)
cv2.imwrite("matchedImagepy.jpg",background)
cv2.imshow("matched_image", background)
cv2.waitKey(0)