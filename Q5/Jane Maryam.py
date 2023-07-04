
import cv2
import numpy as np
from scipy.io.wavfile import read, write
#
# # converting pdf file to image(jpg and grey scale)
# input_file = r"Twinkle, Twinkle, Little Star.pdf"
# file_handle = fitz.open(input_file)
# page = file_handle[0]   # the index shows the page number
# page_img = page.get_pixmap()
# page_img.save('Twinkle.jpg')
img = cv2.imread("Jane Maryam.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
print(img.shape)

# detecting staff (horizontal lines)
def detecting_staffs(img_gray):
    edge = cv2.Canny(img_gray, 100, 200, 3)   # develop edged_image using cv2.Canny (image, T_lower, T_upper, aperture_size, L2Gradient)
# function so that we can detect staffs using HoughLinesP
# default value for L2Gradient is 3, and you can increase it if you want more details up to 7
    cv2.imshow('edge', edge)
    cv2.imwrite("edged.jpg",edge)
    cv2.waitKey(0)
    linesP = cv2.HoughLinesP(edge, 0.1, np.pi / 180, 100, 100, 5)    # cv2.HoughLinesP(edged_image, rho, theta, threshold, minLineLength, maxLineGap)
# output is x1,y1 , x2,y2
    main_lines = []
    for points in linesP:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        if (abs(x1 - x2) > 20) and (abs(y1 - y2) < 5):  # check the first condition
            flag = 1
        # these lines are not the final lines we want and has some unwanted lines too
        # to detect wanted lines, 1.first we should find lines which has delta_y<(constant=5)
        # and 2.then check if this line has been detected before or not
            for temp_line in main_lines:
                if abs(temp_line - y1) < 4:
                    flag = 0
            if flag or main_lines == []:
                main_lines.append(y1)
                cv2.line(img, (0, y1), (1742, y1), (0, 255, 0), 2, cv2.LINE_AA)

    # main lines which we want is selected now sort them based on the y_coordinate in the sheet
    main_lines.sort()
    return main_lines

main_lines = detecting_staffs(img_gray)
cv2.imshow('line', img)
cv2.imwrite("lines.jpg",img)
cv2.waitKey(0)

# detecting notes:
quarter_up_template = cv2.imread("quarter_up.png")
half_up_template = cv2.imread("half_up.png")
quarter_down_template = cv2.imread("quarter_down.png")
half_down_template = cv2.imread("half_down.png")
dot_template= cv2.imread("dot.png")

quarter1_gray = cv2.cvtColor(quarter_up_template, cv2.COLOR_BGR2GRAY)
print(quarter1_gray.shape)
threshold_quarter1 = 0.5
half1_gray = cv2.cvtColor(half_up_template, cv2.COLOR_BGR2GRAY)
print(half1_gray.shape)
threshold_half1 = 0.52
quarter2_gray = cv2.cvtColor(quarter_down_template, cv2.COLOR_BGR2GRAY)
print(quarter2_gray.shape)
threshold_quarter2 = 0.8
half2_gray = cv2.cvtColor(half_down_template, cv2.COLOR_BGR2GRAY)
print(half2_gray.shape)
threshold_half2 = 0.53
dot_gray = cv2.cvtColor(dot_template, cv2.COLOR_BGR2GRAY)
threshold_dot = 0.55
w_q1, h_q1 = quarter1_gray.shape[::-1]
w_h1, h_h1 = half1_gray.shape[::-1]
w_q2, h_q2 = quarter2_gray.shape[::-1]
w_h2, h_h2 = half2_gray.shape[::-1]
w_d, h_d = dot_gray.shape[::-1]

def find_notes_in_image(img, template, threshold,dur):
    img2 = cv2.imread("Jane Maryam.png")
    matched2 = cv2.matchTemplate(img, dot_template, cv2.TM_CCOEFF_NORMED)
    locations2 = np.where(matched2 > threshold_dot)
                # template matching:
    img = cv2.imread("Jane Maryam.png")
    matched = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(matched > threshold)

    # highlight matched notes:
    rectangles = list(zip(*locations[::-1], locations[0].shape[0] * [dur]))  # to combine two or more iterables into a single iterable
    rectangles = sorted(rectangles, key=lambda x: (x[0], x[1]))  # sorted(iterable, key=key, reverse=reverse)
    rectangles = np.asarray(rectangles)  # convert to array
    doted=list(zip(locations[0].shape[0] * [dur]))
    keep = [0]
    x_margin = 20
    y_margin = 20
    prev_x = rectangles[0][0]
    prev_y = rectangles[0][1]

    for i in range(1, rectangles.shape[0]):
        if abs(rectangles[i][0] - prev_x) > x_margin or abs(rectangles[i][1] - prev_y) > y_margin:
            keep.append(i)
            prev_x = rectangles[i][0]
            prev_y = rectangles[i][1]

    rectangles = rectangles[keep]
    for point in locations2:
        temp_dot_X , temp_dot_Y = point[0] + 13, point[1] + 10
        for i in rectangles:
            if (abs(i[1] - temp_dot_Y) < 75):
                i[2]=i[2]*1.5
    # for pt in rectangles:
    #     cv2.rectangle(img, (pt[0] - w, pt[1] - h), (pt[0] + 2 * w, pt[1] + 2 * h), (0, 0, 255), 1)
    return rectangles

rectangles_Q1=find_notes_in_image(img1,quarter_up_template,threshold_quarter1,2)
rectangles_H1=find_notes_in_image(img1,half_up_template,threshold_half1,1)
rectangles_Q2=find_notes_in_image(img1,quarter_up_template,threshold_quarter2,2)
rectangles_H2=find_notes_in_image(img1,half_up_template,threshold_half2,1)
rectangles_D=find_notes_in_image(img1,dot_template,threshold_dot,0)

def all_types_of_note(rectangles_Q1,rectangles_Q2,rectangles_H1,rectangles_H2,rectangles_D):
    for pt in rectangles_D:
        cv2.rectangle(img, (pt[0] , pt[1] ), (pt[0] + w_d, pt[1] + h_d), (0, 0, 255), 1)
    for pt in rectangles_H1:
        cv2.rectangle(img, (pt[0] , pt[1] ), (pt[0] + w_h1, pt[1] + h_h1), (0, 0, 255), 1)
    for pt in rectangles_Q1:
        cv2.rectangle(img, (pt[0] - w_q1, pt[1] - h_q1), (pt[0] + 2 * w_q1, pt[1] + 2 * h_q1), (0, 0, 255), 1)
    for pt in rectangles_H2:
        cv2.rectangle(img, (pt[0] , pt[1] ), (pt[0] + w_h2, pt[1] + h_h2), (0, 0, 255), 1)
    for pt in rectangles_Q2:
        cv2.rectangle(img, (pt[0] - w_q2, pt[1] - h_q2), (pt[0] + 2 * w_q2, pt[1] + 2 * h_q2), (0, 0, 255), 1)
    rectangles = np.ndarray.tolist(rectangles_Q1)
    rectangles.extend(np.ndarray.tolist(rectangles_H1))
    rectangles.extend(np.ndarray.tolist(rectangles_D))
    rectangles.extend(np.ndarray.tolist(rectangles_Q2))
    rectangles.extend(np.ndarray.tolist(rectangles_H2))
    rectangles=sorted(rectangles,key=lambda x: (x[0] ,x[1]))
    rectangles = np.asarray(rectangles)
    cv2.imshow('notes', img)
    cv2.imwrite("notes.jpg", img)
    cv2.waitKey(0)
    return rectangles
rectangles=all_types_of_note(rectangles_Q1,rectangles_Q2,rectangles_H1,rectangles_H2,rectangles_D)

def detecting_note(rectangles):
    # In this part, we just have a single bar:
    HADL = 8   # half of average distance of lines in these bars

    # in the following we have the accepted slip for notes which are on a line, between 2 lines or under the lower line:
    on_slip=HADL/2
    between_slip=HADL
    lower_slip=2*HADL+on_slip

    music_notes=[]  # compiling the whole song

    for note in rectangles:
        i=0
        difference_y=[0,0,0,0,0]
        for item in main_lines:
            difference_y[i] = abs(item-note[1])
            i=i+1
        sorted_difference=np.argsort(difference_y)  # now we have index of the nearest line to the note in sorted_difference[0]
        if (sorted_difference[0]==0):
            if(difference_y[sorted_difference[0]] <= on_slip):
                music_notes.append(sorted_difference[0] + 1)
            elif(difference_y[sorted_difference[0]] <= between_slip):
                music_notes.append(0.5)
            elif(difference_y[sorted_difference[0]] <= lower_slip):
                music_notes.append(0)
            else:
                music_notes.append(-0.5)
        elif (difference_y[sorted_difference[0]] <= on_slip):     # the note is on a line
            music_notes.append(sorted_difference[0] + 1)
        elif (difference_y[sorted_difference[0]] <= between_slip):   # the note is between 2 lines
            music_notes.append((sorted_difference[0] + sorted_difference[1]) / 2 + 1)
        elif (difference_y[sorted_difference[0]] <= lower_slip ):        # the note is under the last line
            if(sorted_difference[0]==4):
                music_notes.append(6)
            else:
                music_notes.append(6.5)

    # now we have the position of all notes of all bars, so it's time to distinguish each note and build the final_notes as follows:
        final_notes=[]
        for i,note in enumerate(music_notes):
            if (note == 6.5):
                final_notes.append([2, 3, rectangles[i][2]])   # B3  number, octave, duration
            elif (note == 6):
                final_notes.append([3, 4, rectangles[i][2]])   # C4
            elif (note == 5.5):
                final_notes.append([5, 4, rectangles[i][2]])   # D4
            elif (note == 5):
                final_notes.append([7, 4, rectangles[i][2]])   # E4
            elif (note == 4.5):
                final_notes.append([8, 4, rectangles[i][2]])   # F4
            elif (note == 4):
                final_notes.append([10, 4, rectangles[i][2]])  # G4
            elif (note == 3.5):
                final_notes.append([0, 4, rectangles[i][2]])   # A4
            elif (note == 3):
                final_notes.append([2, 4, rectangles[i][2]])   # B4
            elif (note == 2.5):
                final_notes.append([3, 5, rectangles[i][2]])   # C5
            elif (note == 2):
                final_notes.append([5, 5, rectangles[i][2]])   # D5
            elif (note == 1.5):
                final_notes.append([7, 5, rectangles[i][2]])   # E5
            elif (note == 1):
                final_notes.append([8, 5, rectangles[i][2]])   # F5
            elif (note == 0.5):
                final_notes.append([10, 5, rectangles[i][2]])  # G5
            elif (note == 0):
                final_notes.append([0, 5, rectangles[i][2]])   # A5
            elif (note == -0.5):
                final_notes.append([2, 5, rectangles[i][2]])   # B5
    return final_notes
final_notes=detecting_note(rectangles)




#playing detected notes using code in helper file:
fs1 = 44100
tempo=120
notes_base = 2**(np.arange(12)/12)*27.5
notes_duration = np.array([3200, 1600, 800, 400, 200, 100])*0.7/tempo*225/3
notes_ann = ['A', 'A#', 'B', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#']


def sin_wave(f, n, fs):
    x = np.linspace(0, 2*np.pi, n)
    ring = 30
    xp = np.linspace(0, -1*(n*ring/fs), n)
    y = np.sin(x*f*(n/fs))*np.exp(xp)
    z = np.zeros([n, 2])
    z[:, 0] = y
    z[:, 1] = y
    return z

def play_note(note_id, octave, dur, fs):
    if (note_id < 3) :
        octave += 1
    y = sin_wave(notes_base[note_id]*2**octave, int(notes_duration[dur]*(fs/1000)), fs)
    sd.play(y, fs)
    sd.wait()
    return

def put_note(note_id, octave, dur, fs):
    if (note_id < 3) :
        octave += 1
    y = sin_wave(notes_base[note_id]*2**octave, int(notes_duration[dur]*(fs/1000)), fs)
    return y

def get_music(music_notes, fs):
    m = []
    for item in music_notes:
        y = put_note(item[0], item[1], item[2], fs)
        m.append(y)
    m = np.concatenate(m, 0)
    return m


notes_play=get_music(final_notes,fs1)
#sd.play(notes_play)
write("Ave Maria.wav", fs1 , notes_play.astype(np.float32))