
import cv2
import numpy as np
from scipy.io.wavfile import read, write

img = cv2.imread("Ave Maria.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
print(img.shape)

# detecting staff (horizontal lines)
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

cv2.imshow('line', img)
cv2.imwrite("lines.jpg",img)
cv2.waitKey(0)

# detecting notes:
quarter_template = cv2.imread("quarter.png")
half_template = cv2.imread("half.png")
whole_template = cv2.imread("whole.png")

quarter_gray = cv2.cvtColor(quarter_template, cv2.COLOR_BGR2GRAY)
print(quarter_gray.shape)
threshold_quarter = 0.7
half_gray = cv2.cvtColor(half_template, cv2.COLOR_BGR2GRAY)
print(quarter_gray.shape)
threshold_half = 0.75
whole_gray = cv2.cvtColor(whole_template, cv2.COLOR_BGR2GRAY)
print(whole_gray.shape)
threshold_whole = 0.7
w_q, h_q = quarter_gray.shape[::-1]
w_h, h_h = half_gray.shape[::-1]
w_w, h_w = whole_gray.shape[::-1]

def find_notes_in_image(img, template, threshold):
    # template matching:
    matched = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(matched > threshold)

    # highlight matched notes:
    rectangles = list(zip(*locations[::-1]))  # to combine two or more iterables into a single iterable
    rectangles = sorted(rectangles, key=lambda x: (x[0], x[1]))  # sorted(iterable, key=key, reverse=reverse)
    rectangles = np.asarray(rectangles)  # convert to array
    print(len(rectangles))
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
    # for pt in rectangles:
    #     cv2.rectangle(img, (pt[0] - w, pt[1] - h), (pt[0] + 2 * w, pt[1] + 2 * h), (0, 0, 255), 1)
    return rectangles

rectangles_Q=find_notes_in_image(img1,quarter_template,threshold_quarter)
rectangles_H=find_notes_in_image(img1,half_template,threshold_half)
rectangles_W=find_notes_in_image(img1,whole_template,threshold_whole)
for pt in rectangles_W:
    cv2.rectangle(img, (pt[0] , pt[1] ), (pt[0] + w_w, pt[1] + h_w), (0, 0, 255), 1)
for pt in rectangles_H:
    cv2.rectangle(img, (pt[0] , pt[1] ), (pt[0] + w_h, pt[1] + h_h), (0, 0, 255), 1)
for pt in rectangles_Q:
    cv2.rectangle(img, (pt[0] - w_q, pt[1] - h_q), (pt[0] + 2 * w_q, pt[1] + 2 * h_q), (0, 0, 255), 1)

cv2.imshow('line', img)
rectangles = np.ndarray.tolist(rectangles_Q)
rectangles.extend(np.ndarray.tolist(rectangles_H))
rectangles.extend(np.ndarray.tolist(rectangles_W))

# In this part, we just have a single bar:
# bar = []

HADL = 8   # half of average distance of lines in these bars

# notes_center = rectangles + [0 ,h-HADL]
#
# # now separate lines of each bar:
# bar_lines=main_lines[0:5]

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
    for note in music_notes:
        if (note == 6.5):
            final_notes.append([2, 3, 4])   # B3  number, octave, duration
        elif (note == 6):
            final_notes.append([3, 4, 4])   # C4
        elif (note == 5.5):
            final_notes.append([5, 4, 4])   # D4
        elif (note == 5):
            final_notes.append([7, 4, 4])   # E4
        elif (note == 4.5):
            final_notes.append([8, 4, 4])   # F4
        elif (note == 4):
            final_notes.append([10, 4, 4])  # G4
        elif (note == 3.5):
            final_notes.append([0, 4, 4])   # A4
        elif (note == 3):
            final_notes.append([2, 4, 4])   # B4
        elif (note == 2.5):
            final_notes.append([3, 5, 4])   # C5
        elif (note == 2):
            final_notes.append([5, 5, 4])   # D5
        elif (note == 1.5):
            final_notes.append([7, 5, 4])   # E5
        elif (note == 1):
            final_notes.append([8, 5, 4])   # F5
        elif (note == 0.5):
            final_notes.append([10, 5, 4])  # G5
        elif (note == 0):
            final_notes.append([0, 5, 4])   # A5
        elif (note == -0.5):
            final_notes.append([2, 5, 4])   # B5


# cv2.imshow('note detected', img)
cv2.imwrite("notes.jpg",img)
cv2.waitKey(0)

#playing detected notes using code in helper file:
fs1 = 44100
tempo=120
notes_base = 2**(np.arange(12)/12)*27.5
notes_duration = np.array([3200, 1600, 800, 400, 200, 100])*0.7/tempo*225
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
len(final_notes)
write("Ave Maria.wav", fs1 , notes_play.astype(np.float32))