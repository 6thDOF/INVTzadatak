import cv2
import numpy as np
from ultralytics import YOLO


def centroid(bbox):
    '''
    this function returns the center of the boudning box for detected vehicles
    bbox contains 2 coordinates in the following order x1,y1,x2,y2
    '''
    return ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)


def distance(pt1, pt2):
    '''
    this function returns the distance between 2 points
    points are formated as (x,y)
    '''
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def VehicleTracker(path, threshold, fsave=True, fshow=True):
    '''
    path --> path to the video
    threshold --> max distance in pixels to match detections and IDs in frames
    fsave --> flag to save the file or not
    fshow --> flag to visually represent the results in real time or not
    '''
    video = cv2.VideoCapture(path)  # loading
    fps = video.get(cv2.CAP_PROP_FPS)  # get the fps of the video
    width, height = (1280, 720)  # the resulting video width and height

    # save results if the fsave flag is True
    if fsave:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('VideoDetections.mp4', fourcc,
                              fps, (width, height))

    # loading the pre-trained YOLO 11 model
    model = YOLO('weights/yolo11n.pt')

    # model detects a wast number of classes,
    # thus we have to define the ones we need to filter
    # classes represent car, bus and a truck respectively
    VehicleClassIDs = [2, 5, 7]

    # variables for tracking
    # this is a dictionary:
    # {trackID: list (FrameNumber, Centroid, bbox)}
    tracks = {}
    NextTrackID = 0
    # list of (trackID, Centroid) for detected vehicles from the previous frame
    PrevDetections = []
    threshold = 100  # maximum distance in pixels to match detections

    FrameNumber = 0

    ''' PROCESSING THE VIDEO '''
    while True:
        ret, frame = video.read()
        if ret:
            # resize frame because cv2 has a problem displaying a large image
            frame = cv2.resize(frame, (1280, 720))
            FrameNumber += 1

            # get YOLO prediction
            results = model(frame)

            # extracting the results in form of numpy
            # detection has [x1,y1,x2,y2,confidence,class]
            detections = results[0].boxes.data.cpu().numpy()

            # store what we need from all detections and filtering the classes
            CurrentDetection = []
            for det in detections:
                if int(det[-1]) in VehicleClassIDs:
                    bbox = [det[0], det[1], det[2], det[3]]
                    cnt = centroid(bbox)
                    CurrentDetection.append((cnt, bbox))

            # match the ids based on centroids from previous and current frame
            UpdDetections = []
            for cnt, bbox in CurrentDetection:
                AssignedID = None
                minDist = float('inf')
                for (prevID, prevCnt) in PrevDetections:
                    dist = distance(cnt, prevCnt)
                    if dist < threshold and dist < minDist:
                        minDist = dist
                        AssignedID = prevID

                # handle no matches and assigning new ID
                if AssignedID is None:
                    AssignedID = NextTrackID
                    NextTrackID += 1

                UpdDetections.append((cnt, bbox, AssignedID))

                # update tracks parameter with current detection
                if AssignedID not in tracks:
                    tracks[AssignedID] = []
                tracks[AssignedID].append((FrameNumber, cnt, bbox))

            # saving the current frame as the previous one for matching
            PrevDetections = [(id, cnt) for (cnt, b, id) in UpdDetections]

            # drawing of the bounding boxes and IDs on the frame
            for cnt, bbox, id in UpdDetections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                text = f"ID: {id}"
                track = tracks[id]
                if len(track) >= 2:
                    (prevF, prevCnt, _), (currF, currCnt, _) = track[-2:]
                    dt = (currF - prevF)/fps
                    if dt > 0:
                        speed = distance(prevCnt, currCnt)/dt
                        text += f", {speed:.1f} px/sec"
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # save results if the fsave flag is True
            if fsave:
                out.write(frame)

            # display results if the fshow flag is True
            if fshow:
                cv2.imshow("Vehicle Tracking", frame)
                # pressing q will stop processing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    if fsave:
        out.release()
    video.release()
    cv2.destroyAllWindows()

    ''' SPEED CALCULATION AND SAVING THE IMAGE OF THE FASTEST VEHICLE'''
    VehicleSpeeds = {}  # dictionary: {trackID: AverageSpeed}
    for id, track in tracks.items():
        # filtering bad detections where vehicle was detected in only one frame
        if len(track) < 2:
            continue
        TotalSpeed = 0
        count = 0
        for i in range(1, len(track)):
            prevFrame, prevCnt, bbox = track[i-1]
            currFrame, currCnt, bbox = track[i]
            dist = distance(prevCnt, currCnt)
            dt = (currFrame - prevFrame)/fps
            if dt > 0:
                TotalSpeed += dist/dt
                count += 1
        if count > 0:
            VehicleSpeeds[id] = TotalSpeed/count

    FastID = None
    MaxSp = 0
    for id, speed in VehicleSpeeds.items():
        if speed > MaxSp:
            MaxSp = speed
            FastID = id
    print(f"\nFastest vehicle had ID: {FastID} and speed: {MaxSp} px/sec.")

    if FastID is not None:
        LastDetection = tracks[FastID][-1]
        LastFrameNumber, cnt, bbox = LastDetection

        video = cv2.VideoCapture(path)
        video.set(cv2.CAP_PROP_POS_FRAMES, LastFrameNumber-1)
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))
            x1, y1, x2, y2 = map(int, bbox)
            cropped = frame[y1:y2, x1:x2]
            cv2.imwrite("FastestVehicle.png", cropped)
        video.release()


print("The task of this program is to detect the vehicles on the video,",
      "calculate their speeds and save an image of the fastest vehicle.")
print("By default, the program will display the results in real time",
      "and then save the resulting video in the current directory.")
print("Pressing 'q' after clicking the real time desplay will stop the video.")
path = input("Enter the path to the video file:\n")
flag1 = input("Display the results in real time? (type y or n)\n")
if flag1 == "y":
    fshow = True
elif flag1 == "n":
    fshow = False
else:
    fshow = True
flag2 = input("Would you like to save the resulting video? (type y or n)\n")
if flag2 == "y":
    fsave = True
elif flag2 == "n":
    fsave = False
else:
    fsave = True

VehicleTracker(path=path, threshold=50, fsave=fsave, fshow=fshow)
