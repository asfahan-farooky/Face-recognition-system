import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime, time, timedelta
import os
import csv
import shutil

# Configuration
ATTENDANCE_DIR = "attendance_records"
IMAGE_DIR = "captured_images"
CLASS_SCHEDULE = {
    "9-10": {"start": time(9, 0), "end": time(10, 0), "type": "lecture"},
    "10-11": {"start": time(10, 0), "end": time(11, 0), "type": "lecture"},
    "11-12": {"start": time(11, 0), "end": time(12, 0), "type": "lecture"},
    "12-13": {"start": time(12, 0), "end": time(13, 0), "type": "lecture"},
    "13-14": {"start": time(13, 0), "end": time(14, 0), "type": "lunch"},
    "14-16": {"start": time(14, 0), "end": time(16, 0), "type": "lab"}
}
LATE_THRESHOLD = 15  # minutes
FACE_MATCH_THRESHOLD = 0.55  # Lower is more strict (default is 0.6)

# Create directories if not exists
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, "teachers"), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, "students"), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, "unknown"), exist_ok=True)

def load_face_encodings():
    try:
        with open('encodings.pkl', 'rb') as f:
            encodeListKnown, classNames = pickle.load(f)
        print(f"[INFO] Loaded {len(classNames)} encodings")
        return encodeListKnown, classNames
    except FileNotFoundError:
        print("[ERROR] encodings.pkl not found. Please create it first.")
        return [], []
    except Exception as e:
        print(f"[ERROR] Loading encodings: {str(e)}")
        return [], []

encodeListKnown, classNames = load_face_encodings()

# Separate teachers and students
teachers = [name for name in classNames if name.startswith("TEACHER_")]
students = [name for name in classNames if not name.startswith("TEACHER_")]

def get_current_session():
    now = datetime.now().time()
    for session_name, times in CLASS_SCHEDULE.items():
        if times["start"] <= now <= times["end"]:
            return session_name, times
    return None, None

def get_attendance_filename(role):
    today = datetime.now().strftime('%Y-%m-%d')
    session_name, _ = get_current_session()
    if session_name:
        return os.path.join(ATTENDANCE_DIR, f'{role}_attendance_{today}_{session_name}.csv')
    return os.path.join(ATTENDANCE_DIR, f'{role}_attendance_{today}.csv')

def save_face_image(img, faceLoc, name):
    y1, x2, y2, x1 = faceLoc
    face_img = img[y1:y2, x1:x2]
    
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    
    if name == "UNKNOWN":
        folder = "unknown"
    elif name in teachers:
        folder = "teachers"
    else:
        folder = "students"
    
    filename = f"{name}_{timestamp}.jpg"
    save_path = os.path.join(IMAGE_DIR, folder, filename)
    cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

def mark_attendance(name, status):
    role = "teacher" if name in teachers else "student"
    filename = get_attendance_filename(role)
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Role', 'Status', 'Time', 'Date', 'Session', 'Session Type'])
        
        now = datetime.now()
        session_name, session_info = get_current_session()
        
        writer.writerow([
            name,
            role,
            status,
            now.strftime('%H:%M:%S'),
            now.strftime('%Y-%m-%d'),
            session_name if session_name else "N/A",
            session_info["type"] if session_info else "N/A"
        ])

def check_existing_attendance(name, current_session):
    role = "teacher" if name in teachers else "student"
    filename = get_attendance_filename(role)
    if not os.path.exists(filename):
        return False
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row['Name'] == name and row['Session'] == current_session:
                return True
    return False

def initialize_daily_attendance():
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    
    for session_name in CLASS_SCHEDULE:
        if CLASS_SCHEDULE[session_name]["type"] != "lunch":
            filename = os.path.join(ATTENDANCE_DIR, f'student_attendance_{today}_{session_name}.csv')
            if not os.path.exists(filename):
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Name', 'Role', 'Status', 'Time', 'Date', 'Session', 'Session Type'])
                    for student in students:
                        writer.writerow([student, 'student', 'Absent', '', today, session_name, CLASS_SCHEDULE[session_name]["type"]])

def draw_info_panel(img, present_count, absent_count, late_count):
    h, w = img.shape[:2]
    
    overlay = img.copy()
    cv2.rectangle(overlay, (w-350, 0), (w, 150), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    current_time = datetime.now().strftime('%H:%M:%S')
    session_name, session_info = get_current_session()
    
    cv2.putText(img, f"Time: {current_time}", (w-340, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    if session_name:
        cv2.putText(img, f"Session: {session_name}", (w-340, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.putText(img, f"Type: {session_info['type']}", (w-340, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    
    cv2.putText(img, f"Present: {present_count}", (w-340, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(img, f"Late: {late_count}", (w-340, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

def recognize_faces(img):
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    recognized_names = []
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=FACE_MATCH_THRESHOLD)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDis)
        minDistance = faceDis[matchIndex]
        
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        
        if minDistance <= FACE_MATCH_THRESHOLD and matches[matchIndex]:
            name = classNames[matchIndex].upper()
            confidence = f"{(1 - minDistance) * 100:.2f}%"
            color = (0, 255, 0)  # Green for known faces
        else:
            name = "UNKNOWN"
            confidence = "N/A"
            color = (0, 0, 255)  # Red for unknown faces
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if name != "UNKNOWN":
            cv2.putText(img, confidence, (x1 + 6, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        recognized_names.append((name, (y1, x2, y2, x1)))
    
    return recognized_names

# Initialize attendance for the day
initialize_daily_attendance()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Recognize faces
    recognized_names = recognize_faces(img)
    
    # Count attendance stats
    session_name, _ = get_current_session()
    present_count = late_count = absent_count = 0
    
    if session_name:
        role = "student"
        filename = get_attendance_filename(role)
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Session'] == session_name:
                        if row['Status'] == 'Present':
                            present_count += 1
                        elif row['Status'] == 'Late':
                            late_count += 1
                        elif row['Status'] == 'Absent':
                            absent_count += 1
    
    # Process recognized faces
    for name, faceLoc in recognized_names:
        save_face_image(img, faceLoc, name)
        
        if name != "UNKNOWN" and session_name:
            if not check_existing_attendance(name, session_name):
                session_info = CLASS_SCHEDULE[session_name]
                now = datetime.now()
                current_time = now.time()
                
                if session_info["type"] == "lecture":
                    late_time = time(session_info["start"].hour, 
                                   session_info["start"].minute + LATE_THRESHOLD)
                    status = 'Late' if current_time > late_time else 'Present'
                else:
                    status = 'Present'
                
                mark_attendance(name, status)

    # Display info panel
    draw_info_panel(img, present_count, absent_count, late_count)
    
    # Display help text
    cv2.putText(img, "Press 'Q' to quit", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Smart Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()