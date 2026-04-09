# 🎓 Smart Face Recognition Attendance System

## 📌 Overview

This project is a **Smart Attendance System** built using **Face Recognition Technology**.
It automatically detects and recognizes faces in real-time and marks attendance based on class schedules.

The system also tracks:

* Present / Absent / Late status
* Session timing (lecture/lab)
* Saves captured face images
* Generates CSV attendance reports

---

## 🚀 Features

* 🎯 Real-time face detection and recognition
* 🧑‍🏫 Separate handling for Teachers and Students
* ⏰ Automatic attendance based on class schedule
* ⚠️ Late detection (based on time threshold)
* 📊 CSV-based attendance records
* 📸 Captures and stores face images
* ❌ Unknown face detection and storage
* 🖥️ Live dashboard with attendance stats

---

## 🛠️ Technologies Used

* Python 🐍
* OpenCV
* face_recognition library
* NumPy
* CSV handling
* Pickle (for encoding storage)

---

## 📂 Project Structure

```
Face-recognition-system/
│── attendance_records/
│── captured_images/
│   ├── teachers/
│   ├── students/
│   └── unknown/
│── encodings.pkl
│── main.py
│── requirements.txt
```

---

## ⚙️ How It Works

1. Loads pre-trained face encodings from `encodings.pkl`
2. Captures live video from webcam
3. Detects and recognizes faces
4. Matches faces with stored data
5. Marks attendance based on:

   * Current time
   * Class schedule
   * Late threshold
6. Saves attendance in CSV files
7. Stores face images in categorized folders

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```
git clone https://github.com/asfahan-farooky/Face-recognition-system.git
```

### 2️⃣ Go to Project Folder

```
cd Face-recognition-system
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Project

```
python main.py
```

---

## ⏰ Class Schedule Logic

* Lecture sessions: Attendance + Late detection
* Lab sessions: Only attendance
* Lunch: Ignored

Late threshold: **15 minutes**

---

## 📊 Output

* Attendance stored in:

  * `attendance_records/*.csv`
* Images stored in:

  * `captured_images/students/`
  * `captured_images/teachers/`
  * `captured_images/unknown/`

---

## ⚠️ Requirements

* Webcam
* Pre-trained `encodings.pkl` file
* Python 3.x

---

## 🤝 Contributing

Feel free to fork this repository and improve it.

---

## 📧 Author

**Asfahan Farooky**
