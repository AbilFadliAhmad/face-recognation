import os
import cv2
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from insightface.app import FaceAnalysis

app = Flask(__name__)

# --- INISIALISASI INSIGHTFACE ---
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1, det_size=(640, 640)) # ctx_id=-1 untuk CPU

DB_FILE = "database.db"

# --- INISIALISASI DATABASE SQLITE ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # embedding disimpan sebagai BLOB (Bytes)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- FUNGSI UCAPAN WAKTU ---
def get_greeting(name):
    hour = datetime.now().hour
    if 4 <= hour < 10:
        ucapan = "Selamat Pagi"
    elif 10 <= hour < 15:
        ucapan = "Selamat Siang"
    elif 15 <= hour < 18:
        ucapan = "Selamat Sore"
    else:
        ucapan = "Selamat Malam"
    return f"{ucapan}, {name}!"

# --- ROUTE UTAMA ---
@app.route("/")
def index():
    return """
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>Aplikasi Pengenalan Wajah Fullscreen</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                /* Animasi Slide Up untuk Menu Bawah (Mobile Optimized) */
                .slide-up {
                    transform: translateY(100%);
                    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
                    will-change: transform;
                }
                .slide-up.active {
                    transform: translateY(0);
                }
                
                /* Overlay Grafik Scan Sci-Fi */
                .scan-line {
                    position: absolute;
                    width: 100%;
                    height: 3px;
                    background: rgba(56, 189, 248, 0.6);
                    box-shadow: 0 0 12px rgba(56, 189, 248, 0.9);
                    animation: scan 2.8s linear infinite;
                }
                @keyframes scan {
                    0% { top: -5%; }
                    100% { top: 105%; }
                }
            </style>
            <!-- Face-API.js -->
            <script defer src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js"></script>
        </head>
        <body class="bg-black text-white m-0 p-0 overflow-hidden h-screen w-screen relative font-sans select-none">

            <!-- Container Fullscreen Webcam -->
            <div class="absolute inset-0 w-full h-full flex items-center justify-center bg-black">
                <video id="webcam" autoplay playsinline muted class="w-full h-full object-cover transform -scale-x-100"></video>
                
                <!-- Grafik Scan Line -->
                <div id="scanOverlay" class="absolute inset-0 pointer-events-none">
                    <div class="scan-line"></div>
                </div>
            </div>

            <!-- Canvas Tersembunyi -->
            <canvas id="canvas" class="hidden"></canvas>

            <!-- UI Overlay Atas (Status Badge) -->
            <div class="fixed top-5 left-0 w-full flex justify-center pointer-events-none z-30 px-4">
                <div id="statusBadge" class="bg-slate-900/80 backdrop-blur-md px-5 py-2 rounded-full border border-slate-700/60 flex items-center gap-3 shadow-2xl transition-all duration-300">
                    <span id="statusDot" class="w-2.5 h-2.5 rounded-full bg-yellow-500 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.8)]"></span>
                    <span id="statusText" class="text-xs sm:text-sm font-medium tracking-wide">Memuat Model Deteksi...</span>
                </div>
            </div>

            <!-- UI Overlay Pesan (Loading / Alert / Ucapan) -->
            <div id="messageOverlay" class="fixed top-20 left-0 w-full flex justify-center pointer-events-none z-30 hidden transition-all duration-300 px-4">
                <div id="messageBox" class="bg-blue-600/90 backdrop-blur-md text-white px-6 py-3 rounded-2xl shadow-2xl border border-blue-400/50 font-semibold text-sm sm:text-base tracking-wide flex items-center gap-3">
                    <span id="loadingSpinner" class="hidden w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                    <span id="messageText">Wajah terdeteksi, mohon tunggu...</span>
                </div>
            </div>

            <!-- BOTTOM SHEET: Menu Aksi (Fixed untuk Mobile) -->
            <div id="bottomMenu" class="fixed bottom-0 left-0 w-full bg-slate-900/95 backdrop-blur-xl border-t border-slate-700/60 p-6 rounded-t-3xl shadow-[0_-10px_40px_rgba(0,0,0,0.7)] z-50 slide-up max-h-[85vh] overflow-y-auto">
                <div class="w-full max-w-md mx-auto flex flex-col items-center">
                    
                    <!-- Indikator Drag -->
                    <div class="w-12 h-1.5 bg-slate-600 rounded-full mb-5"></div>

                    <!-- Form Registrasi Wajah Baru -->
                    <div id="registerForm" class="hidden w-full">
                        <h3 class="text-xl font-bold text-center text-blue-400 mb-1">Wajah Tidak Dikenali</h3>
                        <p class="text-xs sm:text-sm text-slate-400 text-center mb-5">Silakan masukkan nama Anda untuk mendaftar ke sistem.</p>
                        
                        <input type="text" id="newName" placeholder="Nama Lengkap Anda..." autocomplete="off" class="w-full bg-slate-800 border border-slate-600 text-white px-4 py-3.5 rounded-xl mb-4 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 text-center text-base sm:text-lg placeholder-slate-500">
                        
                        <div class="flex gap-3">
                            <button type="button" onclick="resumeScanning()" class="w-1/3 bg-slate-700 hover:bg-slate-600 active:scale-95 text-white font-medium py-3 rounded-xl transition">Batal</button>
                            <button type="button" onclick="registerFace()" class="w-2/3 bg-blue-600 hover:bg-blue-500 active:scale-95 text-white font-bold py-3 rounded-xl transition shadow-lg shadow-blue-600/30">Simpan Wajah</button>
                        </div>
                    </div>

                    <!-- Form Edit Nama -->
                    <div id="editForm" class="hidden w-full">
                        <h3 class="text-base sm:text-lg font-semibold text-center text-slate-200 mb-3">Ingin mengubah nama Anda?</h3>
                        <input type="hidden" id="currentUserId">
                        
                        <input type="text" id="updateName" autocomplete="off" class="w-full bg-slate-800 border border-slate-600 text-white px-4 py-3 rounded-xl mb-4 focus:outline-none focus:border-yellow-500 text-center text-base sm:text-lg">
                        
                        <div class="flex gap-3">
                            <button type="button" onclick="resumeScanning()" class="w-1/3 bg-slate-700 hover:bg-slate-600 active:scale-95 text-white font-medium py-3 rounded-xl transition">Tutup</button>
                            <button type="button" onclick="editName()" class="w-2/3 bg-yellow-600 hover:bg-yellow-500 active:scale-95 text-white font-bold py-3 rounded-xl transition shadow-lg shadow-yellow-600/30">Ubah Nama</button>
                        </div>
                    </div>

                </div>
            </div>

            <script>
                const video = document.getElementById('webcam');
                const canvas = document.getElementById('canvas');
                
                // UI Elements
                const statusDot = document.getElementById('statusDot');
                const statusText = document.getElementById('statusText');
                const scanOverlay = document.getElementById('scanOverlay');
                const messageOverlay = document.getElementById('messageOverlay');
                const messageBox = document.getElementById('messageBox');
                const messageText = document.getElementById('messageText');
                const loadingSpinner = document.getElementById('loadingSpinner');
                
                // Bottom Menu Elements
                const bottomMenu = document.getElementById('bottomMenu');
                const registerForm = document.getElementById('registerForm');
                const editForm = document.getElementById('editForm');
                const currentUserId = document.getElementById('currentUserId');
                const newNameInput = document.getElementById('newName');
                const updateNameInput = document.getElementById('updateName');

                let currentFrameBlob = null;
                let isProcessing = false;
                let isPaused = false;

                // 1. Inisialisasi Face-API & Webcam
                async function initFaceAPI() {
                    try {
                        statusText.textContent = "Memuat Model Deteksi...";
                        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
                        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                        
                        statusText.textContent = "Membuka Kamera...";
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } 
                        });
                        video.srcObject = stream;
                        
                        video.onloadedmetadata = () => {
                            video.play();
                            statusDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]";
                            statusText.textContent = "Mencari Wajah...";
                            
                            // Mulai Loop Deteksi Wajah Lokal Setiap 500ms
                            startFaceDetectionLoop();
                        };
                    } catch (err) {
                        console.error(err);
                        statusDot.className = "w-2.5 h-2.5 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]";
                        statusText.textContent = "Gagal Mengakses Kamera";
                    }
                }

                // 2. Loop Deteksi Wajah Lokal di Browser
                function startFaceDetectionLoop() {
                    setInterval(async () => {
                        if (isPaused || isProcessing || video.paused || video.ended) return;
                        
                        try {
                            const detections = await faceapi.detectAllFaces(
                                video, 
                                new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.45 })
                            );

                            if (detections.length > 0) {
                                isProcessing = true;
                                
                                // Tampilkan Loading
                                showLoadingState("Wajah terdeteksi, mohon tunggu...");
                                
                                await processRecognition();
                                
                                isProcessing = false;
                            }
                        } catch (e) {
                            console.error("Detection error:", e);
                            isProcessing = false;
                        }
                    }, 500);
                }

                // 3. Tangkap Frame Gambar dari Kamera
                function captureFrame() {
                    return new Promise((resolve) => {
                        canvas.width = video.videoWidth || 640;
                        canvas.height = video.videoHeight || 480;
                        const ctx = canvas.getContext('2d');
                        
                        ctx.translate(canvas.width, 0);
                        ctx.scale(-1, 1);
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        
                        canvas.toBlob((blob) => {
                            currentFrameBlob = blob;
                            resolve(blob);
                        }, 'image/jpeg', 0.85);
                    });
                }

                // Helper: Tampilkan Status Loading
                function showLoadingState(msg) {
                    messageText.textContent = msg;
                    loadingSpinner.classList.remove('hidden');
                    messageOverlay.classList.remove('hidden');
                    messageBox.className = "bg-blue-600/90 backdrop-blur-md text-white px-6 py-3 rounded-2xl shadow-2xl border border-blue-400/50 font-semibold text-sm sm:text-base tracking-wide flex items-center gap-3";
                }

                // Helper: Tampilkan Pesan Sukses / Error / Warning
                function showMessage(msg, type) {
                    loadingSpinner.classList.add('hidden');
                    messageText.textContent = msg;
                    messageOverlay.classList.remove('hidden');
                    
                    if (type === 'success') {
                        messageBox.className = "bg-emerald-600/90 backdrop-blur-md text-white px-6 py-3 rounded-2xl font-semibold text-sm sm:text-base tracking-wide shadow-2xl border border-emerald-400/50 flex items-center gap-3";
                        setTimeout(() => {
                            messageOverlay.classList.add('hidden');
                        }, 4000);
                    } else if (type === 'error') {
                        messageBox.className = "bg-rose-600/90 backdrop-blur-md text-white px-6 py-3 rounded-2xl font-semibold text-sm sm:text-base tracking-wide shadow-2xl border border-rose-400/50 flex items-center gap-3";
                        setTimeout(() => {
                            messageOverlay.classList.add('hidden');
                        }, 4000);
                    } else { // warning
                        messageBox.className = "bg-yellow-500/90 backdrop-blur-md text-slate-900 px-6 py-3 rounded-2xl font-bold text-sm sm:text-base tracking-wide shadow-2xl border border-yellow-300 flex items-center gap-3";
                    }
                }

                // Helper: Buka Bottom Sheet Menu
                function openBottomMenu(type) {
                    isPaused = true;
                    video.pause();
                    scanOverlay.classList.add('hidden');
                    
                    registerForm.classList.add('hidden');
                    editForm.classList.add('hidden');
                    
                    if (type === 'register') {
                        registerForm.classList.remove('hidden');
                        newNameInput.value = '';
                        setTimeout(() => newNameInput.focus(), 300);
                    } else if (type === 'edit') {
                        editForm.classList.remove('hidden');
                    }
                    
                    bottomMenu.classList.add('active');
                }

                // Helper: Tutup Bottom Sheet & Resume Kamera
                function resumeScanning() {
                    bottomMenu.classList.remove('active');
                    messageOverlay.classList.add('hidden');
                    
                    setTimeout(() => {
                        video.play();
                        scanOverlay.classList.remove('hidden');
                        isPaused = false;
                        statusDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]";
                        statusText.textContent = "Mencari Wajah...";
                    }, 350);
                }

                // 4. Kirim Frame ke Flask (InsightFace)
                async function processRecognition() {
                    if (isPaused) return;

                    const imageBlob = await captureFrame();
                    if (!imageBlob) return;

                    const formData = new FormData();
                    formData.append("image", imageBlob, "scan.jpg");

                    try {
                        const res = await fetch("/recognize", { method: "POST", body: formData });
                        const data = await res.json();

                        if (data.status === "recognized") {
                            showMessage(data.message, 'success');
                            currentUserId.value = data.user_id;
                            updateNameInput.value = data.name;
                            openBottomMenu('edit');
                        } else if (data.status === "unknown") {
                            showMessage("Wajah Baru Terdeteksi", 'warning');
                            openBottomMenu('register');
                        } else {
                            messageOverlay.classList.add('hidden');
                        }
                    } catch (err) {
                        console.error("Recognition Error:", err);
                        showMessage("Gagal terhubung ke server", 'error');
                    }
                }

                // 5. Submit Form Registrasi Wajah Baru
                async function registerFace() {
                    const name = newNameInput.value.trim();
                    if (!name) return alert("Silakan masukkan nama Anda!");
                    if (!currentFrameBlob) return alert("Frame gambar tidak ditemukan!");

                    const formData = new FormData();
                    formData.append("image", currentFrameBlob, "reg.jpg");
                    formData.append("name", name);

                    try {
                        const res = await fetch("/register", { method: "POST", body: formData });
                        const data = await res.json();

                        if (data.status === "success") {
                            showMessage("Registrasi Berhasil!", 'success');
                            resumeScanning();
                        } else {
                            alert("Gagal: " + data.message);
                        }
                    } catch (err) {
                        alert("Terjadi kesalahan jaringan.");
                    }
                }

                // 6. Submit Form Edit Nama
                async function editName() {
                    const id = currentUserId.value;
                    const newName = updateNameInput.value.trim();
                    if (!newName) return alert("Nama tidak boleh kosong!");

                    const formData = new FormData();
                    formData.append("user_id", id);
                    formData.append("new_name", newName);

                    try {
                        const res = await fetch("/edit_name", { method: "POST", body: formData });
                        const data = await res.json();

                        if (data.status === "success") {
                            showMessage("Nama berhasil diubah!", 'success');
                            resumeScanning();
                        } else {
                            alert("Gagal: " + data.message);
                        }
                    } catch (err) {
                        alert("Terjadi kesalahan jaringan.");
                    }
                }

                // Jalankan Inisialisasi
                window.addEventListener('DOMContentLoaded', initFaceAPI);
            </script>
        </body>
        </html>
    """

# --- ROUTE: PENGENALAN WAJAH ---
@app.route("/recognize", methods=["POST"])
def recognize():
    print("Recognize route called")
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "Gambar tidak ditemukan"}), 400

    file = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    faces = face_app.get(img)

    if len(faces) == 0:
        return jsonify({"status": "error", "message": "Wajah tidak terdeteksi. Coba lagi."}), 400

    # Ambil embedding wajah dari gambar yang diupload (tipe float32)
    unknown_emb = faces[0].embedding

    # Ambil data dari SQLite untuk pencarian 1:N
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM users")
    rows = cursor.fetchall()
    conn.close()

    best_match_name = None
    best_match_id = None
    highest_similarity = -1

    for row in rows:
        user_id, name, emb_blob = row
        # Konversi byte BLOB kembali ke array numpy
        db_emb = np.frombuffer(emb_blob, dtype=np.float32) 
        
        # Hitung Cosine Similarity
        similarity = np.dot(unknown_emb, db_emb) / (np.linalg.norm(unknown_emb) * np.linalg.norm(db_emb))
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_name = name
            best_match_id = user_id

    # Ambang batas ArcFace (0.4)
    if highest_similarity > 0.4:
        return jsonify({
            "status": "recognized",
            "user_id": best_match_id,
            "name": best_match_name,
            "message": get_greeting(best_match_name)
        })
    else:
        return jsonify({
            "status": "unknown",
            "message": "Wajah tidak dikenali. Silakan input nama untuk wajah ini."
        })

# --- ROUTE: REGISTRASI WAJAH BARU ---
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    if not name or "image" not in request.files:
         return jsonify({"status": "error", "message": "Nama dan gambar wajib ada"}), 400

    file = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    faces = face_app.get(img)

    if len(faces) == 0:
        return jsonify({"status": "error", "message": "Wajah tidak terdeteksi saat registrasi."}), 400

    emb = faces[0].embedding
    emb_bytes = emb.tobytes() # Konversi array numpy ke bytes agar bisa masuk SQLite

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, emb_bytes))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": f"Wajah berhasil disimpan sebagai {name}"})

# --- ROUTE: EDIT NAMA ---
@app.route("/edit_name", methods=["POST"])
def edit_name():
    user_id = request.form.get("user_id")
    new_name = request.form.get("new_name")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET name = ? WHERE id = ?", (new_name, user_id))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Nama berhasil diupdate!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003, debug=True)