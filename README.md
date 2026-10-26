# Project-1-PCV
# Gundam Dodge: Mini Game Berbasis Warna Tangan

Game horizontal dengan tema space. Player mengendalikan **Wing Gundam Zero (mode burung)** untuk menghindari serangan **Zaku** yang datang dari kanan. Pemain **tidak memakai keyboard atau mouse**. Pesawat dikendalikan lewat **warna sarung tangan** yang terbaca webcam, memakai Python dan OpenCV **tanpa MediaPipe**.

## Gimmick

- **Sarung tangan putih** membuat pesawat maju.
- **Sarung tangan hitam** membuat pesawat mundur.
- **Tinggi tangan di kamera** menentukan pesawat naik atau turun.
- **Zaku acak**: dua jenis Zaku muncul dari kanan dengan ketinggian dan kecepatan acak.
- **Level naik otomatis**: Zaku makin cepat dan makin sering.
- **Nyawa dan kebal sementara**: 3 nyawa, pesawat berkedip dan kebal sekitar 2 detik setelah tertabrak.
- **Background bergulir** untuk memberi kesan melaju di luar angkasa.
- **Foto referensi latar**: tombol `C` mengambil foto latar tanpa tangan, sehingga rambut, baju, dan dinding tidak ikut terbaca.

## Kontrol

| Input | Aksi |
|---|---|
| Sarung tangan putih | Maju |
| Sarung tangan hitam | Mundur |
| Tinggi tangan | Naik / turun |
| `C` | Ambil foto referensi latar (**tanpa tangan di frame**) |
| `P` | Pause |
| `R` | Restart |
| `ESC` | Keluar |

## Penerapan Materi 

| Materi | Penerapan di proyek | Lokasi |
|---|---|---|
| **Read image** | Memuat gambar player, Zaku, dan background dengan `cv2.imread` (termasuk channel alpha untuk transparansi) | `game.py` (`load_img`, `load_background`) |
| **Show image** | Menampilkan game dan preview kamera dengan `cv2.imshow` | `main.py`, `tes_detector.py` |
| **Filter color image** | Mask warna dengan `cv2.inRange` pada ruang warna HSV | `detector.py` (`detect`) |
| **Filter color video** | Filter warna diterapkan pada tiap frame video webcam (`cv2.VideoCapture`) secara real-time | `main.py`, `detector.py` |
| **Transformasi intensitas** | Konversi ke grayscale dan *thresholding* (`cv2.threshold`) pada selisih frame terhadap foto referensi untuk memisahkan objek baru (sarung tangan) dari latar | `detector.py` (`_foreground`) |
| **Filter spasial** | *Gaussian blur* (`cv2.GaussianBlur`) untuk menghaluskan gambar dan mengurangi noise sebelum diproses | `detector.py` |
| **Model warna** | Konversi **BGR ke HSV** (`cv2.cvtColor`). Putih = saturasi rendah dan kecerahan tinggi, hitam = kecerahan rendah | `detector.py`, `kalibrasi.py` |
| **Morfologi** | *Opening* (membuang bintik noise) dan *closing* (menutup lubang pada objek) dengan `cv2.morphologyEx` | `detector.py` (`_clean`, `_foreground`) |

## Cara Kerja Singkat

1. Webcam mengambil frame, lalu frame dicerminkan.
2. Frame diburamkan (Gaussian blur) dan diubah ke HSV.
3. Selisih frame terhadap foto referensi membuat mask objek baru.
4. `inRange` membuat mask putih dan mask hitam, lalu digabung dengan mask objek baru.
5. Morfologi opening dan closing membersihkan mask.
6. Kontur terbesar yang lolos filter ukuran dan bentuk dipilih. Warnanya menentukan aksi dan posisinya menentukan ketinggian pesawat.
7. Game memperbarui pesawat, Zaku, tabrakan, dan skor, lalu menggambar hasilnya.

## Struktur Proyek

```
├── assets/            gambar player, Zaku, dan background
├── detector.py        deteksi warna sarung tangan
├── kalibrasi.py       kalibrasi batas warna putih dan hitam
├── game.py            logika game
├── main.py            program utama
├── tes_kamera.py      tes webcam
├── tes_gambar.py      tes pemuatan gambar
├── tes_detector.py    tes dan penyetelan deteksi
└── requirements.txt
```

## Instalasi dan Menjalankan

1. Install Python 3.10 atau lebih baru.
2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install library:
   ```bash
   pip install -r requirements.txt
   ```
4. **Kalibrasi warna** (sekali, atau saat cahaya berubah):
   ```bash
   python kalibrasi.py
   ```
   Tekan `W` lalu klik 8-10 titik di sarung tangan putih, tekan `B` lalu klik 8-10 titik di sarung tangan hitam, tekan `S` untuk menyimpan ke `warna.json`.
5. Jalankan game:
   ```bash
   python main.py
   ```
   Keluarkan tangan dari frame, duduk diam, lalu tekan `C` untuk mengambil foto referensi. Setelah itu game siap dimainkan.

## Tips Agar Deteksi Akurat

- Gunakan sarung tangan putih dan hitam polos.
- Pakai baju berwarna kalau bisa bukan putih atau hitam.
- Duduk diam setelah menekan `C`. Tekan `C` lagi kalau bergeser atau cahaya berubah.
- Gunakan cahaya merata dari depan.

## Teknologi

Python, OpenCV, NumPy
