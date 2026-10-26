# Final Project Oprec PENSHIP - Muh. Rifai Nababbil Munsir (3126500001)

## Identitas
- Nama: Muh. Rifai Nababbil Munsir
- NRP: 3126500001
- Program studi: D3 Teknik Informatika

## Project Roboflow
- Project ID Roboflow: rifais-workspace-lekje/final-project-penship, version 12
- Link Roboflow Universe: https://universe.roboflow.com/rifais-workspace-lekje/final-project-penship

## Dataset
- Jumlah foto asli: bottle 221, laptop 228, pencil 325
- Metode pengambilan foto: Objek difoto dari jarak dekat dan menengah. Latar belakang yang digunakan bervariasi seperti meja, lantai, dan tembok dengan pencahayaan kamar.
- Fitur AI labeling Roboflow yang digunakan: Saya tidak menggunakan fitur AI otomatis. Saya melakukan labeling (bounding box) secara manual agar kotaknya benar-benar pas di objek dan tidak banyak background yang ikut terdeteksi.

## Training
- Lingkungan training: Laptop Lokal (GPU NVIDIA RTX 2050 4GB VRAM)
- Model: yolov8s.pt
- Grafik: lihat folder grafik/

## Pengembangan Kode
Daftar perubahan terhadap kode dasar, beserta alasan dan pengaruhnya terhadap hasil:
1. **Mencegah Error VRAM Habis (OOM):** Saya menambahkan logika `try-except` pada saat *training*. Jika VRAM RTX 2050 saya kepenuhan, kodenya akan otomatis mengurangi *batch size* (membagi dua) dan mencoba lagi tanpa *crash*.
2. **Mengganti Model ke YOLOv8s:** Saya mengubah model dari versi Nano (`yolov8n.pt`) menjadi versi Small (`yolov8s.pt`) agar model bisa lebih pintar dalam mengenali objek, meskipun komputasi *training*-nya sedikit lebih berat.
3. **Menghapus Augmentasi yang Merusak:** Benda kecil seperti pensil sangat susah dideteksi jika pakai augmentasi seperti `mixup` (ditumpuk) atau `erasing` (ditutup kotak hitam). Jadi, saya menghapus pengaturan itu dan hanya menggunakan augmentasi dasar seperti pencahayaan (`hsv_v`) dan `mosaic`.
4. **Otomatisasi Laporan CSV:** Saya menambahkan skrip agar hasil evaluasi `mAP50` dan waktu *training* otomatis tercatat ke dalam *file* CSV agar mempermudah evaluasi akhir.

## Hasil Eksperimen
| No | Skenario | Epochs | Batch | Optimizer | mAP50 | mAP50-95 | Keterangan |
|:--:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | Baseline (kode dasar) | 50 | 16 | auto | - | - | Titik acuan awal |
| 1 | Augmentasi Bawaan (Mixup/Erasing) | 100 | 16 | auto | 0.089 | - | Akurasi sangat anjlok karena objek pensil tidak terdeteksi akibat gambar terlalu terdistorsi augmentasi. |
| 2 | Optimasi Akhir (YOLOv8s, Bebas Halusinasi) | 30 | 16 | auto | 0.553 | 0.364 | Peningkatan pesat. Model sangat akurat mengenali laptop dan botol. |

## Analisis dan Kendala
- **Analisis penyebab skenario terbaik menghasilkan nilai lebih tinggi:** Mengganti model menjadi `yolov8s` dan menghapus augmentasi yang merusak (`mixup`, `erasing`) sangat membantu performa. Model berhasil mendapatkan akurasi sangat sempurna untuk deteksi laptop (1.00) dan sangat baik untuk botol (0.77).
- **Kendala yang ditemui dan cara penyelesaiannya:** Kendala utama ada pada objek pensil. Karena pensil bentuknya sangat tipis dan panjang, model sering salah menebak garis di tembok atau lantai sebagai pensil (*false positive* tinggi di bagian background). Solusi yang saya terapkan saat *training* adalah menghapus augmentasi yang berisiko merusak bentuk benda tipis. Lalu, untuk mengurangi salah tebak saat percobaan kamera (inferensi), saya menggunakan bantuan latar belakang kertas HVS putih dan merekam pensil dari jarak dekat agar lebih mudah dikenali oleh sistem.