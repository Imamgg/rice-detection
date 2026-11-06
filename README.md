# Rupa Beras

Aplikasi Streamlit untuk deteksi varietas dan jumlah butir beras menggunakan YOLOv3, YOLOv8n, serta RT-DETR-L. Mendukung perbandingan beberapa model, anotasi gambar, grafik varietas, ekspor PNG/CSV/ZIP, dan panduan penggunaan.

## Menjalankan secara lokal

Gunakan Python 3.11. Pada PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Buka http://localhost:8501. Bobot model diunduh otomatis saat model pertama kali digunakan dan disimpan dalam folder `models/`. Proses pertama memerlukan koneksi internet dan dapat lebih lama. Akses unduhan mengikuti ketersediaan serta kuota Google Drive; kegagalan dicatat pada log server dan pengguna dapat mencoba kembali.

Untuk mengunduh semua bobot sebelum server dijalankan, opsional:

```powershell
.\.venv\Scripts\python scripts/download_models.py
```

## Deployment Streamlit

1. Hubungkan repositori `Imamgg/rice-detection` ke layanan hosting Streamlit.
2. Pilih branch `main`, entrypoint `app.py`, dan Python 3.11.
3. Pasang dependensi dari `requirements.txt` (otomatis pada Streamlit Community Cloud).
4. Pastikan server dapat mengunduh file dari Google Drive dan menulis ke folder `models/`.
5. Sediakan memori yang cukup untuk PyTorch dan model pilihan. YOLOv8n merupakan bobot paling kecil; YOLOv3 memerlukan memori lebih besar. Maksimum dua model disimpan dalam cache.

Aplikasi menggunakan CPU secara bawaan. GPU tersedia jika PyTorch mendeteksi CUDA. Restart pada hosting dengan penyimpanan sementara dapat mengharuskan pengunduhan ulang. Push ke GitHub belum mempublikasikan server aplikasi.

## Penggunaan

Pilih satu atau beberapa model, unggah satu PNG/JPG/JPEG/WebP, kemudian tekan **Jalankan deteksi**. Batas unggah 20 MB dan 25 megapiksel. Periksa hasil setiap model melalui tab dan unduh hasil sebagai PNG atau ZIP. Jalankan ulang setelah mengganti gambar atau pengaturan. Menu **Panduan penggunaan** menjelaskan alur dan interpretasi hasil.

Model `riceblue` mengenali Ciherang, IR64, dan Mentik. Model `rice` lainnya mengenali Arborio, Basmati, Ipsala, Jasmine, dan Karacadag. Bandingkan model dengan kelompok varietas yang sesuai. Confidence merupakan keyakinan model, bukan akurasi keseluruhan; mAP tidak dihitung tanpa ground truth.

## Isi repositori

- `app.py`: antarmuka dan ekspor hasil.
- `inference.py`: pembacaan gambar, inferensi, dan anotasi.
- `model_store.py`: pemetaan dan pengunduhan bobot sesuai manifest.
- `drive-manifest.json`: enam ID checkpoint yang digunakan aplikasi.
- `assets/style.css` dan `.streamlit/config.toml`: tampilan serta konfigurasi Streamlit.
- `requirements.txt`: dependensi deployment.
- `scripts/download_models.py`: pengunduhan awal opsional.

Bobot `.pt`, dataset, laporan HKI, paper, tangkapan layar, artefak pelatihan, file sementara, dan konfigurasi agen tidak disertakan. Jangan menyimpan kredensial dalam repo; `.env` dan `.streamlit/secrets.toml` diabaikan Git.
