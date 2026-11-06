import hashlib
import io
import json
import zipfile
import logging
import base64
import pandas as pd
from PIL import UnidentifiedImageError, Image
import streamlit as st
from inference import ROOT, experiments, run_detector, read_image, png_bytes

st.set_page_config(page_title='Rupa Beras · Meja Inspeksi', page_icon=':material/grain:', layout='wide')
@st.cache_data
def font_css():
    font = base64.b64encode((ROOT / 'assets/fonts/Manrope.ttf').read_bytes()).decode()
    return "@font-face{font-family:'Manrope';src:url(data:font/ttf;base64," + font + ") format('truetype');font-weight:200 800;font-display:swap;}"

st.markdown(f'<style>{font_css()}</style>', unsafe_allow_html=True)
st.markdown(f'<style>{(ROOT / "assets/style.css").read_text()}</style>', unsafe_allow_html=True)
runs = experiments()
labels = {p.name: p for p in runs}

def model_label(name):
    return name

with st.sidebar:
    st.markdown('''<div class="app-brand"><span class="brand-icon"><svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M7 20 17 4M10 15c-5 0-6-4-5-7 4 0 7 2 5 7ZM14 10c-1-5 2-7 6-7 1 4-1 7-6 7Z"/></svg></span><div>Rupa Beras<small>Inspeksi varietas beras</small></div></div>''', unsafe_allow_html=True)
    page = st.radio('Ruang kerja', ['Deteksi & bandingkan', 'Panduan penggunaan'], label_visibility='collapsed')
    st.divider()
    st.markdown('<div class="control-heading">Konfigurasi analisis</div>', unsafe_allow_html=True)
    ready = [p.name for p in runs]
    default = next((n for n in ready if n.startswith('yolov8n')), ready[0] if ready else None)
    selected = st.multiselect('Model yang digunakan', ready, default=[default] if default else [])
    confidence = st.slider('Ambang confidence', .05, .95, .25, .05,
                           help='Nilai lebih tinggi menyaring prediksi dengan keyakinan rendah.')
    with st.expander('Pengaturan inferensi'):
        iou = st.slider('IoU NMS · YOLO', .1, .9, .7, .05,
                        help='RT-DETR memakai prediksi set dan tidak menggunakan NMS yang sama dengan YOLO.')
        size = st.selectbox('Resolusi inferensi', [640, 768, 800], help='Resolusi yang sama digunakan untuk semua model terpilih.')
        import torch
        device_label = st.selectbox('Perangkat', ['CPU'] + (['CUDA GPU'] if torch.cuda.is_available() else []))
        device = '0' if device_label == 'CUDA GPU' else 'cpu'

st.markdown(f'<div class="workspace-topbar"><span>Ruang kerja / {page}</span><span class="model-availability">{len(ready)} model tersedia</span></div>', unsafe_allow_html=True)

if page == 'Deteksi & bandingkan':
    st.title('Deteksi varietas beras')
    st.markdown('<p class="intro">Periksa varietas, lokasi, dan jumlah butir. Bandingkan hasil beberapa model dalam satu analisis.</p>', unsafe_allow_html=True)
    left, right = st.columns([1.6, 1], gap='large')
    with left:
        st.subheader('Gambar yang akan dianalisis')
        upload = st.file_uploader('Unggah gambar beras', type=['png', 'jpg', 'jpeg', 'webp'],
                                  help='PNG, JPG, atau WebP. Maksimal 20 MB dan 25 megapiksel.')
    with right:
        st.subheader('Siapkan gambar Anda')
        st.write('Gunakan gambar tajam dengan pencahayaan merata dan latar polos. Usahakan setiap butir terlihat jelas dan tidak saling menutupi.')
        st.caption('Petunjuk lengkap tersedia di menu Panduan penggunaan.')
    if upload is None:
        st.markdown('''<div class="empty"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"/><circle cx="8" cy="8" r="1.5"/><path d="m4 17 5-5 4 4 3-3 4 4"/></svg><h3>Hasil inspeksi akan tampil di sini</h3><p>Unggah gambar dan jalankan deteksi untuk melihat anotasi, jumlah varietas, dan perbandingan model.</p></div>''', unsafe_allow_html=True)
        st.button('Jalankan deteksi', type='primary', disabled=True)
    else:
        data = upload.getvalue()
        try:
            if len(data) > 20 * 1024 * 1024:
                raise ValueError('Ukuran berkas melebihi 20 MB.')
            image = read_image(data)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f'Gambar tidak dapat dibaca: {exc} Pilih berkas gambar lain.')
            st.stop()
        with left:
            st.image(image, caption=f'Citra asli · {image.width} × {image.height} piksel', width='stretch')
        signature = hashlib.sha256(data + json.dumps([selected, confidence, iou, size, device]).encode()).hexdigest()
        if st.button('Jalankan deteksi', type='primary', disabled=not selected, icon=':material/search:'):
            outputs, errors = {}, {}
            for name in selected:
                with st.spinner(f'Menganalisis dengan {model_label(name)}…'):
                    try:
                        outputs[name] = run_detector(str(labels[name] / 'weights/best.pt'),
                            image, confidence, iou, size, device)
                    except Exception as exc:
                        logging.exception('Inference failed for %s', name)
                        errors[name] = 'Analisis belum berhasil. Coba kembali atau pilih model lain.'
            st.session_state['analysis'] = {'signature': signature, 'outputs': outputs, 'errors': errors}
        state = st.session_state.get('analysis')
        if state and state['signature'] != signature:
            st.info('Gambar atau pengaturan berubah. Jalankan deteksi untuk memperbarui hasil.')
        if state and state['signature'] == signature:
            for name, error in state['errors'].items():
                st.error(f'{model_label(name)}: {error}')
            outputs = state['outputs']
            if outputs:
                st.divider()
                st.subheader('Hasil inspeksi')
                st.caption('Nilai keyakinan menunjukkan seberapa yakin model terhadap prediksi setiap butir, bukan jaminan ketepatan.')
                summary = []
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                    for name, out in outputs.items():
                        frame = out['rows']
                        summary.append({'Model': model_label(name), 'Butir': len(frame),
                            'Keyakinan rata-rata': round(float(frame.confidence.mean()), 3) if len(frame) else None,
                            'Inferensi (ms)': round(out['inference_ms'], 2), 'Total proses (ms)': round(out['elapsed_ms'], 2)})
                        bundle.writestr(f'{name}/deteksi.csv', frame.to_csv(index=False))
                        bundle.writestr(f'{name}/anotasi.png', png_bytes(out['image']))
                    bundle.writestr('pengaturan.json', json.dumps({'confidence': confidence, 'iou_yolo': iou, 'imgsz': size, 'device': device, 'models': selected}, indent=2))
                    bundle.writestr('perbandingan.csv', pd.DataFrame(summary).to_csv(index=False))
                st.dataframe(pd.DataFrame(summary), hide_index=True, width='stretch')
                st.caption('Waktu pemrosesan bergantung pada perangkat, resolusi, dan model yang dipilih. Analisis pertama dapat memerlukan waktu lebih lama.')
                for tab, (name, out) in zip(st.tabs([model_label(n) for n in outputs]), outputs.items()):
                    with tab:
                        visual, detail = st.columns([1.7, 1], gap='large')
                        with visual:
                            st.image(out['image'], caption=model_label(name), width='stretch')
                        with detail:
                            frame = out['rows']
                            st.metric('Butir terdeteksi', len(frame))
                            if frame.empty:
                                st.info('Belum ada butir terdeteksi. Coba turunkan ambang keyakinan atau gunakan gambar yang lebih jelas dengan latar polos.')
                            else:
                                counts = frame.groupby('varietas').size().reindex(out['names'], fill_value=0).rename('Jumlah butir')
                                st.bar_chart(counts, color='#246347')
                            st.download_button('Unduh gambar anotasi', png_bytes(out['image']), f'{name}.png', 'image/png', key=f'png-{name}')
                        with st.expander('Detail prediksi setiap butir'):
                            st.dataframe(frame.rename(columns={'varietas': 'Varietas', 'confidence': 'Keyakinan'}), hide_index=True, width='stretch')
                st.download_button('Unduh semua hasil (.zip)', archive.getvalue(), 'hasil-inspeksi-beras.zip', 'application/zip', type='primary')
else:
    st.title('Panduan penggunaan')
    st.markdown('<p class="intro">Dari persiapan gambar hingga menyimpan hasil inspeksi.</p>', unsafe_allow_html=True)
    st.subheader('Menjalankan deteksi')
    st.markdown("""
1. **Siapkan gambar.** Gunakan PNG, JPG, JPEG, atau WebP dengan ukuran maksimal 20 MB dan 25 megapiksel. Pilih gambar tajam, latar polos, dan pencahayaan merata.
2. **Pilih model.** Gunakan panel samping untuk memilih model deteksi. Model dengan nama riceblue mendeteksi Ciherang, IR64, dan Mentik; model rice lainnya mendeteksi Arborio, Basmati, Ipsala, Jasmine, dan Karacadag.
3. **Pilih model deteksi.** Gunakan satu model untuk analisis tunggal atau beberapa model untuk membandingkan hasil pada gambar yang sama. Pilih model dengan kelompok varietas yang sama untuk perbandingan yang sesuai.
4. **Unggah gambar dan jalankan deteksi.** Tekan **Jalankan deteksi**, lalu tunggu hingga analisis selesai.
5. **Periksa dan simpan hasil.** Lihat kotak deteksi, jumlah tiap varietas, dan detail prediksi. Unduh gambar anotasi atau seluruh hasil sebagai ZIP.
""")
    st.divider()
    reading, settings = st.columns(2, gap='large')
    with reading:
        st.subheader('Membaca hasil')
        st.markdown("""
- **Kotak deteksi** menunjukkan lokasi dan prediksi varietas setiap butir.
- **Butir terdeteksi** menghitung prediksi yang lolos ambang keyakinan.
- **Keyakinan** adalah nilai antara 0 dan 1. Nilai 0,90 berarti model memiliki keyakinan tinggi terhadap prediksinya; nilai ini bukan akurasi keseluruhan.
- **Grafik varietas** merangkum jumlah butir per kelas untuk setiap model.
- **Tab model** menampilkan hasil masing-masing model secara terpisah. Jumlah deteksi antarmodel tidak dijumlahkan.
""")
    with settings:
        st.subheader('Menyesuaikan analisis')
        st.markdown("""
- **Ambang confidence:** naikkan jika banyak objek keliru terdeteksi; turunkan jika butir belum terdeteksi.
- **IoU NMS · YOLO:** mengatur penyaringan kotak pada YOLO. Nilai lebih rendah menyaring kotak yang bertumpuk lebih kuat. Pengaturan ini tidak berlaku pada RT-DETR.
- **Resolusi inferensi:** gunakan resolusi yang sama ketika membandingkan beberapa model. Resolusi lebih tinggi dapat menambah waktu proses.
- **Perangkat:** CPU tersedia secara bawaan. Pilihan GPU muncul jika perangkat mendukungnya.
""")
    st.subheader('Pertanyaan umum')
    with st.expander('Mengapa tidak ada butir yang terdeteksi?'):
        st.write('Pastikan kelompok varietas sudah sesuai. Coba gambar lebih tajam dengan latar polos, kurangi butir yang saling menutupi, atau turunkan ambang keyakinan, lalu jalankan kembali.')
    with st.expander('Mengapa hasil antar model berbeda?'):
        st.write('Setiap model dapat menghasilkan kotak, varietas, dan keyakinan berbeda. Bandingkan hasil pada gambar serta pengaturan yang sama, lalu periksa prediksi secara visual.')
    with st.expander('Apa yang terjadi setelah gambar atau pengaturan diubah?'):
        st.write('Jalankan deteksi kembali untuk memperbarui hasil. Aplikasi menandai hasil sebelumnya agar tidak tertukar dengan pengaturan baru.')
    with st.expander('Apa isi unduhan ZIP?'):
        st.write('ZIP berisi gambar anotasi dan tabel CSV koordinat serta keyakinan untuk setiap model, tabel perbandingan, dan pengaturan analisis dalam JSON. Koordinat x1, y1, x2, y2 menggunakan piksel pada gambar asli.')

st.markdown('<div class="footer"><span>Rupa Beras</span><span>Deteksi varietas dan penghitungan butir</span></div>', unsafe_allow_html=True)
