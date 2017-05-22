#Run app
docker run -v $(pwd):/home/data -t -i carlosym1/rkt_dicom_ecg_peaks_detection data/in/ data/out/