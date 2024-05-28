## DeepECG Repo

![image](flowchart_deepecg.drawio.png)

#### CLI_tinixml2dict.py:

Generate the df with the specified directory

`python cli_xml2df.py --xml_path "/media/data1/muse_ge/ecg_retrospective" --out_path '.' --verbose True --save True`

#### CLI_TinyGetWaveform.py

Generate the lead data with the specified file

`python cli_lead.py --data_path "./df_xml_2022_11_27_n_738393.csv" --out_path "."`

### Extract PNG files from parquet

`python utils/generate_png_from_parquet.py /media/data1/muse_ge/ECG_ad202207_1453937_cat_labels_MUSE_vs_CARDIOLOGIST_v1.2.parquet --processes 8 --output_csv data/df_ecg_parquet.csv --outdir ecg_png_parquet3/`
