## DeepECG Repo

![image](flowchart_deepecg.drawio.png)

#### CLI_tinixml2dict.py:

Generate the df with the specified directory

`python cli_xml2df.py --xml_path "/media/data1/muse_ge/ecg_retrospective" --out_path '.' --verbose True --save True`

#### CLI_TinyGetWaveform.py

Generate the lead data with the specified file

`python cli_lead.py --data_path "./df_xml_2022_11_27_n_738393.csv" --out_path "."`

### Extract PNG files from parquet

#### For MHI data

`python main.py batch_dataframe /media/data1/muse_ge/ECG_ad202207_1453937_cat_labels_MUSE_vs_CARDIOLOGIST_v1.2.parquet --processes 8 --output_csv data/df_ecg_parquet.csv --out_dir ecg_png_parquet3/ --dataset MHI --anonymize --show_diagnosis`

#### For MIMICIV data

`python main.py batch_dataframe /volume/DeepECG_Preprocess/utils/MIMICIV.csv --processes 8 --output_csv data/df_ecg_mimiciv.csv --out_dir ecg_png_mimiciv/ --dataset MIMICIV --diagnosis_column report --npy_path_column waveform_path --anonymize --show_diagnosis`

### Plot individual ECG from different sources

#### From XML file

`python main.py xml path/to/your/xml_file.xml --save --anonymize --show_diagnosis`

#### From NPY file

`python main.py npy path/to/your/npy_file.npy --save --anonymize --show_diagnosis`

#### From dataframe (CSV or Parquet)

`python main.py dataframe path/to/your/dataframe.csv --index 0 --save --anonymize --show_diagnosis`

Replace `path/to/your/file` with the actual path to your input file. Adjust other parameters as needed:

- `--save`: Save the plot instead of displaying
- `--anonymize`: Anonymize the saved plot removing MRN, Date, etc
- `--show_diagnosis`: Show diagnosis in the plot (on the fiugure)
- `--width`: Set the width of the output image (default: 2500)
- `--out_dir`: Specify the output directory for saved plots (default: "tmp/")
- `--dataset`: Choose the dataset type (MHI or MIMICIV, default: MIMICIV)

For dataframe sources, you can also specify column names:

- `--npy_path_column`: Column name for NPY file path (default: "npy_path")
- `--patient_id_column`: Column name for patient ID (default: "new_PatientID")
- `--diagnosis_column`: Column name for diagnosis (default: "dictionary_diagnosis")
