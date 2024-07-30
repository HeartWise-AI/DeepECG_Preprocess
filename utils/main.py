import argparse

from CLI_xml2df import tinyxml2df
from ecg_plotter import create_ecg_plotter
from multiprocessing_utils import batch_process_from_dataframe


def main():
    parser = argparse.ArgumentParser(description="Plot ECG data from various sources")
    parser.add_argument(
        "source_type",
        choices=["xml", "npy", "dataframe", "batch_dataframe"],
        help="Source type of ECG data",
    )
    parser.add_argument("source_path", help="Path to the source file")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index for dataframe (either CSV or PARQUET) file (ignored for XML and NPY)",
    )
    parser.add_argument(
        "--dataset", choices=["MHI", "MIMICIV"], default="MIMICIV", help="Dataset type"
    )
    parser.add_argument("--out_dir", default="tmp/", help="Output directory for saved plots")
    parser.add_argument("--width", type=int, default=2500, help="Width of the output image")
    parser.add_argument("--save", action="store_true", help="Save the plot instead of displaying")
    parser.add_argument("--anonymize", action="store_true", help="Anonymize the saved plot")
    parser.add_argument(
        "--processes", type=int, default=1, help="Number of processes for batch processing"
    )
    parser.add_argument(
        "--output_csv",
        default="batch_processing_results.csv",
        help="Path to save the output CSV file for batch processing",
    )
    parser.add_argument(
        "--npy_path_column",
        default="npy_path",
        help="Column name for NPY file path in dataframe (either CSV or PARQUET)",
    )
    parser.add_argument(
        "--patient_id_column",
        default="new_PatientID",
        help="Column name for patient ID in dataframe (either CSV or PARQUET)",
    )
    parser.add_argument(
        "--diagnosis_column",
        default="dictionary_diagnosis",
        help="Column name for diagnosis in dataframe (either CSV or PARQUET)",
    )
    parser.add_argument(
        "--show_diagnosis", action="store_true", help="Show diagnosis in the plot"
    )
    args = parser.parse_args()

    if args.source_type == "batch_dataframe":
        batch_process_from_dataframe(
            args.source_path,
            args.out_dir,
            args.dataset,
            processes=args.processes,
            output_csv=args.output_csv,
            npy_path_column=args.npy_path_column,
            patient_id_column=args.patient_id_column,
            diagnosis_column=args.diagnosis_column,
            anonymize=args.anonymize,
            show_diagnosis=args.show_diagnosis,
        )
    else:
        plotter = create_ecg_plotter(
            args.source_type,
            args.source_path,
            index=args.index,
            dataset=args.dataset,
            out_dir=args.out_dir,
            width=args.width,
            npy_path_column=args.npy_path_column,
            patient_id_column=args.patient_id_column,
            diagnosis_column=args.diagnosis_column,
        )

        plotter.plot_ecg(
            save=args.save, anonymize=args.anonymize, show_diagnosis=args.show_diagnosis
        )


if __name__ == "__main__":
    main()
