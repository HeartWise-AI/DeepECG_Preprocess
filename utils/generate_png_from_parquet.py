import argparse
from multiprocessing import Pool

import pandas as pd
from plot_ecg_from_xml import plot_ecg_from_xml
from tqdm import tqdm


def process_xml_path(args):
    xml_path, out_dir, dataset = args
    result = plot_ecg_from_xml(
        xml_path,
        out_dir=out_dir,
        title="",
        save=True,
        anonymize=False,
        width=1250,
        dataset=dataset,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PNGs from ECG XML files in a Parquet file"
    )
    parser.add_argument(
        "parquet_file", type=str, help="Path to the Parquet file containing XML paths"
    )
    parser.add_argument("--processes", type=int, default=8, help="Number of processes to use")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/df_ecg_parquet.csv",
        help="Path to save the output CSV file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="ecg_png_parquet/",
        help="Directory to save the output PNG files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MHI", "MIMICIV"],
        default="MHI",
        help="Dataset type (MHI or MIMICIV)",
    )

    args = parser.parse_args()

    df_parquet = pd.read_parquet(args.parquet_file)

    with Pool(processes=args.processes) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_xml_path,
                    [(xml_path, args.outdir, args.dataset) for xml_path in df_parquet.xml_path],
                ),
                total=len(df_parquet.xml_path),
                desc="Processing XML files",
            )
        )

    df_result = pd.DataFrame(results)
    df_result.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
