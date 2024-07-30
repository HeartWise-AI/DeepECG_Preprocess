from multiprocessing import Pool

import pandas as pd
from ecg_plotter import create_ecg_plotter
from tqdm import tqdm


def process_ecg(args):
    plotter, title, save, anonymize = args
    return plotter.plot_ecg(title=title, save=save, anonymize=anonymize)


def process_multiple_ecgs(plotters, titles, save=True, anonymize=True, processes=8):
    with Pool(processes=processes) as pool:
        args = [(plotter, title, save, anonymize) for plotter, title in zip(plotters, titles)]
        results = list(
            tqdm(pool.imap(process_ecg, args), total=len(plotters), desc="Processing ECGs")
        )
    return results


def process_ecg_from_dataframe(args):
    (
        df_ecg,
        index,
        out_dir,
        dataset,
        npy_path_column,
        patient_id_column,
        diagnosis_column,
        anonymize,
        show_diagnosis,
    ) = args
    plotter = create_ecg_plotter(
        source_type="dataframe",
        source_path=df_ecg,
        index=index,
        npy_path_column=npy_path_column,
        patient_id_column=patient_id_column,
        diagnosis_column=diagnosis_column,
        dataset=dataset,
        out_dir=out_dir,
    )
    img, result = plotter.plot_ecg(save=True, anonymize=anonymize, show_diagnosis=show_diagnosis)
    return {"index": index, "result": result}


def batch_process_from_dataframe(
    dataframe_filepath,
    out_dir,
    dataset,
    processes=8,
    output_csv=None,
    npy_path_column="npy_path",
    patient_id_column="new_PatientID",
    diagnosis_column="dictionary_diagnosis",
    anonymize=False,
    show_diagnosis=True,
):
    if dataframe_filepath.endswith(".parquet"):
        df_ecg = pd.read_parquet(dataframe_filepath)
    elif dataframe_filepath.endswith(".csv"):
        df_ecg = pd.read_csv(dataframe_filepath)
    else:
        raise ValueError("Input file must be either .parquet or .csv")

    from multiprocessing import Manager

    with Manager() as manager:
        shared_list = manager.list()

        with Pool(processes=processes) as pool:
            raw_results = list(
                tqdm(
                    pool.imap(
                        process_ecg_from_dataframe,
                        [
                            (
                                df_ecg,
                                index,
                                out_dir,
                                dataset,
                                npy_path_column,
                                patient_id_column,
                                diagnosis_column,
                                anonymize,
                                show_diagnosis,
                            )
                            for index in range(len(df_ecg))
                        ],
                    ),
                    total=len(df_ecg),
                    desc="Processing ECGs from Parquet",
                )
            )

            # Extract required information from results and add to the shared list
            for result in raw_results:
                index = result["index"]
                npy_path = df_ecg.iloc[index][npy_path_column]
                output_file = result["result"]
                shared_list.append(
                    {"index": index, "npy_path": npy_path, "output_file": output_file}
                )

        # Create DataFrame from the shared list
        df_results = pd.DataFrame(list(shared_list))

    if output_csv:
        import os

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return df_results
