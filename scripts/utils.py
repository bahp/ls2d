


if __name__ == '__main__':

    # ------------------
    # Libraries
    # ------------------
    # Generic libraries
    import pandas as pd
    import numpy as np

    # Own
    from ls2d.utils import format_workbench
    from ls2d.utils import format_pipeline
    from ls2d.utils import format_demographics

    # ------------------
    # Main
    # ------------------
    # Constants
    PATH ='./objects/results.csv'

    # Load data
    df = pd.read_csv(PATH)

    # Formattings
    wb = format_workbench(df)
    pp = format_pipeline(df.loc[0, :])

    # Show
    print("\nResults:")
    print(df.dtypes)
    print(df)
    print("\n\nWorkbench:")
    print(wb.dtypes)
    print(wb)
    print("\n\nPipeline:")
    print(pp)