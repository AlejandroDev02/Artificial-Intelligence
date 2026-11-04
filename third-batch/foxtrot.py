#ON PANDAS AND DATASET MANIPULATION BECAUSE I FUCKED UP LAST TIME

import pandas as pd
import numpy as np

def showcase_pandas_clipping():
    # 1. Create a dummy dataset
    data = {
        'A': range(1, 11),
        'B': range(11, 21),
        'C': range(21, 31),
        'D': range(31, 41),
        'E': range(41, 51)
    }
    df = pd.DataFrame(data)
    
    print("--- Original DataFrame ---")
    print(df.head(3), "\n...\n")

    # ==========================================
    # WORKING WITH COLUMNS
    # ==========================================

    # 1. Getting Headers
    headers = df.columns.tolist()
    print("Headers:", headers, "\n")

    # 2. Cutting off the very last column
    # iloc uses integer-location based indexing. [:, :-1] means all rows, all columns except the last.
    df_no_last_col = df.iloc[:, :-1]
    print("--- Cut off the last column (iloc[:, :-1]) ---")
    print(df_no_last_col.head(3), "\n")

    # 3. Selecting specific columns by name
    df_selected_cols = df[['A', 'C', 'E']]
    print("--- Specific columns (df[['A', 'C', 'E']]) ---")
    print(df_selected_cols.head(3), "\n")

    # 4. Dropping specific columns
    df_dropped = df.drop(columns=['B', 'D'])
    print("--- Dropped columns B and D (df.drop(columns=['B', 'D'])) ---")
    print(df_dropped.head(3), "\n")

    # 5. Slicing columns between two names
    # loc uses label-based indexing.
    df_col_slice = df.loc[:, 'B':'D']
    print("--- Slice columns B to D (loc[:, 'B':'D']) ---")
    print(df_col_slice.head(3), "\n")

    # 6. Filter columns using string matching/regex
    df_filtered_cols = df.filter(regex='^(A|E)$')
    print("--- Filter columns by regex matching A or E ---")
    print(df_filtered_cols.head(3), "\n")


    # ==========================================
    # WORKING WITH ROWS
    # ==========================================

    # 7. Clipping the top N rows
    df_head = df.head(4)
    print("--- Top 4 rows (head(4)) ---")
    print(df_head, "\n")

    # 8. Clipping the bottom N rows
    df_tail = df.tail(3)
    print("--- Bottom 3 rows (tail(3)) ---")
    print(df_tail, "\n")

    # 9. Slicing rows by index position
    # Cuts rows starting from index 2 up to (but not including) index 6
    df_row_slice = df.iloc[2:6, :]
    print("--- Rows by position 2 to 5 (iloc[2:6, :]) ---")
    print(df_row_slice, "\n")

    # 10. Truncating rows before or after specific index labels
    df_truncated = df.truncate(before=2, after=5)
    print("--- Truncated rows before index 2 and after 5 (truncate()) ---")
    print(df_truncated, "\n")

    # 11. Dropping specific rows by index
    df_drop_rows = df.drop(index=[0, 1, 9])
    print("--- Dropped rows 0, 1, and 9 (drop(index=[...])) ---")
    print(df_drop_rows.head(4), "\n")


    # ==========================================
    # SUBSETTING BOTH ROWS AND COLUMNS
    # ==========================================

    # 12. Cutting rows and columns simultaneously
    # e.g., First 5 rows, and columns 2 to 4 (excluding 4)
    df_subset_both = df.iloc[0:5, 1:3]
    print("--- Subset both: rows 0-4, cols 1-2 (iloc[0:5, 1:3]) ---")
    print(df_subset_both, "\n")

    # ==========================================
    # FIXING HEADERS STUCK IN THE FIRST ROW
    # ==========================================

    # 13. Elevating the first row to act as headers and clipping it out
    messy_data = {
        0: ['ID', 101, 102, 103],
        1: ['Name', 'Alice', 'Bob', 'Charlie'],
        2: ['Role', 'Admin', 'User', 'Guest']
    }
    df_messy = pd.DataFrame(messy_data)
    
    print("--- DataFrame with headers stuck in row 0 ---")
    print(df_messy, "\n")
    
    # Elevate the first row (index 0) to be the columns
    df_messy.columns = df_messy.iloc[0]
    
    # Clip off that first row by slicing from row 1 onward
    df_clean = df_messy[1:].reset_index(drop=True)
    
    print("--- DataFrame after elevating headers and clipping the first row ---")
    print(df_clean, "\n")


if __name__ == '__main__':
    showcase_pandas_clipping()