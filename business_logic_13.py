import pandas as pd
import streamlit as st
import logging
from threading import Lock
import os

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

lock = Lock()

def safe_get_value(row, col):
    return row[col] if col in row and pd.notna(row[col]) else 0

def check_mismatch(row, index, column_name, expected_value, mismatched_data):
    actual_value = safe_get_value(row, column_name)
    if actual_value != expected_value:
        mismatched_data.append({
            'Row': index + 3,
            'Date': row['date'],
            'Column': column_name,
            'Expected': expected_value,
            'Actual': actual_value
        })

def find_mismatches(df_filtered):
    mismatched_data = []
    for index, row in df_filtered.iterrows():
        try:
            
            calculated_selling_amt = safe_get_value(row, 'rate') * safe_get_value(row, 'quantity')                                 
            check_mismatch(row, index, 'selling amount', calculated_selling_amt, mismatched_data)

            calculated_buying_amt = safe_get_value(row, 'buying amt ai') * 100 / 105            
            check_mismatch(row, index, 'buying amount', calculated_buying_amt, mismatched_data)

            calculated_buying_amount_ai = safe_get_value(row, 'selling amount') - safe_get_value(row, 'commission')
            check_mismatch(row, index, 'buying amt ai', calculated_buying_amount_ai, mismatched_data)
            
            calculated_commission = safe_get_value(row, 'selling amount') * 0.1
            check_mismatch(row, index, 'commission', calculated_commission, mismatched_data)
        except Exception as e:
            logging.error(f"Error processing row {index + 3}: {e}")

    return mismatched_data


def analysis_data(analysis_df, months_to_include):

    # Fill missing values with 0 for specified columns
    columns_to_fill = ['selling amount', 'commission', 'buying amt ai']
    analysis_df[columns_to_fill] = analysis_df[columns_to_fill].fillna(0)

    # Calculate additional columns
    analysis_df['buying_amount'] = analysis_df['buying amt ai']
    analysis_df['total_gmv'] = analysis_df['selling amount']

    # Group data by month and calculate sums
    metrics = ['buying_amount', 'total_gmv', 'commission']
    grouped_df = analysis_df.groupby('month')[metrics].sum().reset_index()


# Pivot the grouped data for summary
    summary_df = grouped_df.set_index('month').T
    summary_df.columns.name = None  # Remove the name of columns

    # Reorder columns based on months to include
    summary_df = summary_df[months_to_include]

    # Calculate percentage change for specific rows only (up to 'net_revenue')
    if len(summary_df.columns) > 1:
        latest_month, previous_month = summary_df.columns[-1], summary_df.columns[-2]

        # Define rows for which the change percentage should be calculated
        rows_to_calculate = ['regular_gmv', 'event_gmv', 'commission', 'karbon', 'paxs_difference', 'pd_revenue', 'net_revenue', 'total_gmv']

        # Calculate the percentage change and handle cases where previous_month is 0 to avoid division by zero
        change_percentage = ((summary_df[latest_month] - summary_df[previous_month]) / summary_df[previous_month]) * 100

        # Replace infinite values (when previous_month is 0) with 100%, and NaN values with 0
        change_percentage = change_percentage.replace([float('inf'), -float('inf')], 100).fillna(0)

        # Round to 1 decimal place
        summary_df['change_percentage'] = change_percentage.round(1)

        # Ensure only specific rows have the change calculated
        summary_df.loc[~summary_df.index.isin(rows_to_calculate), 'change_percentage'] = None
    else:
        # If only one month is present, set 'change_percentage' as 'NA'
        summary_df['change_percentage'] = 'NA'

    return summary_df


def calculate_aggregated_values(df_filtered):
    sum_buying_amt_ai_regular= df_filtered['buying amt ai'].sum()
    sum_selling_amt_regular = df_filtered['selling amount'].sum()
    sum_commission = df_filtered['commission'].sum()
    
    aggregated_data = {
        'Buying Amt AI (Regular)': sum_buying_amt_ai_regular,
        'Selling Amt (Regular)': sum_selling_amt_regular,
        'Commission': sum_commission,
    }

    return aggregated_data

def format_dataframe(df):
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df[column] = df[column].map(lambda x: f"{x:.1f}")
    return df

def fmt_inr(df):
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df[column] = df[column].map(lambda x: f"{int(x):,}")
    return df

def format_all_columns_with_color(df):
    # Format numerical columns to one decimal place
    for column in df.select_dtypes(include=['float', 'int']).columns:
        if column != 'change_percentage':  # Avoid formatting 'change_percentage' twice
            df[column] = df[column].map(lambda x: f"{x:.1f}")

    # Format 'change_percentage' column to display with a percentage symbol
    if 'change_percentage' in df.columns:
        df['change_percentage'] = df['change_percentage'].map(lambda x: f"{x:.1f}%")

    # Apply conditional formatting to the 'change_percentage' column
    def highlight_change_percentage(val):
        try:
            color = 'red' if float(val.strip('%')) < 0 else 'green'
        except ValueError:
            color = 'black'  # Default color if conversion fails
        return f'color: {color}'

    # Apply the styling to the DataFrame
    styled_df = df.style.applymap(highlight_change_percentage, subset=['change_percentage'])
    
    return styled_df


def display_dataframes(mismatched_data, aggregated_data, summary_df):
    st.write("sodexio sales")
    st.markdown("---")

    if mismatched_data:
        mismatched_df = pd.DataFrame(mismatched_data)
        st.write("<span style='color:red'>Mismatched Data:heavy_exclamation_mark:</span>", unsafe_allow_html=True)
        st.table(format_dataframe(mismatched_df))
    else:
        st.write("<span style='color:green'>No mismatch found.</span> :white_check_mark:", unsafe_allow_html=True)
    st.markdown("---")

    aggregated_df = pd.DataFrame(list(aggregated_data.items()), columns=['Parameter', 'Value'])
    st.subheader("Aggregated Values")
    st.table(fmt_inr(aggregated_df))

    on = st.toggle("View Analysis")
    if on:
        st.table(format_all_columns_with_color(summary_df))
    st.markdown("---")


def business_logic_13(df_selected_month, analysis_df, months_to_include):
    # Perform business logic on selected month and last three months data
    mismatched_data = find_mismatches(df_selected_month)
    aggregated_data = calculate_aggregated_values(df_selected_month)

    # Perform analysis on the last three months data
    summary_df = analysis_data(analysis_df, months_to_include)  # Pass months_to_include here

    # Display all relevant dataframes and results
    display_dataframes(mismatched_data, aggregated_data, summary_df)

#-----------------------------------------------------auto Pnl------------------------------------------------------------------
def load_business_logic(df_filtered):
    try:
        # Clean the data by stripping spaces and converting strings to lowercase
        df_filtered = df_filtered.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

        # Raise an error if the filtered dataframe is empty
        if df_filtered.empty:
            raise ValueError("No data available for the selected month.")

        # Group by 'cost centre' to process each group separately
        grouped = df_filtered.groupby('cost centre')

        pnl_data_list = []  # Store individual P&L data

        # Iterate through each group by cost centre
        for cost_centre, group in grouped:
            # Process each group
            full_data = group

            # Apply the filtering logic within the current group
            no_of_days = group[(group['quantity'] > 0)]

            pnl_data = pd.DataFrame({
                'days': [no_of_days['date'].nunique()],
                'regular buying amount': [full_data['buying amt ai'].sum()],
                'regular selling amount': [full_data['selling amount'].sum()]
            })

            pnl_data_list.append(pnl_data)  # Append the result for each cost centre

        if pnl_data_list:
            final_pnl_data = pd.concat(pnl_data_list, ignore_index=True)
            return format_pnl_dataframe(final_pnl_data)
        else:
            st.warning("No valid data to process.")
            return None

    except Exception as e:
        st.error(f"Error loading Auto P&L logic data: {e}")
        logging.error(f"Error loading Auto P&L logic data: {e}")
        return None

def format_pnl_dataframe(df_filtered):
    for column in df_filtered.select_dtypes(include=['float', 'int']).columns:
        df_filtered[column] = df_filtered[column].map(lambda x: f"{x:.1f}")
    return df_filtered

def load_pnl_excel_data(p_and_l_file_path):
    try:
        pnl_df = pd.read_excel(p_and_l_file_path, header=0)
        pnl_df = pnl_df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        return pnl_df
    except FileNotFoundError:
        st.error("P&L file not found. Please check the file path.")
        logging.error("P&L file not found.")
        return None

def save_pnl_excel_data(df_filtered, p_and_l_file_path):
    try:
        with pd.ExcelWriter(p_and_l_file_path, mode='w', engine='openpyxl') as writer:
            df_filtered.to_excel(writer, index=False)
        if os.path.exists(p_and_l_file_path):
            os.chmod(p_and_l_file_path, 0o666)
        else:
            st.error(f"File not found: {p_and_l_file_path}")
    except PermissionError:
        st.error("Permission denied: You don't have the necessary permissions to change the permissions of this file.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error saving Excel data: {e}")

def process_data(pnl_df, pnl_data):
    try:
        # Mapping based on the column names
        pnl_mapping = {
            'days': 'days',
            'regular buying amount': 'regular buying amount',
            'regular selling amount': 'regular selling amount',
            }

        # Check columns exist before mapping
        for source_col, target_col in pnl_mapping.items():
            if source_col in pnl_data.columns and target_col in pnl_df.columns:
                pnl_df[target_col] = pnl_data[source_col]
            else:
                logging.warning(f"Column '{source_col}' in pnl_data or '{target_col}' in pnl_df not found.")

        logging.info("Data processed and mapped successfully.")
        return pnl_df

    except Exception as e:
        st.error(f"Error processing data: {e}")
        logging.error(f"Error processing data: {e}")
        return pnl_df

def punch_pnl_data_to_blank_row(df_filtered, month, p_and_l_file_path): 
    try:
        pnl_df = load_pnl_excel_data(p_and_l_file_path)
        if pnl_df is None:
            return

        if 'month' not in df_filtered.columns or 'month' not in pnl_df.columns:
            st.error("The 'month' column is missing from one of the dataframes.")
            return

        # Create 'identifier' column, prioritize 'review id' completely if it exists, else use 'cost centre'
        if 'review id' in df_filtered.columns and not df_filtered['review id'].isna().all():
            df_filtered['identifier'] = df_filtered['review id']
        else:
            df_filtered['identifier'] = df_filtered['cost centre']
            
        if 'review id' in pnl_df.columns and not pnl_df['review id'].isna().all():
            pnl_df['identifier'] = pnl_df['review id']
        else:
            pnl_df['identifier'] = pnl_df['cost centre']

        # Track if any record was successfully punched
        record_punched = False

        # Iterate through each group by 'identifier' in the filtered data
        for key_value, group in df_filtered.groupby('identifier'):
            st.write(f"Processing identifier: {key_value}")

            pnl_data = load_business_logic(group)
            if pnl_data is None:
                continue

            # Get matching rows by identifier and month
            pnl_df_matching = pnl_df[(pnl_df['identifier'] == key_value) & (pnl_df['month'] == month)]

            if pnl_df_matching.empty:
                st.error(f"No matching records found for identifier: {key_value} and month: {month}")
                continue

            # Ensure the number of rows matches between pnl_data and pnl_df_matching
            if len(pnl_data) == 1 and len(pnl_df_matching) > 1:
                st.warning(f"More than one matching record found for identifier: {key_value} and month: {month}. Aggregating pnl_data.")
                pnl_data = pnl_data.iloc[0]  # Take the first row in pnl_data if there's only one

            # Update specific columns only
            pnl_df.loc[(pnl_df['identifier'] == key_value) & (pnl_df['month'] == month), 
                       ['days','regular buying amount', 'regular selling amount']] = pnl_data[['days' ,'regular buying amount', 'regular selling amount']].values

            # If matching records found, set the flag to True
            record_punched = True

        # Save the P&L Excel data if any record was punched
        if record_punched:
            save_pnl_excel_data(pnl_df, p_and_l_file_path)
            st.success(f"P&L Data punched successfully for month: {month}")
        else:
            st.warning(f"No records were punched for month: {month}")

    except Exception as e:
        st.error(f"Error processing P&L data: {e}")
        logging.error(f"Error processing P&L data: {e}")
def clear_pnl_data(p_and_l_file_path, month):
    """
    Clears the P&L data for the specified month by setting relevant fields to zero.
    """
    try:
        pnl_df = load_pnl_excel_data(p_and_l_file_path)
        if pnl_df is None:
            return

        # Ensure that the 'month' column exists
        if 'month' not in pnl_df.columns:
            st.error("The 'month' column is missing from the P&L data.")
            return

        # Create the 'identifier' column for matching
        pnl_df['identifier'] = pnl_df['review id'].combine_first(pnl_df['cost centre']) if 'review id' in pnl_df.columns else pnl_df['cost centre']

        # Filter the data for the selected month
        pnl_df_matching = pnl_df[pnl_df['month'] == month]

        if pnl_df_matching.empty:
            st.warning(f"No data found for the month: {month}")
            return

        # Define the columns to clear
        columns_to_clear = [
            'days', 'regular buying amount', 
            'regular selling amount', 'event buying amount', 
            'event selling amount', 'penalty on vendor', 'penalty on smartq', 'sams'
        ]

        # Clear the relevant columns for the selected month
        pnl_df.loc[pnl_df['month'] == month, columns_to_clear] = 0

        save_pnl_excel_data(pnl_df, p_and_l_file_path)
        st.success(f"P&L data cleared successfully for the month: {month}")
        st.dataframe(pnl_df)

    except Exception as e:
        st.error(f"Error clearing P&L data: {e}")
        logging.error(f"Error clearing P&L data: {e}")


        
#-------------------------------------------------------Auto Dump--------------------------------------------------

def dump_data(df_filtered, month, dump_file_path):
    dump_mapping = {
       'date': 'date',
        'month': 'month',
        'day': 'day',
        'site name': 'site name',
        'order type': 'order type',
        'buying amt ai': 'buying amt ai',
        'selling amount': 'selling amount',
        'commission': 'commission'
    }
    
    try:
        dump_df = load_dump_data(dump_file_path)
        if dump_df is None:
            return

        if not dump_df.empty:
            last_row = dump_df.iloc[-1]
            last_row_df = last_row.to_frame().T
            last_row_df.insert(0, 'row number', len(dump_df) + 1)
            st.write("Last updated row before current dump:")
            st.dataframe(last_row_df)

        mapped_df = pd.DataFrame()
        for dump_col, df_col in dump_mapping.items():
            if df_col in df_filtered.columns and dump_col in dump_df.columns:
                mapped_df[dump_col] = df_filtered[df_col]

        if 'selling management fee' in df_filtered.columns:
            # Group by 'site name' and calculate the sum of 'selling management fee' for each group
            grouped = df_filtered.groupby('site name')

            for site_name, group in grouped:
                selling_sum = group['selling management fee'].sum()
                
                # Create a new row for each group with the aggregated data
                new_row = pd.DataFrame({
                    'month': [month],
                    'site name': [site_name],
                    'order type': ['management fee'],
                    'selling pax': 1,
                    'selling price': [selling_sum],
                    'selling amount': [selling_sum]
                })
                
            # Append the new row to the dump_df DataFrame
            dump_df = pd.concat([dump_df, new_row], ignore_index=True)

        updated_df = pd.concat([dump_df, mapped_df], ignore_index=True)
        save_updated_dump_data(updated_df, dump_file_path)
        logging.info("Filtered data appended to the dump file successfully.")
        st.success("Filtered data appended to the dump file successfully.")

    except Exception as e:
        st.error(f"Error dumping data: {e}")
        logging.error(f"Error dumping data: {e}")

def load_dump_data(dump_file_path):
    try:
        dump_df = pd.read_excel(dump_file_path, header=0)
        dump_df.columns = dump_df.columns.str.lower().str.strip()
        return dump_df
    except FileNotFoundError:
        st.write("Output file not found. Please check the file path.")
        return None

def save_updated_dump_data(df, dump_file_path):
    try:
        df.to_excel(dump_file_path, index=False)
        if os.path.exists(dump_file_path):
            os.chmod(dump_file_path, 0o666)
        else:
            st.write("File not found:", dump_file_path)
    except PermissionError:
        st.write("Permission denied: You don't have the necessary permissions to change the permissions of this file.")







#--------------------------
        no_of_days = df[(df['quantity'] > 0)]

        # Group the data by 'identifier' and 'month'
        grouped_data = df.groupby(['identifier', 'month'])
        no_of_days_grouped = no_of_days.groupby(['identifier', 'month'])

        # Create the P&L DataFrame
        pnl_data = pd.DataFrame({
            'days': no_of_days_grouped['date'].nunique(),
            'regular buying amount': grouped_data['buying amt ai'].sum(),
            'regular selling amount': grouped_data['selling amount'].sum(),
        
        }).reset_index()

        return format_dataframe(pnl_data)

    except Exception as e:
        st.error(f"Error loading business logic data: {e}")
        return None

def format_dataframe(df):
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df[column] = df[column].map(lambda x: f"{x:.1f}")
    return df

def load_pnl_data(p_and_l_file_path):
    try:
        pnl_df = pd.read_excel(p_and_l_file_path, header=0)
        pnl_df = pnl_df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        return pnl_df
    except FileNotFoundError:
        st.error("Output file not found. Please check the file path.")
        return None

def save_updated_data(df, p_and_l_file_path):
    try:
        with lock:
            df.to_excel(p_and_l_file_path, index=False)
            if os.path.exists(p_and_l_file_path):
                os.chmod(p_and_l_file_path, 0o666)
            else:
                st.error(f"File not found: {p_and_l_file_path}")
    except PermissionError:
        st.error("Permission denied: You don't have the necessary permissions to change the permissions of this file.")

def process_data(pnl_df, pnl_data):
    try:
        pnl_mapping = {
            'days': 'days',
            'regular buying amount': 'regular buying',
            'regular selling amount': 'selling -gmv',
        }

        # Rename columns according to the mapping
        pnl_data = pnl_data.rename(columns=pnl_mapping)

        # Merge data
        pnl_merged_df = pd.merge(pnl_df, pnl_data, left_on=['cost centre', 'month'], right_on=['identifier', 'month'], how='left', suffixes=('', '_new'))

        # Check for unmatched data
        unmatched = pnl_data[~pnl_data.set_index(['identifier', 'month']).index.isin(pnl_merged_df.set_index(['cost centre', 'month']).index)]
        if not unmatched.empty:
            st.error("Could not find a match for cost centre & month.")
            return None, None

        # Update the original data with the new values
        for col in pnl_mapping.values():
            pnl_merged_df[col] = pnl_merged_df[f"{col}_new"].combine_first(pnl_merged_df[col])
        pnl_merged_df.drop(columns=[f"{col}_new" for col in pnl_mapping.values()], inplace=True)

        # Identify rows with updated data
        updated_rows = pnl_merged_df[(pnl_merged_df[list(pnl_mapping.values())] != pnl_df[list(pnl_mapping.values())]).any(axis=1)].copy()

        # Prepare the updated rows DataFrame
        updated_rows = updated_rows[['cost centre', 'month', 'site name'] + list(pnl_mapping.values())]
        updated_rows.columns = ['Cost Centre', 'Month', 'Site Name'] + [f"Updated {col.title().replace('_', ' ')}" for col in pnl_mapping.values()]
        updated_rows.dropna(subset=[f"Updated {col.title().replace('_', ' ')}" for col in pnl_mapping.values()], how='all', inplace=True)

        return pnl_merged_df, updated_rows
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None, None

def update_p_and_l(df, selected_month, p_and_l_file_path):
    pnl_data = load_business_logic(df, selected_month)
    if pnl_data is None:
        return
    pnl_df = load_pnl_data(p_and_l_file_path)
    if pnl_df is None:
        return
    pnl_merged_df, updated_rows = process_data(pnl_df, pnl_data)
    if pnl_merged_df is not None:
        save_updated_data(pnl_merged_df, p_and_l_file_path)
        st.success("Successfully Punched P&L")
        st.dataframe(format_dataframe(updated_rows))

def clear_p_and_l_data(df, selected_month, p_and_l_file_path):
    pnl_data = load_business_logic(df, selected_month)
    if pnl_data is None:
        return
    pnl_df = load_pnl_data(p_and_l_file_path)
    if pnl_df is None:
        return

    try:
        pnl_mapping = {
            'days': 'days',
            'regular buying amount': 'regular buying',
            'regular selling amount': 'selling -gmv',
        }

        # Clear the data by setting the relevant columns to None
        for identifier_val, month in zip(pnl_data['identifier'], pnl_data['month']):
            if not ((pnl_df['cost centre'] == identifier_val) & (pnl_df['month'] == month)).any():
                st.error(f"Could not find a match for cost centre {identifier_val} & month {month}.")
                return
            pnl_df.loc[(pnl_df['cost centre'] == identifier_val) & (pnl_df['month'] == month), pnl_mapping.values()] = None

        save_updated_data(pnl_df, p_and_l_file_path)
        st.write("Data cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing data: {e}")

#-------------------------------------------------------Auto Dump--------------------------------------------------

def dump_data(df_filtered, month, dump_file_path):
    dump_mapping = {
       'date': 'date',
        'month': 'month',
        'day': 'day',
        'site name': 'site name',
        'order type': 'order type',
        'buying amt ai': 'buying amt ai',
        'selling amount': 'selling amount',
        'commission': 'commission'
         }
    
    try:
        dump_df = load_dump_data(dump_file_path)
        if dump_df is None:
            return

        if not dump_df.empty:
            last_row = dump_df.iloc[-1]
            last_row_df = last_row.to_frame().T
            last_row_df.insert(0, 'row number', len(dump_df) + 1)
            st.write("Last updated row before current dump:")
            st.dataframe(last_row_df)

        mapped_df = pd.DataFrame()
        for dump_col, df_col in dump_mapping.items():
            if df_col in df_filtered.columns and dump_col in dump_df.columns:
                mapped_df[dump_col] = df_filtered[df_col]

        if 'selling management' in df_filtered.columns:
            selling_sum = df_filtered['selling management'].sum()
            new_row = pd.DataFrame({
                'month': [month],
                'site name': [df_filtered['site name'].iloc[0] if 'site name' in df_filtered.columns else None],
                'order type': ['management fee'],
                'selling amount': [selling_sum]
            })
            dump_df = pd.concat([dump_df, new_row], ignore_index=True)

        updated_df = pd.concat([dump_df, mapped_df], ignore_index=True)
        save_updated_dump_data(updated_df, dump_file_path)
        logging.info("Filtered data appended to the dump file successfully.")
        st.success("Filtered data appended to the dump file successfully.")

    except Exception as e:
        st.error(f"Error dumping data: {e}")
        logging.error(f"Error dumping data: {e}")

def load_dump_data(dump_file_path):
    try:
        dump_df = pd.read_excel(dump_file_path, header=0)
        dump_df.columns = dump_df.columns.str.lower().str.strip()
        return dump_df
    except FileNotFoundError:
        st.write("Output file not found. Please check the file path.")
        return None

def save_updated_dump_data(df, dump_file_path):
    try:
        df.to_excel(dump_file_path, index=False)
        if os.path.exists(dump_file_path):
            os.chmod(dump_file_path, 0o666)
        else:
            st.write("File not found:", dump_file_path)
    except PermissionError:
        st.write("Permission denied: You don't have the necessary permissions to change the permissions of this file.")
