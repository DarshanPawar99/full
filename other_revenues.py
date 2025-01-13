import pandas as pd
import streamlit as st
import logging
from threading import Lock
import os



# Lock for concurrency handling
lock = Lock()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
def pivot_and_average_prices(df_filtered):
    # Replace blank values with 'N/A'
    df_filtered = df_filtered.fillna('N/A')
    
    # Create the pivot table
    pivot_df = df_filtered.pivot_table(
        index=['site name', 'vendor', 'session', 'meal type', 'order type','buying amt ai', 'selling amount','remarks'],
        aggfunc='size'
    ).reset_index(name='days')
    
    return pivot_df


def find_mismatches(df_filtered):
    mismatched_data = []
    for index, row in df_filtered.iterrows():
        try:
            calculated_buying_price = safe_get_value(row, 'buying price ai') / safe_get_value(row, 'gst')
            check_mismatch(row, index, 'buying price', calculated_buying_price, mismatched_data)

            calculated_buying_amt = safe_get_value(row, 'buying price ai') * safe_get_value(row, 'buying pax')
            check_mismatch(row, index, 'buying amt ai', calculated_buying_amt, mismatched_data)
            
            calculated_selling_amount = safe_get_value(row, 'selling pax') * safe_get_value(row, 'selling price')
            check_mismatch(row, index, 'selling amount', calculated_selling_amount, mismatched_data)
            
            calculated_commission = safe_get_value(row, 'selling amount') - safe_get_value(row, 'buying amt ai') 
            check_mismatch(row, index, 'commission', calculated_commission, mismatched_data)
        except Exception as e:
            logging.error(f"Error processing row {index + 3}: {e}")

    return mismatched_data


def analysis_data(analysis_df, months_to_include):

    analysis_df['total_gmv'] = analysis_df['selling amount']
    analysis_df['total_buying'] = analysis_df['buying amt ai']

    # Group data by month and calculate sums
    metrics = ['total_buying', 'total_gmv']
    grouped_df = analysis_df.groupby('month')[metrics].sum().reset_index()


    # Pivot the grouped data for summary
    summary_df = grouped_df.set_index('month').T
    summary_df.columns.name = None  # Remove the name of columns

    # Reorder columns to match the order of the months to display
    summary_df = summary_df[months_to_include]

    # Calculate percentage change for specific rows only (up to 'net_revenue')
    if len(summary_df.columns) > 1:
        latest_month, previous_month = summary_df.columns[-1], summary_df.columns[-2]

        # Define rows for which the change percentage should be calculated
        rows_to_calculate = ['total_buying', 'total_gmv']

        # Calculate the percentage change and round to 1 decimal place
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
    support_cost_data = df_filtered[df_filtered['order type'].isin(['support cost', 'transportation'])]
    sum_buying_support_cost = support_cost_data['buying amt ai'].sum()
    sum_selling_support_cost = support_cost_data['selling amount'].sum()

    management_fee_data = df_filtered[df_filtered['order type'].isin(['management fee'])]
    sum_buying_management_fee = management_fee_data['buying amt ai'].sum()
    sum_selling_management_fee = management_fee_data['selling amount'].sum()

    biodegradable_cost_data = df_filtered[df_filtered['order type'].isin(['biodegradable'])]
    sum_buying_biodegradable_cost = biodegradable_cost_data['buying amt ai'].sum()
    sum_selling_biodegradable_cost = biodegradable_cost_data['selling amount'].sum()

    manpower_cost_data = df_filtered[df_filtered['order type'].isin(['salary', 'manpower cost'])]
    sum_buying_manpower_cost = manpower_cost_data['buying amt ai'].sum()
    sum_selling_manpower_cost = manpower_cost_data['selling amount'].sum()
    
    tech_cost_data = df_filtered[df_filtered['order type'].isin(['tech fee only(tech site)', 'tech support(fullstack)'])]
    sum_buying_tech_cost = tech_cost_data['buying amt ai'].sum()
    sum_selling_tech_cost = tech_cost_data['selling amount'].sum()
    
    buying_data = {
        'Category': ['Support Cost', 'Management Fee', 'Biodegradable', 'Manpower', 'Tech Support'],
        'Amount': [sum_buying_support_cost, sum_buying_management_fee, sum_buying_biodegradable_cost, sum_buying_manpower_cost, sum_buying_tech_cost]
    }

    selling_data = {
        'Category': ['Support Cost', 'Management Fee', 'Biodegradable', 'Manpower', 'Tech Support'],
        'Amount': [sum_selling_support_cost, sum_selling_management_fee, sum_selling_biodegradable_cost, sum_selling_manpower_cost, sum_selling_tech_cost]
    }

    buying_df = pd.DataFrame(buying_data)
    selling_df = pd.DataFrame(selling_data)

    return buying_df, selling_df

def find_buying_value_issues(df_filtered):
    buying_value_issues = []
    for index, row in df_filtered.iterrows():
        if (safe_get_value(row, 'buying pax') > 0 or safe_get_value(row, 'buying price ai') > 0) and safe_get_value(row, 'buying amt ai') == 0:
            buying_value_issues.append({
                'Row': index + 3,
                'Site Name': row['site name'],
                'Session': row['session'],
                'Mealtype': row['meal type'],
                'Ordertype': row['order type'],
                'Buying Pax': row['buying pax'],
                'Buying Price AI': row['buying price ai'],
                'Buying Amount AI': row['buying amt ai']
            })
    return buying_value_issues

def find_selling_value_issues(df_filtered):
    selling_value_issues = []
    for index, row in df_filtered.iterrows():
        if (safe_get_value(row, 'selling pax') > 0 or safe_get_value(row, 'selling price') > 0) and safe_get_value(row, 'selling amount') == 0:
            selling_value_issues.append({
                'Row': index + 3,
                'Site Name': row['site name'],
                'Session': row['session'],
                'Mealtype': row['meal type'],
                'Ordertype': row['order type'],
                'Selling Pax': row['selling pax'],
                'Selling Price': row['selling price'],
                'Selling Amount': row['selling amount']
            })
    return selling_value_issues

def find_double_entry_issues(df_filtered):
    double_entry = []
    for index, row in df_filtered.iterrows():
        if (safe_get_value(row, 'buying amt ai') > 0 and safe_get_value(row, 'selling amount') > 0):
            double_entry.append({
                'Row': index + 3,
                'Site Name': row['site name'],
                'Session': row['session'],
                'Mealtype': row['meal type'],
                'Ordertype': row['order type'],
                'Buying Amount AI': row['buying amt ai'],
                'Selling Amount': row['selling amount']
            })
    return double_entry


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

def display_dataframes(pivot_df, mismatched_data, buying_df, selling_df, buying_value_issues, selling_value_issues, double_entry, summary_df):
    st.subheader("Other Revenues Data")
    st.table(format_dataframe(pivot_df))
    st.markdown("---")

    if mismatched_data:
        mismatched_df = pd.DataFrame(mismatched_data)
        st.write("<span style='color:red'>Mismatched Data:heavy_exclamation_mark:</span>", unsafe_allow_html=True)
        st.table(format_dataframe(mismatched_df))
        st.markdown("---")
    else:
        st.write("<span style='color:green'>No mismatch found.</span> :white_check_mark:", unsafe_allow_html=True)
        st.markdown("---")

    if buying_value_issues:
        buying_value_issues_df = pd.DataFrame(buying_value_issues)
        st.write("<span style='color:red'>Buying Value Issues</span> :heavy_exclamation_mark:", unsafe_allow_html=True)
        st.dataframe(format_dataframe(buying_value_issues_df))
        st.markdown("---")
    else:
        st.write("<span style='color:green'>No buying value issues found.</span> :white_check_mark:", unsafe_allow_html=True)
        st.markdown("---")

    if selling_value_issues:
        selling_value_issues_df = pd.DataFrame(selling_value_issues)
        st.write("<span style='color:red'>Selling Value Issues</span> :heavy_exclamation_mark:", unsafe_allow_html=True)
        st.dataframe(format_dataframe(selling_value_issues_df))
        st.markdown("---")
    else:
        st.write("<span style='color:green'>No selling value issues found.</span> :white_check_mark:", unsafe_allow_html=True)
        st.markdown("---")

    if double_entry:
        double_entry_df = pd.DataFrame(double_entry)
        st.write("<span style='color:red'>Buying and Selling in single Row.</span> :heavy_exclamation_mark:", unsafe_allow_html=True)
        st.dataframe(format_dataframe(double_entry_df))
        st.markdown("---")
    else:
        st.write("<span style='color:green'>NO Buying and Selling in single Row.</span> :white_check_mark:", unsafe_allow_html=True)
        st.markdown("---")

    st.subheader("Aggregated Values")
    st.write("Buying Support")
    st.table(format_dataframe(buying_df))
    st.write("Selling Support")
    st.table(format_dataframe(selling_df))
    on = st.toggle("View Analysis")
    if on:
        st.table(format_all_columns_with_color(summary_df))
    st.markdown("---")

def other_revenues(df_selected_month, analysis_df, months_to_include):
    
    # Sidebar multiselect for site name
    selected_sites = st.sidebar.multiselect("Select Site Name(s)", df_selected_month['site name'].unique())

    # Filter dataframe by selected sites
    if selected_sites:
        df_filtered = df_selected_month[df_selected_month['site name'].isin(selected_sites)]
    else:
        df_filtered = df_selected_month  # In case no site is selected, show all data

    pivot_df = pivot_and_average_prices(df_filtered)
    mismatched_data = find_mismatches(df_filtered)
    buying_df, selling_df = calculate_aggregated_values(df_filtered)
    buying_value_issues = find_buying_value_issues(df_filtered)
    selling_value_issues = find_selling_value_issues(df_filtered)
    double_entry = find_double_entry_issues(df_filtered)

    if selected_sites:
        analysis_df = analysis_df[analysis_df['site name'].isin(selected_sites)]
    else:
        analysis_df = analysis_df  # Handle no selection scenario

    # Perform analysis on the last three months data
    summary_df = analysis_data(analysis_df, months_to_include)  # Pass months_to_include here

    # Display all relevant dataframes and results
    display_dataframes(pivot_df, mismatched_data, buying_df, selling_df, buying_value_issues, selling_value_issues, double_entry, summary_df)

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
            

            # Apply the filtering logic within the current group
            vendor_payable = group[group['order type'].isin(['support cost','transportation','management fee', 'biodegradable','salary','manpower cost', 'tech fee only(tech site)','tech support(fullstack)', 'dishwash support'])]
            management = group[group['order type'].isin(['management fee'])]
            tech_support = group[group['order type'].isin(['support cost','transportation', 'biodegradable','salary','manpower cost', 'tech fee only(tech site)','tech support(fullstack)', 'dishwash support'])]

            pnl_data = pd.DataFrame({
            
                'vendor_support_cost': [vendor_payable['buying amt ai'].sum()],
                'management_fee': [management['selling amount'].sum()],
                'tech_support_fee': [tech_support['selling amount'].sum()]
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
            'vendor_support_cost': 'vendor_support_cost',
                'management_fee': 'management_fee',
                'tech_support_fee': 'tech_support_fee'
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
                       ['vendor_support_cost', 'management_fee', 'tech_support_fee']] = pnl_data[['vendor_support_cost', 'management_fee', 'tech_support_fee']].values

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
def clear_pnl_data(p_and_l_file_path, month, identifier=None):
    """
    Clears the P&L data for the specified month and matching identifier by setting relevant fields to zero.
    """
    try:
        # Load the P&L data
        pnl_df = load_pnl_excel_data(p_and_l_file_path)
        if pnl_df is None:
            return

        # Ensure that the 'month' column exists
        if 'month' not in pnl_df.columns:
            st.error("The 'month' column is missing from the P&L data.")
            return

        # Create the 'identifier' column for matching
        pnl_df['identifier'] = pnl_df['review id'].combine_first(pnl_df['cost centre']) if 'review id' in pnl_df.columns else pnl_df['cost centre']

        # Define the columns to clear
        columns_to_clear = [
            'vendor_support_cost', 'management_fee', 'tech_support_fee']

        # Ensure an identifier is passed
        if identifier is None:
            st.error("No identifier provided for clearing P&L data.")
            return

        # Clear only rows that match both identifier and month
        pnl_df.loc[(pnl_df['month'] == month) & (pnl_df['identifier'] == identifier), columns_to_clear] = 0

        # Save the updated P&L data
        save_pnl_excel_data(pnl_df, p_and_l_file_path)
        st.success(f"P&L data cleared successfully for identifier: {identifier} and month: {month}")
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
        'cost centre': 'cost centre', 
        'site name': 'site name',
        'vendor code': 'vendor code',
        'vendor': 'vendor',
        'session': 'session',
        'meal type': 'meal type',
        'order type': 'order type',
        'client mg/pre order': 'client mg/pre order',
        'ordered pax/vendor mg': 'ordered pax/vendor mg',
        'actual consumption': 'actual consumption',
        'buying pax': 'buying pax',
        'buying price': 'buying price',
        'buying price ai': 'buying price ai',
        'buying transportation': 'buying transportation',
        'buying amt ai': 'buying amt ai',
        'selling pax': 'selling pax',
        'selling price': 'selling price',
        'selling transportation': 'selling transportation',
        'selling amount': 'selling amount',
        'penalty on vendor': 'penalty on vendor',
        'penalty on smartq': 'penalty on smartq',
        'commission': 'commission',
        'amount': 'amount'
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
