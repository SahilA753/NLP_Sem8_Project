import pandas as pd

def append_with_updated_dialogue_id(csv_file_path, output_file_path=None):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Find the maximum Dialogue_ID
    max_dialogue_id = df['Dialogue_ID'].max()
    
    # Create a copy of the dataframe
    new_rows = df.copy()
    
    # Update the Dialogue_ID in the new rows
    new_rows['Dialogue_ID'] = new_rows['Dialogue_ID'] + max_dialogue_id
    
    # Update Sr No. to continue from the last value
    last_sr_no = df['Sr No.'].max()
    new_rows['Sr No.'] = range(last_sr_no + 1, last_sr_no + 1 + len(new_rows))
    
    # Concatenate the original dataframe with the new rows
    result_df = pd.concat([df, new_rows], ignore_index=False)
    
    # Save to a new file if specified
    if output_file_path:
        result_df.to_csv(output_file_path, index=False)
    
    return result_df

result_df = append_with_updated_dialogue_id('final.csv', output_file_path='makkari.csv')