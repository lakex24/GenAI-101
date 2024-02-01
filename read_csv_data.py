# NBA player data from: https://www.kaggle.com/datasets/justinas/nba-players-data?resource=download
# Actually only contains 2551 unique values
import csv

# Replace 'your_file.csv' with the path to your actual CSV file
# csv_file_path = 'data/nba_players/all_seasons.csv'

def extract_nba_player_names_heights(csv_file_path):
        
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)

            # Extracting the header (column names)
            headers = next(reader)
            # print("Column Names:", headers)

            # Reading and printing each row of data

            # print the all rows in the csv
            player_heights = {}

            for row in reader:
                player_name = row[1]  # assuming we dont have identical names ...
                player_height = row[4]

                # Debug to see duplicated names ...
                if player_name in player_heights.keys():
                    # print('exists: ', player_name,
                    #       'height', player_heights[player_name],
                    #       'new_height', player_height)
                    continue
                    
                player_heights[player_name] = float(player_height)
            
            # Print each row values with corresponding header name
            # for row in reader:
            #     for header, value in zip(headers, row):
            #         print(f"{header}: {value}")
            #     print("\n")  # Newline for readability between rows

    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return player_heights
