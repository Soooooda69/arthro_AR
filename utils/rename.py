import os
import sys
import time

def rename(dir):
    # Set the path to the directory containing your images
    directory_path = dir + '/images/'
    # Get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # Sort the list of files to ensure they are in the correct order
    file_list.sort()

    # Initialize a counter for the new filenames
    counter = 1

    # Iterate through the files and rename them
    for filename in file_list:
        # Construct the new filename with leading zeros
        new_filename = f"{counter:09d}.jpg"
        
        # Create the full path for the source and destination files
        source_path = os.path.join(directory_path, filename)
        destination_path = os.path.join(directory_path, new_filename)
        
        # Rename the file
        os.rename(source_path, destination_path)
        
        # Increment the counter
        counter += 1
    print("All files have been renamed.")

def fake_timestamp(file_folder,fps):
    # Define the frame rate (frames per second)
    files = os.listdir(file_folder+'/images/')
    # Define the number of images or frames
    num_frames = len(files)  # Change this to the number of frames you have

    timestamps = [int(time.time() * 1e9) + int(1e9 / fps) * i for i in range(num_frames)]

    with open(file_folder +'/timestamp.txt', 'w') as save_file:
        
        # write the generated timestamps
        for frame_index, timestamp in enumerate(timestamps):
            
            # print(f"Frame {frame_index + 1}: Timestamp = {timestamp} seconds")
            save_file.write(str(timestamp)+'\n')
    save_file.close()
    print(f"finished timestamps {file_folder}")
    return timestamps

def change_gt_timestamp(file_folder):
    # Define the path to your input and output text files
    gt_file = file_folder +'/poses_gt.txt'
    out_file = file_folder +'/poses_time_gt.txt'
    # Read the content of the input file
    with open(gt_file, "r") as f:
        lines = f.readlines()

    # Open the output file for writing
    with open(out_file, "w") as f:
        for i, line in enumerate(lines):  # Skip the first 1 header lines
            # new_line = f"{i+1}{' '.join(line.strip().split()[1:])}\n"
            parts = line.strip().split()
            
            new_line = f"{i} {parts[1]} {' '.join(parts[2:])}\n"
            # new_line = f"{parts[0][:-16]} {parts[1]} {' '.join(parts[2:])}\n"
            f.write(new_line)

    print("File has been updated.")
if __name__ == "__main__":
    # sub_dirs = os.listdir(sys.argv[1])
    # for dir in sub_dirs:
    #     if dir == '.DS_Store':
    #         continue
    #     dir = os.path.join(sys.argv[1], dir)
    #     print(dir)
    #     timestamps = fake_timestamp(dir)
    #     rename(timestamps, dir)
    #     change_gt_timestamp(timestamps, dir)
    dir = sys.argv[1]
    timestamps = fake_timestamp(dir, fps=25)
    # rename(dir)
    # change_gt_timestamp(dir)