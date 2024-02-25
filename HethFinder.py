from math import floor
import pandas as pd
import numpy as np
import h5py
import matplotlib.pylab as plt
import os
import librosa
import librosa.display
import tensorflow as tf
from datetime import datetime
import ctypes

#FUNCTIONS

#hethfinder main function
def hethfinder_main():
    params = pd.read_csv("hethfinder_params.csv")
    
    completion_strings = []
    for index, row in params.iterrows():
        completion_strings.append(hethfinder_one_file(row['filename'], row['start_time(s)'], row['end_time(s)']))
        print(f'COMPLETED {index + 1}.')
    
    mbox_string = ''
    for comp_string in completion_strings:
        mbox_string += comp_string + '\n\n'
    Mbox('HethFinder Results', mbox_string, 0x00001000)
    

#creates raven file for one file
def hethfinder_one_file(filename_input, wave_start, wave_end):
    if not os.path.exists(filename_input.split('.')[0] + '.wav'):
        return f'failure: could not find {filename_input}.'
    global time_converter
    time_converter = 0.023219814
    global intro_threshold
    intro_threshold = -50
    global directory
    directory = filename_input + '/'
    global filename
    filename = filename_input
    

    #check that the input data is appropriate
    if isinstance(wave_start, str) and not wave_start.isnumeric():
        return f'FAILURE: start time is negative or not a number for {filename}.'
    if isinstance(wave_end, str) and not wave_end.isnumeric():
        return f'FAILURE: end time is negative or not a number for {filename}.'
    if isinstance(wave_start, float) and np.isnan(wave_start):
        wave_start = 0
    if isinstance(wave_end, float) and np.isnan(wave_end):
        wave_end = 0
    wave_start = int(wave_start)
    wave_end = int(wave_end)
    if wave_end != 0 and wave_start >= wave_end:
        return f'FAILURE: start time is greater than or equal to end time for {filename}.'
    if wave_start < 0:
        return f'FAILURE: start time was negative for {filename}.'
    if wave_end < 0:
        return f'FAILURE: end time was negative for {filename}.'
    
    #create a spectrogram
    data = fourier(filename, wave_start, wave_end)
    if isinstance(data, str):
        return f'failure: {filename}. Start time is greater than duration of the wave file.'
    
    #scan spectrogram with ML model
    all_times, all_scores = song_model_scores(data, filename)
    song_time_list, strength_list = get_song_times_from_all(all_times, all_scores)
    if len(song_time_list) <= 0:
        return f"found 0 heth songs in {filename} between {wave_start} seconds and {f'{str(wave_end)} seconds.' if wave_end > 0 else ' end.'}"
    
    #post-processing
    trimmed_song_time_list = remove_bad_timing_songs(song_time_list, strength_list)
    reinserted_song_times = reinsert_weak_songs(all_times, all_scores, list(trimmed_song_time_list))
    
    #get intro note frequencies and timings
    new_song_times, song_freqs = get_start_times_and_freqs_from_pools_with_displays(data, reinserted_song_times, [], False)
    new_song_times = new_song_times + wave_start
    
    make_raven_from_times_and_freqs(new_song_times, song_freqs, wave_start, wave_end)
    return f"successfully create raven table for {filename} from {str(wave_start)} seconds to {f'{str(wave_end)} seconds.' if wave_end > 0 else ' end.'}"




def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


# LOAD A RECORDING AND FOURIER TRANSFORM IT
def fourier(filename, wave_start, wave_end):
    y, sr = librosa.load(filename + '.wav')
    start_int = int(wave_start * sr)
    end_int = int(wave_end * sr)
    duration = librosa.get_duration(y=y, sr=sr)
    if wave_start > duration:
        return 'fail'
    
    if end_int > 0 and wave_end < duration:
        D = librosa.stft(y[start_int:end_int])
    else:
        D = librosa.stft(y[start_int:])
    D = D[120:743,:]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

#FUNCTIONS USED IN THE H5_TO_ALBUM FUNCTION
def load_variables(array):
    global intro_threshold
    array_median = np.median(array)
    intro_threshold = np.median(array) + np.std(array) * (.75 + (abs(-45 - array_median)/20)) + 5
    print(f'intro threshold: {intro_threshold}')

#creates a post introductory pooled spectrogram
def post_pool(array, column, end_column, rows, columns, cell_height, cell_length):
    row_index = 0
    post_note_array = []
    if column + 70 > columns:
        end_column = columns
    while row_index < rows - cell_height:
        col_index = 0
        row_values = []
        while column + col_index <= end_column - cell_length:
            row_values.append(np.mean(array[row_index:row_index + cell_height,column + col_index:column + col_index + cell_length]))
            col_index += cell_length
        post_note_array.append(row_values)
        row_index += cell_height
    return np.array(post_note_array)

def filter_one_pool_new(pool, range_threshold, range_range, horizontals = False):
    thisPool = pool.copy()

    #look for stacks of 3 cells that are similar to each other and remove them
    indexes = []
    for i in range(len(thisPool)):
        for j in range(len(thisPool[0])):
            if i < range_range:
                bot = 0
                top = range_range * 2
            elif i > len(thisPool) - range_range:
                bot = len(thisPool) - 2 * range_range
                top = len(thisPool)
            else:
                top = i + range_range
                bot = i - range_range
            #vol_range = np.max(thisPool[bot:top,j]) - np.min(thisPool[bot:top,j])
            
            
            if i <= len(thisPool) - range_range and (abs(thisPool[i, j] - np.min(thisPool[i:top+1, j])) < range_threshold):
                indexes.append([i, j])
            if i >= range_range and (abs(thisPool[i, j] - np.min(thisPool[bot:i, j])) < range_threshold):
                indexes.append([i, j])
            
            # if vol_range < range_threshold:
            #     indexes.append([i, j])
    for index in indexes:
        thisPool[index[0], index[1]] = -80

    #remove anything quieter than a given amount
    for i in range(len(thisPool)):
        for j in range(len(thisPool[0])):
            if thisPool[i,j] < -65:
                thisPool[i,j] = -80
    
    #remove any full horizontal lines                
    if horizontals:
        for i in range(len(thisPool)):
            if np.sort(thisPool[i,:])[2] > -80:
                thisPool[i, :] = -80
    return thisPool


def filter_one_pool_by_percentile(pool, percentile_threshold):
    thisPool = pool.copy()
    for i in range(len(thisPool)):
        for j in range(len(thisPool[0])):
            if thisPool[i, j] < np.percentile(thisPool, percentile_threshold):
                thisPool[i,j] = -80
    return thisPool

def song_predict(array, column, columns, model):
    if column > len(array[0]) - 69:
        new_array = array[:, column:columns]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((623, 69))
        zero_array.fill(-80)
        col_diff = 69 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
        song_array = new_array.copy()
    else:
        song_array = array[:, column:column+69].copy()
    song_array = (song_array+80)*(255/80)
    song_array = song_array.reshape(-1, 623, 69, 1)
    x_min = song_array.min(axis=(1, 2), keepdims=True)
    x_max = song_array.max(axis=(1, 2), keepdims=True)
    song_array = (song_array - x_min)/(x_max-x_min)
    prediction = model.predict([song_array], verbose=0)
    return prediction[0][0]

def set_up(filename):
    data = fourier(filename)
    with h5py.File(filename + '.h5', 'w') as hf:
        hf.create_dataset(filename + "_dataset", data=data)
    with h5py.File(filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]

def get_times_and_freqs_lists(raven_table):
    sorted_raven_table = raven_table.sort_values(by=['Begin Time (s)'])
    init_begin_times = (sorted_raven_table['Begin Time (s)'].to_numpy())
    init_low_freqs = (sorted_raven_table['Low Freq (Hz)'].to_numpy())
    init_high_freqs = (sorted_raven_table['High Freq (Hz)'].to_numpy())
    begin_times = []
    mid_freqs = []
    #this removes post-intro notes by removing anything within a second of the previous.
    for i in range(len(init_begin_times)):
        if len(begin_times) > 0:
            if init_begin_times[i] - begin_times[-1] > 1:
                begin_times.append(init_begin_times[i])
                mid_freqs.append(np.mean([init_low_freqs[i], init_high_freqs[i]]))
        else:
            begin_times.append(init_begin_times[i])
            mid_freqs.append(np.mean([init_low_freqs[i], init_high_freqs[i]]))
    return begin_times, mid_freqs
    
def song_model_scores(data, filename, begin_time = 0, end_time = 0):
    start = datetime.now()
    last_check = start
    load_variables(data)
    print(np.shape(data))
    song_model = tf.keras.models.load_model('ml_songs.model')
    all_times = []
    all_scores = []
    i = int(begin_time / time_converter)
    start_col = i
    rows = len(data)
    if end_time > 0:
        columns = min(int(end_time / time_converter), len(data[0]))
    else:
        columns = len(data[0])
    print(rows, columns)
    while i < (columns):
        if np.max(data[:, i:i+30]) < intro_threshold:
            j = 0
            while j < 20:
                all_scores.append(0)
                all_times.append((i+j)*time_converter)
                j += 2
            i += 20
            # print("you saved time!")
            continue
        song_prediction = song_predict(data, i, columns, song_model)
        all_scores.append(song_prediction)
        all_times.append(i * time_converter)
        i += 2
        if ((datetime.now() - last_check).total_seconds() > 60):
            print(f"elapsed: {datetime.now() - start}. {round((i - start_col)/(columns - start_col)*100, 2)}% complete.")
            last_check = datetime.now()
    print(f'---------Stage 1 Complete!-------total time: {datetime.now() - start}')
    return np.array(all_times), np.array(all_scores)

def get_song_times_from_all(all_times, all_scores):
    strength_list = []
    song_time_list = []
    jump_ahead = 0
    plateau_length = 0
    plateau_start = 0
    for i in range(len(all_times) - 3):
        if all_scores[i] > .9:
            if plateau_length == 0:
                plateau_start = i
            plateau_length += 1
        else:
            if plateau_length >= 3:
                song_time_list.append(((i + plateau_start)/2) * (time_converter * 2))
                strength_list.append(plateau_length)
                #song_time_list.append(i * time_converter * 2)
            plateau_length = 0
    return np.array(song_time_list), np.array(strength_list)

def reinsert_weak_songs(all_times, all_scores, song_times):
    diffs = []
    for i in range(len(song_times)):
        if i > 0:
            diff = song_times[i] - song_times[i-1]
            diffs.append(diff)
        else:
            diffs.append(0)
    median_rate = np.median(diffs)
    jump_to_i = 0
    for i in range(len(all_times) - 20):
        if i > jump_to_i:
            for verified_time in song_times:
                if (abs(all_times[i] - verified_time) < median_rate - 1.5):
                    break
            else:
                for verified_time in song_times:
                    if i > 20 and (abs(all_times[i] - verified_time) < median_rate + 1):
                        if all_scores[i] > 0.2 and all_scores[i] >= np.max(all_scores[i-20:i+20]):
                            song_times.append(all_times[i])
                            print(f'found one at {all_times[i]}')
                            jump_to_i = i + 20
                            break
    return np.sort(song_times)


def display_spect(array):
    fig, ax = plt.subplots(figsize=(15, 7))
    # img = librosa.display.specshow(array[1], x_axis='time', y_axis=None, sr=22050, ax=ax)
    img = librosa.display.specshow(array, x_axis='time', y_axis=None, sr=22050, ax=ax)
    ax.set_title('Spectrogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    #fig.gca().set_yticks(range(0, 743-120, 25))
    fig.gca().set_ylabel("Row")
    plt.show()

#make raven table from the new song times and song_freqs
def make_raven_from_times_and_freqs(times, freqs, wave_start, wave_end, timestamp=datetime.now().strftime("%m-%d-%Y-%H-%M-%S")):
    selections = []
    for i in range(len(times)):
        selections.append([i+1, f'Spectrogram 1', 1, times[i], times[i] + .4, freqs[i]-400, freqs[i]+400])
    selection_df = pd.DataFrame(selections, columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)'])
    selection_df.to_csv('raven_' + str(wave_start) + '-' + str(wave_end) + '_' + timestamp + '_' + filename + '.txt', sep='\t', index=False)
    
def sec_to_min(num):
    return f'{int(num/60)}:{round(num % 60, 2)}'

#removes songs that are close to others with higher scores
def remove_bad_timing_songs(song_time_list, strength_list):
    #create time diff list
    new_list = []
    diffs = []
    for i in range(len(song_time_list)):
        if i > 0:
            diff = song_time_list[i] - song_time_list[i-1]
            diffs.append(diff)
        else:
            diffs.append(0)
    median_rate = np.median(diffs)
    timing_cutoff = median_rate * .6
    sandwich_cutoff = median_rate * .6
    
    for i in range(len(song_time_list)):      
        #remove sandwiched close songs by timing only
        if i > 0 and i < len(song_time_list) - 1:
            if diffs[i] < sandwich_cutoff and diffs[i+1] < sandwich_cutoff:
                print(f'sandwich at {song_time_list[i]}')
                continue
        #remove preceding close songs by timing and signal strength
        if i < len(song_time_list) -1:
            if diffs[i+1] < timing_cutoff and strength_list[i] <= 5 and strength_list[i+1] >= 6:
                print(f'preceding blip at {song_time_list[i]}')
                continue
        #remove following close songs by timing and signal strength
        if i > 0:
            if diffs[i] < timing_cutoff and strength_list[i] <= 5 and strength_list[i-1] >= 6:
                print(f'following blip at {song_time_list[i]}')
                continue
        new_list.append(song_time_list[i])
    return new_list
    
def find_loudest_box(array, width, height):
    norm_array = array + 80
    rows = len(array)
    cols = len(array[0])
    best_sum = 0
    best_i = 0
    best_j = 0
    for i in range(rows):
        for j in range(cols):
            if i < rows - height and j < cols - width:
                box_sum = np.sum(norm_array[i:i+height,j:j+width])
                # print(box_sum, i, j)
                if box_sum >= best_sum:
                    best_sum = box_sum
                    best_i = i
                    best_j = j
    best_i_perc = best_i/rows
    best_j_perc = best_j/cols
    return best_i, best_j, best_i_perc, best_j_perc

def filter_non_box(array, best_i, best_j, width, height):
    new_array = array.copy()
    rows = len(array)
    cols = len(array[0])
    for i in range(rows):
        for j in range(cols):
            if i < best_i or i >= best_i + height or j < best_j or j >= best_j + width:
                new_array[i, j] = -80
            # if i == best_i or i==best_i + height or j == best_j or j == best_j + width:
            #     new_array[i, j] = 0
    return new_array


#this intro note finder uses a window to move up and down the region searching for the brightest, leftmost, line.
def intro_note_finder(array, box_i_perc, box_j_perc, display=False):
    rows = len(array)
    cols = len(array[0])
    box_length = 10
    box_height = 15
    box_i = int(box_i_perc * rows)
    box_j = int(box_j_perc * cols)
    search_zone_bot = max(0, box_i - 30)
    search_zone_top = min(rows, box_i + 80)
    search_zone_left = max(0, box_j - 20)
    search_zone_right = box_j + 2 * box_length
    if display:
        print("filter paramaters: ", search_zone_bot, search_zone_top, search_zone_left, search_zone_right)
    best_window_score = -100000000
    best_i = 0
    best_j = 0
    mean_diffs = []
    line_vols = []
    scores = []
    js = []
    for i in range(search_zone_bot, search_zone_top - box_height):
        for j in range(search_zone_left, search_zone_right + 1 - box_length):
            window_array = array[i:i+box_height,j:j+box_length]
            if np.max(window_array[5:10,0]) > intro_threshold:
                mean_diffs.append(0)
                line_vols.append(0)
                scores.append(0)
                js.append(j)
                continue
            maxes = []
            col_diffs = []
            for col in range(box_length):
                maxes.append(np.max(window_array[5:10, col]) + 80)
            line_vols.append(np.mean(maxes))
            if np.median(maxes) < intro_threshold:
                mean_diffs.append(0)
                scores.append(0)
                js.append(j)
                continue
            for col in range(box_length):
                col_diffs.append( (np.max(window_array[5:10, col]) - np.median(window_array[11:15])) + (np.max(window_array[5:10, col]) - np.median(window_array[0:4])) )
            mean_diff = np.mean(col_diffs)
            mean_diffs.append(mean_diff/2)
            score = mean_diff - 0.7 * abs(j - (box_j - 15))
            scores.append(score)
            js.append(j)
            if score > best_window_score:
                best_window_score = score
                best_i = i
                best_j = j
    if display:
        plt.plot(mean_diffs)
        #plt.xlim([600, 800])
        plt.show()
        plt.close()
        plt.plot(line_vols)
        #plt.xlim([600, 800])
        plt.plot(scores)
        plt.plot(js)
        plt.show()
        plt.close()
        print('here be lengths', len(mean_diffs), len(scores), len(js))

    final_j = best_j
    max_i = np.argmax(array[best_i + 5: best_i + 10, final_j])
    final_i = best_i + 5 + max_i
    final_j_perc = final_j * cols
    final_i_perc = final_i * rows
    return final_i, final_j, final_i_perc, final_j_perc

def highlight_intro_note(array, intro_i, intro_j):
    new_array = array.copy()
    rows = len(array)
    cols = len(array[0])
    for i in range(rows):
        for j in range(cols):
            if (i == intro_i and j == intro_j):
                new_array[i, j] = 100
            else:
                new_array[i, j] = new_array[i, j] + 80
    return new_array

def highlight_big_intro_note(array, intro_i, intro_j):
    new_array = array.copy()
    rows = len(array)
    cols = len(array[0])
    for i in range(rows):
        for j in range(cols):
            if (i > intro_i - 10 and i < intro_i + 10 and j == intro_j):
                new_array[i, j] = 100
            else:
                new_array[i, j] = new_array[i, j] + 80
    return new_array

def convert_start_i_and_j(time_og, start_i, start_j):
    start_time_of_array = time_og - 20* time_converter
    row_ratio = 623/124
    new_row = int(start_i * row_ratio)
    time_of_note = start_time_of_array + start_j * time_converter
    start_freq = (new_row+120)*10.7666
    return new_row, start_j, start_freq, time_of_note

def get_start_times_and_freqs_from_pools_with_displays(data, song_time_list, time_value_list, display=False):
    shaved_list = []
    if len(time_value_list) > 0:
        for song_time in song_time_list:
            for compare_time in time_value_list:
                if abs(song_time - compare_time) < 1.5:
                    shaved_list.append(song_time)
    else:
        shaved_list = song_time_list
        
    start_times = []
    start_freqs = []
    for time in shaved_list:
        test_pool = post_pool(data, int(time/time_converter)-20, int(time/time_converter) + 70, len(data), len(data[0]), 5, 7)
        time_pool = post_pool(data, int(time/time_converter)-20, int(time/time_converter) + 70, len(data), len(data[0]), 5, 1)
        big_pool = data[:, int(time/time_converter)-20:int(time/time_converter + 70)]
        if display:
            # print(np.shape(test_pool_2))
            print(np.shape(test_pool))
            print(time)
            print(sec_to_min(time))

        filtered_pool = filter_one_pool_new(test_pool, 6, 4, True)    
        lightly_filtered_pool = filter_one_pool_new(time_pool, 5, 4, False)
        box_i, box_j, box_i_perc, box_j_perc = find_loudest_box(filtered_pool, 6, 60)
        only_box_pool = filter_non_box(filtered_pool, box_i, box_j, 6, 60)
        intro_i, intro_j, intro_i_perc, intro_j_perc = intro_note_finder(time_pool, box_i_perc, box_j_perc, display)            
        just_start_note_array = highlight_intro_note(time_pool, intro_i, intro_j)
        if display:
            print('--------------', box_i, box_j, intro_i_perc, intro_j_perc)
            display_spect(test_pool)
            display_spect(lightly_filtered_pool)
            display_spect(filtered_pool)
            display_spect(only_box_pool)
            display_spect(filter_one_pool_by_percentile(test_pool, 90))
            display_spect(just_start_note_array)
        big_i, big_j, song_freq, time_of_note = convert_start_i_and_j(time, intro_i, intro_j)
        #highlight_big_array = highlight_big_intro_note(big_pool, big_i, big_j)
        #display_spect(highlight_big_array)
        if display:
            print("song freq: ", song_freq, ". time_of_note: ", time_of_note, sec_to_min(time_of_note))
        print("time of note: ", time_of_note)
        start_times.append(time_of_note)
        start_freqs.append(song_freq)
    return np.array(start_times), np.array(start_freqs)

hethfinder_main()