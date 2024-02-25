
# HethFinder
## _A program for automated selection of Hermit Thrush Song_

### Overview
HethFinder uses a machine learning model to search through wave files for hermit thrush song and label each song’s introductory note.

**Inputs**:
- .wav file(s) that contain hermit thrush song
- a hethfinder_params.csv file with wave files listed and (optional) start and stop times if you prefer to analyze only a portion of the file for faster, more accurate, performance. It is recommended that you preview the spectrogram beforehand and input a region where clear hermit thrush song occurs.

**Output**:
- .txt file(s) that is formatted to be opened in Raven as a selection table. HethFinder will attempt to identify each hermit thrush song by placing a box around the intro note portion of the song.

### Installation

Currently, HethFinder requires that python be installed on the user’s computer. HethFinder was developed on python version 3.9.12 but other versions are likely to be compatible. Other python libraries need to be installed as well, including tensorflow and librosa (the other imports such as numpy and pandas are sometimes included with a package manager). I recommend using anaconda to manage python and its dependencies. To download anaconda and learn more, go to: https://www.anaconda.com/download

### Usage

**Setup**
1.	Begin with a folder that contains the following files:
    i.	the wave file(s) to be analyzed.
    ii.	the HethFinder.py file.
    iii.	the hethfinder_params .csv file.
2.	Open hethfinder_parameters.csv and enter the .wav filename(s) into the filename column. Enter all filenames to be analyzed. Do not change anything else about this file, such as column names.
3.	(Optional) In corresponding columns, enter a start time and end time for each wave file. If either is left blank, HethFinder will assume that the start time is 0 and the end time is the end of the .wav file. HethFinder will run more quickly (and accurately) if it does not need to evaluate regions that are noisy or have little to no hermit thrush song.

**Run HethFinder**
1.	Open HethFinder.py in whatever code editor you use. Popular options are VSCode, Idle, or Jupyter Labs.
2.	Run HethFinder. It will take approximately as long as the selected recordings. For example, each ten-minute file will take approximately ten minutes to analyze. Set it and forget it! HethFinder will produce a .txt file for each .wav file it analyzes. These can be opened in Raven as selection tables. The text files could also very likely be modified to be opened in other acoustic analysis software. They are basically just a list of times (s) and frequencies (Hz).

**Review and Manually Edit**
1.	Open each new selection table and wave file in Raven. Check the selections for errors and fix them if need be. Depending on the research question and amount of error, editing may or may not be necessary. HethFinder accuracy varies depending on several factors, including signal strength, background noise, song-types, and presence of countersinging (for more information, see accuracy tests below).

### How HethFinder Works

HethFinder uses a combination of machine learning and post-processing. There are three stages that it runs through on each recording, described briefly below:

#### _1. Machine Learning Model_

First, HethFinder moves along the entire spectrogram of the wave file, taking pictures that are ~1.5 seconds long (a little longer than a HETH song) every 46 milliseconds. There is obviously a lot of overlap between these pictures. Imagine a camera with a 1.5s viewport sliding along the spectrogram from left to right rapidly taking pictures. Every snapshot is checked by the machine learning model and assigned a probability from 0 to 1. This number represents how likely it is that the snapshot contains a hermit thrush song beginning at the left edge of the frame.

#### _2. Post-Processing_

The ML model does not provide enough information to know precisely and accurately where HETH songs are. It merely provides moment-to-moment probabilities. The post-processing stage takes this list of probabilities and turns it into a list of timestamps. It considers the magnitude and pattern of the probabilities and chooses a timestamp if the probability is strong enough and timed in a way that makes sense. For example, it discards a signal that is too close in time to an even stronger signal, since HETH almost never sing in rapid succession.

The output of stage 2 is a list of timestamps corresponding to when HETH songs are likely to occur.

### 3. _Intro-note Selection_

If everything has gone well, each timestamp from stage 2 should be within ~500ms of an actual HETH song. The purpose of stage 3 is to narrow the temporal precision and find the frequency of the intro note. Like at stage 1, HethFinder again takes overlapping pictures. This time, it uses a smaller camera frame to sweep and filter the local region for the most acoustically active, hermit-thrush-like area, which should correspond to the post-introductory portion of the song.

It then uses an even smaller, short-but-wide camera frame to sweep up and down the area preceding the post-introductory portion, searching for an intro note. This is typically a long horizontal line on a spectrogram. HethFinder prioritizes the loudest line and the leftmost line, striking a balance between the two if they conflict. Once it chooses a line, it outputs the timestamp at the beginning of the line and the pitch frequency of the middle of the line.

The output of stage three is a raven selection table with timestamps and frequencies corresponding to each selected song.

### Contact and Additional Info:

**Contact:** HethFinder was created by Luke McLean (MA candidate) for the Sean Roach Laboratory at the University of New Brunswick – Saint John Campus. Inquiries about the software can be sent to lmclean@unb.ca 

**Future Versions:** It is my hope that I will continue to improve HethFinder as I use it and find time to work on it. If you’d like to help or have questions, don’t hesitate to contact me.

**HethSorter:**  Finding songs is usually only the first part of analyzing hermit thrush song. After HethFinder finds the songs, HethSorter sorts them into labelled song-types. So, if you have songs that need labelling, be sure to check out HethFinder’s best friend and companion, HethSorter.
