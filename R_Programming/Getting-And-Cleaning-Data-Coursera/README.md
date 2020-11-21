**Getting-and-Cleaning-Data-Week-4-Assignment**  
This repository contains the code necessary to complete exercise 4 of the Getting And Cleaning Data Coursera course. Simply run the run_analysis.R file as is and the data should be downloaded, and the necessary output of tidydata.txt will be constructed.

**Data**  
The data linked to from the course website represents data collected from the accelerometers from the Samsung Galaxy S smartphone.

**run_analysis.R**  
This file does the following:

1. Merges the training and the test sets to create one data set.
2. Extracts only the measurements on the mean and standard deviation for each measurement.
3. Uses descriptive activity names to name the activities in the data set
4. Appropriately labels the data set with descriptive variable names.
5. From the data set in step 4, creates a second, independent tidy data set with the average of each variable for each activity and each subject.

The run_analysis.R file also downloads the necessary dataset zip file and unzips it in order to read in the necessary files.

**tidydata.txt**  
This text file contains the tidy dataset with the average of each variable for each activity and each subject.