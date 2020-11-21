# The run_analysis.R file does the following:

# 1. Merges the training and the test sets to create one data set.
# 2. Extracts only the measurements on the mean and standard deviation for each measurement.
# 3. Uses descriptive activity names to name the activities in the data set
# 4. Appropriately labels the data set with descriptive variable names.
# 5. From the data set in step 4, creates a second, independent tidy data set with the average of each variable for each activity and each subject.

url <- 'https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip'
destfile <- 'UCI HAR Dataset.zip'

# Download the zip file if the zip file doesn't already exist in the directory
if (!file.exists(url)) {
  download.file(url, destfile, mode = "wb")
}

# Unzip the zipfile if it isn't already in the directory
path <- "UCI HAR Dataset"
if (!file.exists(path)) {
  unzip(destfile)
}

# Read in the training data labels, x, and y values
train_labels <- read.table(file.path(path, "train", "subject_train.txt"))
train_x <- read.table(file.path(path, "train", "X_train.txt"))
train_y <- read.table(file.path(path, "train", "y_train.txt"))

# Read in the training data labels, x, and y values
test_labels <- read.table(file.path(path, "test", "subject_test.txt"))
test_x <- read.table(file.path(path, "test", "X_test.txt"))
test_y <- read.table(file.path(path, "test", "y_test.txt"))

# Read in the features
features <- read.table(file.path(path, "features.txt"), as.is = TRUE)

# Read in the activity labels
activities <- read.table(file.path(path, "activity_labels.txt"))

# Rename the columns
colnames(activities) <- c("activityId", "activityLabel")

# Merge the train and test data for x, y, and label values
x_merged <- rbind(train_x, test_x)
y_merged <- rbind(train_y, test_y)
labels_merged <- rbind(train_labels, test_labels)

# Merge all data for complete dataset
complete_dataset <- cbind(labels_merged, x_merged, y_merged)

# Assign the column names to the complete dataset
colnames(complete_dataset) <- c("subject", features[, 2], "activity")

# Extract only the measurements on the mean and standard deviation for each measurement
complete_dataset <- complete_dataset[, grepl("subject|activity|mean|std", colnames(complete_dataset))]

# Use descriptive activity names to name the activities in the dataset
complete_dataset$activity <- factor(complete_dataset$activity, levels = activities[,1], labels = activities[,2]) 
complete_dataset$subject <- as.factor(complete_dataset$subject)

# Rename columns to be more descriptive
complete_data_cols <- colnames(complete_dataset)
complete_data_cols <- gsub("[\\(\\)-]", "", complete_data_cols)
complete_data_cols <- gsub("^f", "Freq_Domain_", complete_data_cols)
complete_data_cols <- gsub("^t", "Time_Domain_", complete_data_cols)
complete_data_cols <- gsub("Acc", "_Accelerometer_", complete_data_cols)
complete_data_cols <- gsub("Gyro", "_Gyroscope_", complete_data_cols)
complete_data_cols <- gsub("Mag", "_Magnitude_", complete_data_cols)
complete_data_cols <- gsub("Freq", "_Frequency_", complete_data_cols)
complete_data_cols <- gsub("mean", "_Mean_", complete_data_cols)
complete_data_cols <- gsub("std", "_Standard_Deviation_", complete_data_cols)
complete_data_cols <- gsub("BodyBody", "Body_", complete_data_cols)

colnames(complete_dataset) <- complete_data_cols

# Create a second, independent tidy set with the average of each variable for each activity and each subject
complete_data_means <- complete_dataset %>% 
  group_by(subject, activity) %>%
  summarise_each(funs(mean))

# Write to file "tidydata.txt"
write.table(complete_data_means, "tidydata.txt", row.names = FALSE, quote = FALSE)



