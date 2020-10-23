## pollutantmean calculates the mean of a pollutant (sulfate or nitrate) across
## a specified list of monitors. The function 'pollutantmean' takes 
## three arguments: 'directory', 'pollutant', and 'id'. Given a vector 
## monitor ID numbers, 'pollutantmean' reads that monitors' particulate 
## matter data from the directory specified in the 'directory' argument 
## and returns the mean of the pollutant across all of the monitors, 
## ignoring any missing values coded as NA
pollutantmean <- function(directory, pollutant, id = 1:332){
        wd <- getwd()
        directory = paste(wd, "/", directory, sep = "")
        monitors <- list.files(path = directory)
        monitor_values <- numeric()
        for(monitor_id in id){ ## loop through all the monitors and get their data
            monitor_data <- read.csv(paste("specdata/", monitors[monitor_id], sep=""))
            monitor_values <- append(monitor_values, monitor_data[[pollutant]]) 
        }
        mean(monitor_values, na.rm = TRUE) ## get the mean of montior values list
}