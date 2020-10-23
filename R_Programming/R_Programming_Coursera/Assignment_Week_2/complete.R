## complete reads a directory full of files and reports the number of completely 
## observed cases in each data file. This function then returns a data 
## frame where the first column is the name of the file and the second 
## column is the number of complete cases. 

complete <- function(directory, id = 1:332) {
      wd <- getwd()
      directory = paste(wd, "/", directory, sep = "")
      monitors <- list.files(path = directory)
      nobs <- numeric()
      for(monitor_id in id){ ## loop through the monitors reading in data
          monitor_data <- read.csv(paste("specdata/", monitors[monitor_id], sep=""))
          ## get the number of completed cases
          comp_cases_sum <- sum(complete.cases(monitor_data))
          nobs <- c(nobs, comp_cases_sum)
      }
      data.frame(id, nobs)
}