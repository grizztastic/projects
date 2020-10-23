## corr calculates the correlation between sulfate and nitrate for 
## monitor locations where the number of completely observed 
## cases (on all variables) is greater than the threshold.
corr <- function(directory, threshold = 0){
      monitors <- list.files(path = directory)
      df <- complete(directory)
      ids <- df[df["nobs"] > threshold, ]$id
      corr <- numeric()
      for(monitor_id in ids){
          monitor_data <- read.csv(monitors[monitor_id])
          dff <- monitor_data[complete.cases(monitor_data), ]
          corr <- c(corr, cor(dff$sulfate, dff$nitrate))
      }
      return(corr)
}