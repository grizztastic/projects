rankall <- function(outcome, num = "best") {
    ## Read outcome data
    outcome1 <- read.csv("rprog_data_ProgAssignment3-data/outcome-of-care-measures.csv", colClasses = "character")
    ## Check that the outcomes are valid
    unique_outcomes <- c("heart attack", "heart failure", "pneumonia")
    if(!outcome %in% unique_outcomes){
      stop("invalid outcome")
    }
    ## Get the desired column to examine based on the outcome condition
    if (outcome == "heart attack") {
      col <- outcome1$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack
    }
    else if (outcome == "heart failure") {
      col <- outcome1$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure
    }
    else {
      col <- outcome1$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia
    }
    ## Sorting the hospital by state dataframe by the desired column values in ascending order and removing NA vals
    sorted_df <- outcome1[order(as.numeric(col), outcome1[,2], na.last = NA), ]
    split_df_by_state <- split(sorted_df, sorted_df$State)
    
    ## Split the sorted dataframe by state and loop over the entire list returning the desired hospital name at the num
    ans = lapply(split_df_by_state, function(x, num) {
        if(num == "best") {
            return (x$Hospital.Name[1])
        }
        else if(num == "worst") {
            return (x$Hospital.Name[nrow(x)])
        }
        else {
            return (x$Hospital.Name[num])
        }
    }, num)
    
    ## Return a data.frame with the proper form
    return (data.frame(hospital=unlist(ans), state=names(ans)) )
}