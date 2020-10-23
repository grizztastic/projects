best <- function(state, outcome) {
    ## Read outcome data
    outcome1 <- read.csv("rprog_data_ProgAssignment3-data/outcome-of-care-measures.csv", colClasses = "character")
    ## Check that state and outcome are valid
    unique_states <- unique(outcome1$State)
    unique_outcomes <- c("heart attack", "heart failure", "pneumonia")
    if(!state %in% unique_states){
        stop("invalid state")}
    else if(!outcome %in% unique_outcomes){
        stop("invalid outcome")
    }
    ## Return hospital name in that state with lowest 30-day death
    ## rate
    hospital_in_state <- outcome1[outcome1$State == state,]
    ## Get the desired column to examine based on the outcome condition
    if (outcome == "heart attack") {
      col <- hospital_in_state$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack
      }
    else if (outcome == "heart failure") {
      col <- hospital_in_state$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure
    }
    else {
      col <- hospital_in_state$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia
    }
    ## Locate rows in which the desired column is at a minimum value
    min_info <- which(as.numeric(col) ==  min(as.numeric(col), na.rm = TRUE))
    best_hospital <- hospital_in_state[min_info, 2]
    ## Sort hospitals in alphabetical order for tie consideration
    best_hospital <- sort(best_hospital)
    return(best_hospital)

}