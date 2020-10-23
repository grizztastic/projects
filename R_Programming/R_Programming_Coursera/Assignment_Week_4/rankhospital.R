rankhospital <- function(state, outcome, num = "best") {
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
  ## Sorting the hospital by state dataframe by the desired column values in ascending order and removing NA vals
  sorted_df <- hospital_in_state[order(as.numeric(col), hospital_in_state[,2], na.last = NA), ]
  if(num == 'best'){
      num <- 1
  }
  else if(num == 'worst'){
      num <- nrow(sorted_df)
  }
  return(sorted_df[num, 2])
}