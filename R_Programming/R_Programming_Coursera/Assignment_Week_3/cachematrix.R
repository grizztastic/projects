## The makeCacheMatrix function does the following 4 things:
## 1. Sets the value of the matrix.
## 2. Gets the value of the matrix.
## 3. Sets the value of the inverse matrix.
## 4. Gets the value of the inverse matrix.
makeCacheMatrix <- function(x = matrix()) {
  inv_matrix <- NULL
  set_matrix <- function(y) { ## Refer to comment #1
    x <<- y
    inv_matrix <<- NULL
  }
  get_matrix <- function() x ## Refer tp comment #2
  setInverse <- function(inverse) inv_matrix <<- inverse ## Refer to comment #3
  getInverse <- function() inv_matrix ## Refer to comment #4
  list(set=set_matrix, get=get_matrix, setInverse=setInverse, getInverse=getInverse) ## Returns list of functions to be used
}

## The cacheSolve function calculates the inverse of the matrix created in the above function. It first
## checks to see if the inverse has already been calculated, and if it has, it gets the inverse from the 
## cache and skips the computation. If the inverse has not been calculated, it will calculate the inverse
## of the matrix and set the value in the cache via the setInverse function.
cacheSolve <- function(x, ...) {
  inv_matrix <- x$getInverse()
  if(!is.null(inv_matrix)) { ## Checks if inverse has already been calculated
    message("getting cached data")
    return(inv_matrix) ## Returns inverse matrix from cache
  }
  data <- x$get() ## Gets matrix 
  inv_matrix <- solve(data) ## Solves inverse of matrix
  x$setInverse(inv_matrix) ## Sets the inverse of matrix in cache
  inv_matrix 
}
