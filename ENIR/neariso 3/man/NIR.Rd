\name{neariso}
\alias{neariso}
\alias{Neariso}
\alias{nearisoGetSolution}
\alias{nearisoGetBreakpoints}
\title{Near-Isotonic Regression}
\description{
  These functions are the main interface functions for calculating Nearly-isotonic solutions
}
\usage{
neariso(y, maxBreaks=100, lambda=NULL)
nearisoGetBreakpoints(nearisoPathObj, maxBreaks=100)
nearisoGetSolution(nearisoPathObj, lambda=nearisoGetBreakpoints(nearisoPathObj))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{y}{ response variable; numeric }
  \item{lambda}{ penalty parameter vector (non-negative) for the difference of coefficients; numeric }
  \item{nearisoPathObj}{Solution object of class \code{nearisoPath} as returned by nir}
  \item{maxBreaks}{maximum number of breakpoints the function should return}
}
\details{
\code{neariso} is the main function to calculate a Nearly-isotonic regression fit and returns an object of class \code{nearisoPath}. If lambda=NULL, then the breakpoints of the linear solution path of beta are chosen, however at most \code{maxBreaks}. See \code{nirGetBreakpoints} for what happens if there are more than \code{maxBreaks} breakpoints.

\code{nearisoGetSolution} takes an object of class \code{nearisoPath} and returns an object of the same class, but with the solution calculated for the given value of \code{lambda}. Advantage in comparison to \code{neariso} is that it uses the already calculated solution and does not recompute the entire solution path, therefore being faster. 

\code{nearisoGetBreakpoints} returns the lambda values at which the piecewise linear solution paths for beta have a breakpoint. If there are more than \code{maxBreaks} such breakpoints, only \code{maxBreaks} representative breakpoints will be returned, including the first and last.
}
\value{
Returns a list with the items
    \item{solObj}{Object of class \code{nearisoSolObj} in which the whole solution path is saved in compact form; used as basis to recalculate fits with new values of \code{lambda} in \code{nearisoGetSolution}}
    \item{lambda}{Values of \code{lambda} for which the solution was calculated}
    \item{df}{Number of different values for beta in the solution; degrees of freedom; same length as \code{lambda}}
    \item{beta}{\code{length(y)} x \code{length(lambda)} matrix with the solution}
  }
\references{Ryan Tibshirani, Holger Hoefling, Rob
  Tibshirani. Nearly-isotonic regression. 2010. To appear in Technometrics.}
\examples{
library(neariso)
# generate some artificial data
y <- rnorm(1000) + (1:1000)/3000

### run the algorithm as default; will output solution at 100 breakpoints for lambda
res0 <- neariso(y)

### apply function nir and get solution directly
lambda = 0:10/10
res <- neariso(y, lambda=lambda)

### apply the function and get the solution later
res2 <- neariso(y, lambda=NULL)
res2 <- nearisoGetSolution(res2, lambda=lambda)

### look at the breakpoints
lambdaBreaks <- nearisoGetBreakpoints(res2, maxBreaks=1000)
res3 <- nearisoGetSolution(res2, lambda=lambdaBreaks)
}
\author{Holger Hoefling}
\keyword{regression}% at least one, from doc/KEYWORDS
\keyword{multivariate}





